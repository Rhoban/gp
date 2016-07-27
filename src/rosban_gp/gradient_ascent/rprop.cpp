#include "rosban_gp/gradient_ascent/rprop.h"

/// Not available in all Eigen versions
static Eigen::VectorXd cwiseSign(const Eigen::VectorXd & v)
{
  Eigen::VectorXd result(v.rows());
  for (int i = 0; i < v.rows(); i++)
  {
    if (v(i) > 0) {
      result(i) = 1;
    }
    else if (v(i) < 0) {
      result(i) = -1;
    }
    else {
      result(i) = 0;
    }
  }
  return result;
}
namespace rosban_gp
{

RProp::Config::Config()
  : epsilon(std::pow(10,-6)),
    max_iterations(1000),
    eta_pos(1.2),
    eta_neg(0.5),
    tuning_space(TuningSpace::Normal)
{
}

std::string RProp::Config::class_name() const
{
  return "rprop_config";
}

void RProp::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>   ("max_iterations", max_iterations, out);
  rosban_utils::xml_tools::write<double>("epsilon"       , epsilon       , out);
  rosban_utils::xml_tools::write<double>("eta_pos"       , eta_pos       , out);
  rosban_utils::xml_tools::write<double>("eta_neg"       , eta_neg       , out);
  rosban_utils::xml_tools::write<std::string>("tuning_space", to_string(tuning_space), out);
}

void RProp::Config::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>   (node, "max_iterations", max_iterations);
  rosban_utils::xml_tools::try_read<double>(node, "epsilon"       , epsilon       );
  rosban_utils::xml_tools::try_read<double>(node, "eta_pos"       , eta_pos       );
  rosban_utils::xml_tools::try_read<double>(node, "eta_neg"       , eta_neg       );
  std::string tuning_space_str;
  rosban_utils::xml_tools::try_read<std::string>(node, "tuning_space", tuning_space_str);
  if (tuning_space_str.size() > 0) { tuning_space = loadTuningSpace(tuning_space_str); }
}

int RProp::Config::write(std::ostream & out) const
{
  int bytes_written = 0;
  bytes_written += rosban_utils::write<int>   (out, max_iterations);
  bytes_written += rosban_utils::write<double>(out, epsilon);
  bytes_written += rosban_utils::write<double>(out, eta_pos);
  bytes_written += rosban_utils::write<double>(out, eta_neg);
  char tmp = static_cast<char>(tuning_space);
  bytes_written += rosban_utils::write<char>  (out, tmp);
  return bytes_written;
}

int RProp::Config::read(std::istream & in)
{
  int bytes_read = 0;
  bytes_read += rosban_utils::read<int>   (in, &max_iterations);
  bytes_read += rosban_utils::read<double>(in, &epsilon);
  bytes_read += rosban_utils::read<double>(in, &eta_pos);
  bytes_read += rosban_utils::read<double>(in, &eta_neg);
  char tmp;
  bytes_read += rosban_utils::read<char>  (in, &tmp);
  tuning_space = static_cast<RProp::TuningSpace>(tmp);
  return bytes_read;
}

Eigen::VectorXd RProp::run(std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func,
                           const Eigen::VectorXd & initial_guess,
                           const Eigen::VectorXd & initial_step_size,
                           const Eigen::MatrixXd & limits,
                           std::shared_ptr<Config> conf)
{
  // Check validity of initial_step
  if (initial_step_size.minCoeff() < 0) {
    throw std::runtime_error("rosban_gp::rProp: negative coeff in initial_step_size forbidden");
  }
  // Initialize config if necessary
  if (!conf) conf = std::shared_ptr<Config>(new Config());
  // Initializing variables
  Eigen::VectorXd guess = cvtToTuningSpace(initial_guess, conf);
  Eigen::VectorXd step_size = cvtToTuningSpace(initial_guess + initial_step_size, conf) - guess;
  Eigen::VectorXd min_guess = cvtToTuningSpace(limits.col(0), conf);
  Eigen::VectorXd max_guess = cvtToTuningSpace(limits.col(1), conf);
  // Computing gradient at initial guess (back in user space)
  Eigen::VectorXd gradient = gradient_func(cvtFromTuningSpace(guess, conf));
  Eigen::VectorXd grad_signs = cwiseSign(gradient);
  // Loop variables
  Eigen::VectorXd last_guess, last_gradient, last_grad_signs;
  int nb_guess = 0;
  // Break in the middle of the loop to avoid code duplication
  while (true){
    // Update guess
    last_guess = guess;
    Eigen::VectorXd delta_guess = grad_signs.cwiseProduct(step_size);
    guess =  guess + delta_guess;
    // Ensure guess does not go outside of limits
    guess = guess.cwiseMin(max_guess).cwiseMax(min_guess);
    // Apply new guess
    // BREAK CONDITION: diff lower than epsilon or nb_guesses > max_iterations
    if ((last_guess-guess).cwiseAbs().maxCoeff() < conf->epsilon) break;
    if (nb_guess > conf->max_iterations)
    {
      //std::cerr << "breaking rProp after " << nb_guess << std::endl;
      break;
    }
    nb_guess++;
    // Update gradient
    last_gradient = gradient;
    last_grad_signs = grad_signs;
    gradient = gradient_func(cvtFromTuningSpace(guess, conf));
    grad_signs = cwiseSign(gradient);
    // Update step size
    for (int i = 0; i < step_size.rows(); i++) {
      // If sign changed, reduce step size
      if (grad_signs(i) * last_grad_signs(i) < 0) {
        step_size(i) = step_size(i) * conf->eta_neg;
      }
      // If sign did not change, increase step size
      else if (grad_signs(i) * last_grad_signs(i) > 0) {
        step_size(i) = step_size(i) * conf->eta_pos;
      }
    }
  }
  // Answer is provided in the user space
  return cvtFromTuningSpace(guess, conf);
}

Eigen::VectorXd RProp::cvtFromTuningSpace(const Eigen::VectorXd & v,
                                          std::shared_ptr<Config> conf)
{
  Eigen::VectorXd result = v;
  if (conf->tuning_space == TuningSpace::Normal) return result;
  for (int row = 0; row < result.rows(); row++) {
    result(row) = std::exp(result(row));
  }
  return result;
}

Eigen::VectorXd RProp::cvtToTuningSpace(const Eigen::VectorXd & v,
                                        std::shared_ptr<Config> conf)
{
  Eigen::VectorXd result = v;
  if (conf->tuning_space == TuningSpace::Normal) return result;
  for (int row = 0; row < result.rows(); row++) {
    result(row) = std::log(result(row));
  }
  return result;
}

std::string to_string(RProp::TuningSpace tuning_space)
{
  switch (tuning_space)
  {
    case RProp::TuningSpace::Normal: return "Normal";
    case RProp::TuningSpace::Log: return "Log";
  }
  throw std::runtime_error("Unknown tuning_space in to_string(RProp::TuningSpace)");
}

RProp::TuningSpace loadTuningSpace(const std::string & tuning_space)
{
  if (tuning_space == "Normal")
  {
    return RProp::TuningSpace::Normal;
  }
  if (tuning_space == "Log")
  {
    return RProp::TuningSpace::Log;
  }
  throw std::runtime_error("Unknown RProp TuningSpace: '" + tuning_space + "'");
}

}
