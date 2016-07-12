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
    eta_neg(0.5)
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
}

void RProp::Config::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>   (node, "max_iterations", max_iterations);
  rosban_utils::xml_tools::try_read<double>(node, "epsilon"       , epsilon       );
  rosban_utils::xml_tools::try_read<double>(node, "eta_pos"       , eta_pos       );
  rosban_utils::xml_tools::try_read<double>(node, "eta_neg"       , eta_neg       );
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
  Eigen::VectorXd guess = initial_guess;
  Eigen::VectorXd step_size = initial_step_size;
  Eigen::VectorXd min_guess = limits.col(0);
  Eigen::VectorXd max_guess = limits.col(1);
  // Computing gradient at initial guess
  Eigen::VectorXd gradient = gradient_func(guess);
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
    gradient = gradient_func(guess);
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
  return guess;
}

}
