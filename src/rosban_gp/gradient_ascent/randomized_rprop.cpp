#include "rosban_gp/gradient_ascent/randomized_rprop.h"

#include "rosban_random/tools.h"

namespace rosban_gp
{

RandomizedRProp::Config::Config()
  : nb_trials(10),
    rprop_conf(new RProp::Config())
{
}

std::string RandomizedRProp::Config::class_name() const
{
  return "randomized_rprop_config";
}

void RandomizedRProp::Config::to_xml(std::ostream &out) const
{
  rosban_utils::xml_tools::write<int>("nb_trials", nb_trials, out);
  rprop_conf->write("rprop_conf", out);
}

void RandomizedRProp::Config::from_xml(TiXmlNode *node)
{
  rosban_utils::xml_tools::try_read<int>   (node, "nb_trials", nb_trials);
  // If other uses this shared_ptr, avoid to overwrite their data
  rprop_conf = std::shared_ptr<RProp::Config>(new RProp::Config());
  rprop_conf->read(node, "rprop_conf");
}

Eigen::VectorXd RandomizedRProp::run(RProp::GradientFunc gradient_func,
                                     ScoringFunc scoring_func,
                                     const Eigen::MatrixXd & limits,
                                     std::shared_ptr<Config> conf)
{
  // Create default conf if none is provided
  if (!conf) conf = std::shared_ptr<Config>(new Config());
  // Defining minimal and maximal initial step sizes
  Eigen::MatrixXd step_size_limits(limits.rows(), limits.cols());
  step_size_limits.col(0) = Eigen::VectorXd::Constant(limits.rows(), conf->rprop_conf->epsilon);
  step_size_limits.col(1) = (limits.col(1) - limits.col(0) ) / 100;//100 should go to a variable
  // Creating random initial guesses and random initial steps
  Eigen::MatrixXd initial_guesses;
  Eigen::MatrixXd initial_step_sizes;
  initial_guesses    = rosban_random::getUniformSamplesMatrix(limits, conf->nb_trials);
  initial_step_sizes = rosban_random::getUniformSamplesMatrix(step_size_limits, conf->nb_trials);
  // Preparing common data
  double best_value = std::numeric_limits<double>::lowest();
  Eigen::VectorXd best_guess = (limits.col(0) + limits.col(1)) / 2;
  // Running several rProp optimization with different starting points
  for (int trial = 0; trial < conf->nb_trials; trial++) {
    Eigen::VectorXd current_guess;
    current_guess = RProp::run(gradient_func,
                               initial_guesses.col(trial),
                               initial_step_sizes.col(trial),
                               limits,
                               conf->rprop_conf);
    double value = scoring_func(current_guess);
    if (value > best_value) {
      best_value = value;
      best_guess = current_guess;
    }
  }
  return best_guess;
}

}
