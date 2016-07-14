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
  rprop_conf->tryRead(node, "rprop_conf");
}

// NOTE: implementation of tuning space is quite weird in order to avoid
//       breaking API.
// TODO: Think about another way once tests have been led
Eigen::VectorXd RandomizedRProp::run(RProp::GradientFunc gradient_func,
                                     ScoringFunc scoring_func,
                                     const Eigen::MatrixXd & limits,
                                     const Config & conf)
{
  // DEBUG
  //std::cout << "auto-tuning with: " << std::endl
  //          << "\tnb_trials     : " << conf.nb_trials      << std::endl
  //          << "\tmax_iterations: " << conf.rprop_conf->max_iterations << std::endl
  //          << "\tepsilon       : " << conf.rprop_conf->epsilon        << std::endl;

  // Get limits inside tuning space
  Eigen::MatrixXd tuning_space_limits = limits;
  tuning_space_limits.col(0) = RProp::cvtToTuningSpace(limits.col(0), conf.rprop_conf);
  tuning_space_limits.col(1) = RProp::cvtToTuningSpace(limits.col(1), conf.rprop_conf);

  // Defining minimal and maximal initial step sizes (in tuning_space
  Eigen::MatrixXd step_size_limits(limits.rows(), limits.cols());
  step_size_limits.col(0) = Eigen::VectorXd::Constant(limits.rows(), conf.rprop_conf->epsilon);
  // Max is 1/100th of the total range
  step_size_limits.col(1) = (tuning_space_limits.col(1) - tuning_space_limits.col(0) ) / 100;
  // Creating random initial guesses and random initial steps (in tuning space)
  Eigen::MatrixXd initial_guesses;
  Eigen::MatrixXd initial_step_sizes;
  initial_guesses    = rosban_random::getUniformSamplesMatrix(tuning_space_limits, conf.nb_trials);
  initial_step_sizes = rosban_random::getUniformSamplesMatrix(step_size_limits, conf.nb_trials);
  // Preparing common data
  double best_value = std::numeric_limits<double>::lowest();
  Eigen::VectorXd best_guess = (limits.col(0) + limits.col(1)) / 2;
  // Running several rProp optimization with different starting points
  for (int trial = 0; trial < conf.nb_trials; trial++) {
    Eigen::VectorXd initial_guess, initial_step_size, tmp;
    // Going back to original space for legacy reasons: QUICK AND DIRTY
    initial_guess = RProp::cvtFromTuningSpace(initial_guesses.col(trial), conf.rprop_conf);
    tmp = initial_guesses.col(trial) + initial_step_sizes.col(trial);
    initial_step_size = RProp::cvtFromTuningSpace(tmp, conf.rprop_conf) - initial_guess;
    // Computing best_guess
    Eigen::VectorXd current_guess;
    current_guess = RProp::run(gradient_func,
                               initial_guess,
                               initial_step_size,
                               limits,
                               conf.rprop_conf);
    double value = scoring_func(current_guess);
    if (value > best_value) {
      best_value = value;
      best_guess = current_guess;
    }
  }
  return best_guess;
}

}
