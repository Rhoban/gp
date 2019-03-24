#include "rhoban_gp/auto_tuning.h"

#include "rhoban_random/tools.h"

#include <iostream>

namespace rhoban_gp
{
void runSimpleGradientAscent(GaussianProcess& gp, const Eigen::VectorXd& initial_guess, const Eigen::VectorXd& gamma,
                             double epsilon)
{
  Eigen::VectorXd last_guess = Eigen::VectorXd::Constant(initial_guess.rows(), -1);
  Eigen::VectorXd guess = initial_guess;
  while ((last_guess - guess).squaredNorm() > epsilon)
  {
    last_guess = guess;
    gp.setParameters(guess);
    Eigen::VectorXd gradient = gp.getMarginalLikelihoodGradient();
    std::cout << "guess: " << guess.transpose() << std::endl << "\tgradient: " << gradient.transpose() << std::endl;
    // Dirty avoid to reach negative numbers for parameters
    Eigen::VectorXd gain = gamma;
    for (int i = 0; i < gradient.rows(); i++)
    {
      if (last_guess(i) + gradient(i) * gain(i) < 0)
      {
        gain(i) = -last_guess(i) / gradient(i) * 0.95;
      }
    }
    guess = last_guess + gain.cwiseProduct(gradient);
  }
  gp.setParameters(guess);
}

Eigen::VectorXd cwiseSign(const Eigen::VectorXd& v)
{
  Eigen::VectorXd result(v.rows());
  for (int i = 0; i < v.rows(); i++)
  {
    if (v(i) > 0)
    {
      result(i) = 1;
    }
    else if (v(i) < 0)
    {
      result(i) = -1;
    }
    else
    {
      result(i) = 0;
    }
  }
  return result;
}

Eigen::VectorXd rProp(std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func,
                      const Eigen::VectorXd& initial_guess, const Eigen::VectorXd& initial_step_size,
                      const Eigen::MatrixXd& limits, double epsilon, int max_nb_guess, double eta_pos, double eta_neg)
{
  if (initial_step_size.minCoeff() < 0)
  {
    throw std::runtime_error("rhoban_gp::rProp: negative coeff in initial_step_size forbidden");
  }
  // Initializing variables
  Eigen::VectorXd guess = initial_guess;
  Eigen::VectorXd step_size = initial_step_size;
  Eigen::VectorXd min_guess = limits.col(0);
  Eigen::VectorXd max_guess = limits.col(1);
  // Computing gradient at initial guess
  Eigen::VectorXd gradient = gradient_func(guess);
  // std::cout << "gradout " << gradient.transpose() << std::endl;
  Eigen::VectorXd grad_signs = cwiseSign(gradient);

  Eigen::VectorXd last_guess, last_gradient, last_grad_signs;
  // Break in the middle of the loop to avoid code duplication
  int nb_guess = 0;
  while (true)
  {
    // Update guess
    last_guess = guess;
    Eigen::VectorXd delta_guess = grad_signs.cwiseProduct(step_size);
    guess = guess + delta_guess;
    // Ensure guess does not go outside of limits
    guess = guess.cwiseMin(max_guess).cwiseMax(min_guess);
    // Apply new guess
    // BREAK CONDITION: diff lower than epsilon
    if ((last_guess - guess).cwiseAbs().maxCoeff() < epsilon)
      break;
    //    if (nb_guess > max_nb_guess - 10){
    //      std::cout << "guess " << nb_guess << std::endl
    //                << "\tguess: " << guess.transpose() << std::endl
    //                << "\tdelta: " << delta_guess.transpose() << std::endl
    //                << "\tgrad : " << gradient.transpose() << std::endl
    //                << "\tgrad_signs : " << grad_signs.transpose() << std::endl
    //                << "\tstep_size : " << step_size.transpose() << std::endl
    //                << "\tepsilon : " << epsilon << std::endl
    //                << "\tdiff : " << (last_guess - guess).transpose() << std::endl;
    //    }
    if (nb_guess > max_nb_guess)
    {
      std::cerr << "breaking rProp after " << nb_guess << std::endl;
      break;
    }
    nb_guess++;
    // Update gradient
    last_gradient = gradient;
    last_grad_signs = grad_signs;
    gradient = gradient_func(guess);
    grad_signs = cwiseSign(gradient);
    // Update step size
    for (int i = 0; i < step_size.rows(); i++)
    {
      // If sign changed, reduce step size
      if (grad_signs(i) * last_grad_signs(i) < 0)
      {
        step_size(i) = step_size(i) * eta_neg;
      }
      // If signd did not change, increase step size
      else if (grad_signs(i) * last_grad_signs(i) > 0)
      {
        step_size(i) = step_size(i) * eta_pos;
      }
    }
  }
  return guess;
}

Eigen::VectorXd rProp(GaussianProcess& gp, const Eigen::VectorXd& initial_guess,
                      const Eigen::VectorXd& initial_step_size, const Eigen::MatrixXd& limits, double epsilon,
                      int max_nb_guess, double eta_pos, double eta_neg)
{
  std::function<Eigen::VectorXd(const Eigen::VectorXd&)> grad_func = [&gp](const Eigen::VectorXd& guess) {
    gp.setParameters(guess);
    return gp.getMarginalLikelihoodGradient();
  };

  Eigen::VectorXd final_guess =
      rProp(grad_func, initial_guess, initial_step_size, limits, epsilon, max_nb_guess, eta_pos, eta_neg);
  gp.setParameters(final_guess);
  return final_guess;
}

Eigen::VectorXd randomizedRProp(GaussianProcess& gp, const Eigen::MatrixXd& limits, double epsilon, int nb_trials,
                                int max_nb_guess, double eta_pos, double eta_neg)
{
  std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func;
  std::function<double(const Eigen::VectorXd)> score_func;
  gradient_func = [&gp](const Eigen::VectorXd& guess) {
    gp.setParameters(guess);
    return gp.getMarginalLikelihoodGradient();
  };
  score_func = [&gp](const Eigen::VectorXd& guess) {
    gp.setParameters(guess);
    return gp.getLogMarginalLikelihood();
  };
  Eigen::VectorXd best_guess;
  best_guess = randomizedRProp(gradient_func, score_func, limits, epsilon, nb_trials, max_nb_guess, eta_pos, eta_neg);
  return best_guess;
}

Eigen::VectorXd randomizedRProp(std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func,
                                std::function<double(const Eigen::VectorXd)> scoring_func,
                                const Eigen::MatrixXd& limits, double epsilon, int nb_trials, int max_nb_guess,
                                double eta_pos, double eta_neg)
{
  // Defining minimal and maximal initial step sizes
  Eigen::MatrixXd step_size_limits(limits.rows(), limits.cols());
  step_size_limits.col(0) = Eigen::VectorXd::Constant(limits.rows(), epsilon);
  step_size_limits.col(1) = (limits.col(1) - limits.col(0)) / 100;
  // Creating random initial guesses and random initial steps
  Eigen::MatrixXd initial_guesses;
  Eigen::MatrixXd initial_step_sizes;
  initial_guesses = rhoban_random::getUniformSamplesMatrix(limits, nb_trials);
  initial_step_sizes = rhoban_random::getUniformSamplesMatrix(step_size_limits, nb_trials);
  // Preparing common data
  double best_value = std::numeric_limits<double>::lowest();
  Eigen::VectorXd best_guess = (limits.col(0) + limits.col(1)) / 2;
  // Running several rProp optimization with different starting points
  for (int trial = 0; trial < nb_trials; trial++)
  {
    Eigen::VectorXd current_guess;
    current_guess = rhoban_gp::rProp(gradient_func, initial_guesses.col(trial), initial_step_sizes.col(trial), limits,
                                     epsilon, max_nb_guess, eta_pos, eta_neg);
    double value = scoring_func(current_guess);
    if (value > best_value)
    {
      best_value = value;
      best_guess = current_guess;
    }
  }
  return best_guess;
}

}  // namespace rhoban_gp
