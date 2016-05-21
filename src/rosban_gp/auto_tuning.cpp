#include "rosban_gp/auto_tuning.h"

#include <iostream>

namespace rosban_gp
{

void runSimpleGradientAscent(GaussianProcess & gp,
                             const Eigen::VectorXd & initial_guess,
                             const Eigen::VectorXd & gamma,
                             double epsilon)
{
  Eigen::VectorXd last_guess = Eigen::VectorXd::Constant(initial_guess.rows(), -1);
  Eigen::VectorXd guess = initial_guess;
  while ((last_guess - guess).squaredNorm() > epsilon)
  {
    last_guess = guess;
    gp.setParameters(guess);
    Eigen::VectorXd gradient = gp.getMarginalLikelihoodGradient();
    std::cout << "guess: " << guess.transpose() << std::endl
              << "\tgradient: " << gradient.transpose() << std::endl;
    // Dirty avoid to reach negative numbers for parameters
    Eigen::VectorXd gain = gamma;
    for (int i = 0; i < gradient.rows(); i++)
    {
      if (last_guess(i) + gradient(i) * gain(i) < 0)
      {
        gain(i) = - last_guess(i) / gradient(i) * 0.95;
      }
    }
    guess = last_guess + gain.cwiseProduct(gradient);
  }
  gp.setParameters(guess);
}

Eigen::VectorXd cwiseSign(const Eigen::VectorXd v)
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

void rProp(GaussianProcess & gp,
           const Eigen::VectorXd & initial_guess,
           const Eigen::VectorXd & initial_step,
           const Eigen::MatrixXd & limits,
           double epsilon,
           double eta_pos,
           double eta_neg)
{
  // Initializing variables
  Eigen::VectorXd guess = initial_guess;
  Eigen::VectorXd step_size = initial_step;
  Eigen::VectorXd min_guess = limits.col(0);
  Eigen::VectorXd max_guess = limits.col(1);
  // Computing gradient at initial guess
  gp.setParameters(guess);
  Eigen::VectorXd gradient = gp.getMarginalLikelihoodGradient();
  Eigen::VectorXd grad_signs = cwiseSign(gradient);
  
  Eigen::VectorXd last_guess, last_gradient, last_grad_signs;
  // Break in the middle of the loop to avoid code duplication
  int nb_guess = 0;
  int max_nb_guess = 100;
  while (true){
    // Update guess
    last_guess = guess;
    Eigen::VectorXd delta_guess = grad_signs.cwiseProduct(step_size);
    //std::cout << "guess: " << guess.transpose() << std::endl
    //          << "\tgradient: " << gradient.transpose() << std::endl
    //          << "\tdelta: " << delta_guess.transpose() << std::endl;
    guess =  guess + delta_guess;
    // Ensure guess does not go outside of limits
    guess = guess.cwiseMin(max_guess).cwiseMax(min_guess);
    // Apply new guess
    gp.setParameters(guess);
    // BREAK CONDITION: diff lower than epsilon
    if ((last_guess-guess).cwiseAbs().maxCoeff() < epsilon) break;
    if (nb_guess > max_nb_guess)
    {
      std::cerr << "breaking rProp after " << nb_guess << std::endl;
      break;
    }
    // Update gradient
    last_gradient = gradient;
    last_grad_signs = grad_signs;
    gradient = gp.getMarginalLikelihoodGradient();
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

}

}
