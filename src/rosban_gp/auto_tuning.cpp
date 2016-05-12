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
    // Dirty hack: avoid to reach negative numbers for parameters
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

}
