#pragma once

#include "rosban_gp/core/gaussian_process.h"

namespace rosban_gp
{

void runSimpleGradientAscent(GaussianProcess & gp,
                             const Eigen::VectorXd & initial_guess,
                             const Eigen::VectorXd & gamma,
                             double epsilon);

/// eta_pos and eta_neg values are set according to default value from wikipedia
Eigen::VectorXd rProp(std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func,
                      const Eigen::VectorXd & initial_guess,
                      const Eigen::VectorXd & initial_step,
                      const Eigen::MatrixXd & limits,
                      double epsilon,
                      int max_nb_guess = 1000,
                      double eta_pos = 1.2,
                      double eta_neg = 0.5);

/// eta_pos and eta_neg values are set according to default value from wikipedia
Eigen::VectorXd rProp(GaussianProcess & gp,
                      const Eigen::VectorXd & initial_guess,
                      const Eigen::VectorXd & initial_step,
                      const Eigen::MatrixXd & limits,
                      double epsilon,
                      int max_nb_guess = 1000,
                      double eta_pos = 1.2,
                      double eta_neg = 0.5);

}
