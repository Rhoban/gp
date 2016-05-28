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
                      const Eigen::VectorXd & initial_step_size,
                      const Eigen::MatrixXd & limits,
                      double epsilon,
                      int max_nb_guess = 1000,
                      double eta_pos = 1.2,
                      double eta_neg = 0.5);

/// eta_pos and eta_neg values are set according to default value from wikipedia
/// Tune up the hyper parameters of the function and return the best found
Eigen::VectorXd rProp(GaussianProcess & gp,
                      const Eigen::VectorXd & initial_guess,
                      const Eigen::VectorXd & initial_step_size,
                      const Eigen::MatrixXd & limits,
                      double epsilon,
                      int max_nb_guess = 1000,
                      double eta_pos = 1.2,
                      double eta_neg = 0.5);

/// Run nb_trials of rProp with random initial values inside limits, return the best solution
/// Tune up the hyper parameters of the function and return the best found
Eigen::VectorXd randomizedRProp(GaussianProcess & gp,
                                const Eigen::MatrixXd & limits,
                                double epsilon,
                                int nb_trials,
                                int max_nb_guess = 1000,
                                double eta_pos = 1.2,
                                double eta_neg = 0.5);

/// Run nb_trials of rProp with random initial values inside limits, return the best solution
/// according to the scoring_func
Eigen::VectorXd randomizedRProp(std::function<Eigen::VectorXd(const Eigen::VectorXd)> gradient_func,
                                std::function<double(const Eigen::VectorXd)> scoring_func,
                                const Eigen::MatrixXd & limits,
                                double epsilon,
                                int nb_trials,
                                int max_nb_guess = 1000,
                                double eta_pos = 1.2,
                                double eta_neg = 0.5);

}
