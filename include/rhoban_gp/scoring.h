#pragma once

#include <Eigen/Core>

namespace rhoban_gp
{

/// Compute the mean squared error
/// MSE = avg( (observation - prediction)^2)
double computeMSE(const Eigen::VectorXd & observations,
                  const Eigen::VectorXd & predictions);

/// Compute the standardized mean squared error
/// SMSE = MSE / Var(observations) 
double computeSMSE(const Eigen::VectorXd & observations,
                   const Eigen::VectorXd & predictions);


/// Compute the mean standardized log loss
/// MSLL = avg(SLL)
double computeMSLL(const Eigen::MatrixXd & inputs,
                   const Eigen::VectorXd & observations,
                   const Eigen::VectorXd & predictions);
}
