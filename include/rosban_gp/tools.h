#pragma once

#include <Eigen/Core>

#include <functional>

namespace rosban_gp
{
typedef std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)> CovarianceFunction;

/// Build a covariance Matrix K such as:
/// K_{i,j} = covar_func(inputs.col(i), inputs.col(j))
Eigen::MatrixXd buildCovarianceMatrix(const Eigen::MatrixXd & inputs,
                                      CovarianceFunction covar_func);

/// Build a covariance Matrix K such as:
/// K_{i,j} = covar_func(inputs_a.col(i), inputs_b.col(j))
Eigen::MatrixXd buildCovarianceMatrix(const Eigen::MatrixXd & inputs_a,
                                      const Eigen::MatrixXd & inputs_b,
                                      CovarianceFunction covar_func);

}
