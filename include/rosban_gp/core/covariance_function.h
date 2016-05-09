#pragma once

#include <Eigen/Core>

namespace rosban_gp
{

class CovarianceFunction
{
public:
  /// Compute covar(x,x)
  double compute(const Eigen::VectorXd & x) const;

  /// Compute covar(x1,x2)
  virtual double compute(const Eigen::VectorXd & x1, const Eigen::VectorXd & x2) const = 0;

  /// Compute the gradient of covar(x1,x2) with respect to the function parameters
  virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd & x1,
                                          const Eigen::VectorXd & x2) const = 0;



  /// Build a covariance Matrix K such as:
  /// K_{i,j} = covar_func(inputs.col(i), inputs.col(j))
  Eigen::MatrixXd buildMatrix(const Eigen::MatrixXd & inputs) const;

  /// Build a covariance Matrix K such as:
  /// K_{i,j} = covar_func(inputs_a.col(i), inputs_b.col(j))
  Eigen::MatrixXd buildMatrix(const Eigen::MatrixXd & inputs_a,
                              const Eigen::MatrixXd & inputs_b) const;
};

}
