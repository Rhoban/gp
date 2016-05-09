#pragma once

#include "rosban_gp/core/covariance_function.h"

namespace rosban_gp
{

/// Implement the following covariance function:
/// f(x,x') = process_noise * e^{-0.5 * sum (|x_i - x'_i|/length_scale)^2}
class SquaredExponential : public CovarianceFunction
{
public:

  SquaredExponential();
  /// Easy access for uni-dimensional input
  SquaredExponential(double length_scale, double process_noise);
  /// Multi-dimensional input
  SquaredExponential(const Eigen::VectorXd & length_scales, double process_noise);

  double compute(const Eigen::VectorXd & x1, const Eigen::VectorXd & x2) const override;

  Eigen::VectorXd computeGradient(const Eigen::VectorXd & x1,
                                  const Eigen::VectorXd & x2) const override;

private:
  Eigen::VectorXd length_scales;
  double process_noise;
};

}
