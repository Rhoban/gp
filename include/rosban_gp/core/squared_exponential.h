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

  virtual ~SquaredExponential();

  virtual CovarianceFunction * clone() const override;

  int getNbParameters() const override;

  /// Get the parameters: [sf, l_1, l_2, ..., l_n]
  Eigen::VectorXd getParameters() const override;

  /// Set the parameters: [sf, l_1, l_2, ..., l_n]
  void setParameters(const Eigen::VectorXd & parameters) override;

  /// Set the limits for the parameters
  virtual Eigen::MatrixXd getParametersLimits() const override;

  double compute(const Eigen::VectorXd & x1, const Eigen::VectorXd & x2) const override;

  Eigen::VectorXd computeGradient(const Eigen::VectorXd & x1,
                                  const Eigen::VectorXd & x2) const override;

  virtual Eigen::MatrixXd computeInputGradient(const Eigen::VectorXd & input,
                                               const Eigen::MatrixXd & points) const override;

private:
  Eigen::VectorXd length_scales;
  double process_noise;
};

}
