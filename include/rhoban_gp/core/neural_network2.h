#pragma once

#include "rhoban_gp/core/covariance_function.h"

namespace rhoban_gp
{
/// Implement the following covariance function:
/// f(x,x') = 2 / pi * asin(transpose(z)*P*z' / sqrt[(1+transpose(z)*P*z)*(1+transpose(z')*P*z')])
/// with:
/// - P: diag(u_0, u_1, ..., u_D)
/// - z: the augmented vector (1 x_1 x_2 ... x_D)
/// - z': the augmented vector (1 x'_1 x'_2 ... x'_D)
/// - u_0: offset of the function from origin
/// - u_i: inverse length scale of dimension i
class NeuralNetwork2 : public CovarianceFunction
{
public:
  /// Create a NeuralNetwork2 with default values
  NeuralNetwork2();
  /// Default values
  NeuralNetwork2(int nb_dimensions);
  /// Easy binder for 1D
  NeuralNetwork2(double u0, double u1);
  /// Set hyper Parameters
  NeuralNetwork2(const Eigen::VectorXd& parameters);

  virtual ~NeuralNetwork2();

  virtual CovarianceFunction* clone() const override;

  void setDim(int dim) override;

  int getNbParameters() const override;

  /// Get the parameters: [u_0, u_1, ..., u_D]
  Eigen::VectorXd getParameters() const override;

  /// Set the parameters: [u_0, u_1, ..., u_D]
  void setParameters(const Eigen::VectorXd& parameters) override;

  /// Set the limits for the parameters
  virtual Eigen::MatrixXd getParametersLimits() const override;

  double compute(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

  // WARNING: Unverified mathematics at work here!!!
  Eigen::VectorXd computeGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

  virtual Eigen::MatrixXd computeInputGradient(const Eigen::VectorXd& input,
                                               const Eigen::MatrixXd& points) const override;

  virtual int getClassID() const override;

private:
  Eigen::VectorXd u;
};

}  // namespace rhoban_gp
