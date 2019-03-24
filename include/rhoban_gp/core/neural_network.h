#pragma once

#include "rhoban_gp/core/covariance_function.h"

namespace rhoban_gp
{
/// Implement the following covariance function:
/// f(x,x') = sf2 * asin(transpose(x)*P*x' / sqrt[(1+transpose(x)*P*x)*(1+transpose(x')*P*x')])
/// with:
/// - sf2: control the signal variance
/// - P: Identity matrix times l^-2
/// - l: the length scale
class NeuralNetwork : public CovarianceFunction
{
public:
  /// Create a NeuralNetwork with default values
  NeuralNetwork();
  /// Default values
  NeuralNetwork(int nb_dimensions);
  /// Set hyper Parameters
  NeuralNetwork(double process_noise, double length_scale);

  virtual ~NeuralNetwork();

  virtual CovarianceFunction* clone() const override;

  void setDim(int dim) override;

  int getNbParameters() const override;

  /// Get the parameters: [sf, l]
  Eigen::VectorXd getParameters() const override;

  /// Set the parameters: [sf, l]
  void setParameters(const Eigen::VectorXd& parameters) override;

  /// Set the limits for the parameters
  virtual Eigen::MatrixXd getParametersLimits() const override;

  double compute(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

  Eigen::VectorXd computeGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const override;

  virtual Eigen::MatrixXd computeInputGradient(const Eigen::VectorXd& input,
                                               const Eigen::MatrixXd& points) const override;

  virtual int getClassID() const override;

private:
  double process_noise;
  double length_scale;
};

}  // namespace rhoban_gp
