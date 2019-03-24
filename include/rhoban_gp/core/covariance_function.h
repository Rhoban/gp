#pragma once

#include <Eigen/Core>

#include "rhoban_utils/serialization/stream_serializable.h"

namespace rhoban_gp
{
class CovarianceFunction : public rhoban_utils::StreamSerializable
{
public:
  virtual ~CovarianceFunction()
  {
  }

  /// Covariance functions need to be cloneable
  virtual CovarianceFunction* clone() const = 0;

  /// Inform the covariance function that the number of parameters has changed
  virtual void setDim(int dim) = 0;

  /// Return the number of parameters
  virtual int getNbParameters() const = 0;

  /// Get the function parameters (order matters)
  virtual Eigen::VectorXd getParameters() const = 0;

  /// Return default guess for parameters
  virtual Eigen::VectorXd getParametersGuess() const;

  /// Return initial step for rProp
  virtual Eigen::VectorXd getParametersStep() const;

  /// Return the limits for the parameters
  /// No default is provided in order to ensure that each covariance function
  /// override the function
  virtual Eigen::MatrixXd getParametersLimits() const = 0;

  /// Set the function parameters (order matters)
  virtual void setParameters(const Eigen::VectorXd& parameters) = 0;

  /// Compute covar(x,x)
  double compute(const Eigen::VectorXd& x) const;

  /// Compute covar(x1,x2)
  virtual double compute(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const = 0;

  /// Compute the gradient of covar(x1,x2) with respect to the function parameters
  virtual Eigen::VectorXd computeGradient(const Eigen::VectorXd& x1, const Eigen::VectorXd& x2) const = 0;

  /// Compute the gradient of the value with respect to the input at the given input
  /// i.e. derivative over input of k(x*,X), with x* = input and X = points
  /// Output matrix has:
  /// - input.rows() rows
  /// - points.cols() columns
  /// Result at row i and col j is:
  /// - The partial derivative of k(x*, X_j) regarding x*_i
  virtual Eigen::MatrixXd computeInputGradient(const Eigen::VectorXd& input, const Eigen::MatrixXd& points) const = 0;

  /// Build a covariance Matrix K such as:
  /// K_{i,j} = covar_func(inputs.col(i), inputs.col(j))
  Eigen::MatrixXd buildMatrix(const Eigen::MatrixXd& inputs) const;

  /// Build a covariance Matrix K such as:
  /// K_{i,j} = covar_func(inputs_a.col(i), inputs_b.col(j))
  Eigen::MatrixXd buildMatrix(const Eigen::MatrixXd& inputs_a, const Eigen::MatrixXd& inputs_b) const;

  // Default implementation of read and setFromStream with following format:
  // - int: nb_parameters
  // - double_array: parameters values
  virtual int writeInternal(std::ostream& out) const override;
  virtual int read(std::istream& in) override;

  /// List all the existing covariance functions
  enum ID : int
  {
    SquaredExponential = 1,
    NeuralNetwork = 2,
    NeuralNetwork2 = 3
  };
};

}  // namespace rhoban_gp
