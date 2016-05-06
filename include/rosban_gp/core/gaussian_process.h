#pragma once

#include "rosban_gp/tools.h"

#include <Eigen/Core>

#include <functional>
#include <memory>

namespace rosban_gp
{

class GaussianProcess
{
public:
  GaussianProcess();

  GaussianProcess(const Eigen::MatrixXd & inputs,
                  const Eigen::VectorXd & observations,
                  CovarianceFunction covar_func);

  /// Update the covariance function
  void setCovarFunc(CovarianceFunction f);

  /// Update the measurement noise
  void setMeasurementNoise(double noise_stddev);

  /// Return a prediction of the value at the given point
  double getPrediction(const Eigen::VectorXd & point);
  /// Return an estimation of the variance at the given point
  double getVariance(const Eigen::VectorXd & point);

  /// Compute the parameters of the distribution at the given point and
  /// update the 'mean' and 'var' values accordingly
  void getDistribParameters(const Eigen::VectorXd & point,
                            double & mean,
                            double & var);

  /// Compute the parameters of the multivariate distribution for the given points
  void getDistribParameters(const Eigen::MatrixXd & points,
                            Eigen::VectorXd & mu,
                            Eigen::MatrixXd & sigma);

  /// Generate the outputs of a random function using the requested inputs
  /// While in the requested Inputs, each column is a different input,
  /// In the result, each row is a different output
  Eigen::VectorXd generateValues(const Eigen::MatrixXd & requested_inputs,
                                 std::default_random_engine & engine);

private:
  /// Update the covariance matrix if required
  void updateCov();

  /// Update the inverse covariance matrix if required
  void updateInverse();

  /// Update the cholesky matrix if required
  void updateCholesky();

  /// Signal that internal data have changed and that it is required to update internal data
  void setDirty();

  /// The covariance function used
  std::shared_ptr<CovarianceFunction> covar_func;
  /// The standard deviation of the measurements
  double measurement_noise;

  /// Known inputs of the gaussian process (row is dimension, col is sample no)
  Eigen::MatrixXd inputs;
  /// Measured outputs (row is sample no)
  Eigen::VectorXd observations;
  /// Covariance matrix of the inputs
  Eigen::MatrixXd cov;
  /// Inverse covariance matrix of the inputs
  Eigen::MatrixXd inv_cov;
  /// The L matrix of the cholesky decomposition of the covariance Matrix (LLT)
  Eigen::MatrixXd cholesky;
  /// Is it necessary to update inverse of the covariance matrix
  bool dirty_cov;
  /// Is it necessary to update inverse of the covariance matrix
  bool dirty_inv;
  /// Is it necessary to update the cholesky matrix
  bool dirty_cholesky;
};

}
