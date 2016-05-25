#pragma once

#include "rosban_gp/core/covariance_function.h"

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
                  std::unique_ptr<CovarianceFunction> covar_func);
  // Enabling copy of GaussianProcess
  GaussianProcess(const GaussianProcess & other);
  // Allowing left affectation
  GaussianProcess & operator=(const GaussianProcess & other);

  /// Set the parameters for measurement noise and covariance function
  /// Order is as follows: [measurement_noise, covar_parameters]
  void setParameters(const Eigen::VectorXd & parameters);

  /// Return current values for the parameters
  Eigen::VectorXd getParameters() const;

  /// Return default guess for parameters
  Eigen::VectorXd getParametersGuess() const;
  /// Return initial step for rProp
  Eigen::VectorXd getParametersStep() const;
  /// Return the limits for the parameters
  Eigen::MatrixXd getParametersLimits() const;

  /// Since modifying the CovarianceFunction requires to set the flags to dirty,
  /// access is const only.
  const CovarianceFunction & getCovarFunc() const;

  /// Update the covariance function
  void setCovarFunc(std::unique_ptr<CovarianceFunction> f);

  /// Update the measurement noise
  void setMeasurementNoise(double noise_stddev);

  /// Return a prediction of the value at the given point
  double getPrediction(const Eigen::VectorXd & point);
  /// Throw an error if some parameters are dirty 
  double getPrediction(const Eigen::VectorXd & point) const;
  /// Return an estimation of the variance at the given point
  double getVariance(const Eigen::VectorXd & point);
  /// Throw an error if some parameters are dirty 
  double getVariance(const Eigen::VectorXd & point) const;

  /// Return the predicted gradient at the given point
  Eigen::VectorXd getGradient(const Eigen::VectorXd & point);
  /// Throw an error if some parameters are dirty 
  Eigen::VectorXd getGradient(const Eigen::VectorXd & point) const;

  /// Compute the parameters of the distribution at the given point and
  /// update the 'mean' and 'var' values accordingly
  void getDistribParameters(const Eigen::VectorXd & point,
                            double & mean,
                            double & var);
  /// Throw an error if some parameters are dirty 
  void getDistribParameters(const Eigen::VectorXd & point,
                            double & mean,
                            double & var) const;

  /// Compute the parameters of the multivariate distribution for the given points
  void getDistribParameters(const Eigen::MatrixXd & points,
                            Eigen::VectorXd & mu,
                            Eigen::MatrixXd & sigma);

  /// Generate the outputs of a random function using the requested inputs
  /// While in the requested Inputs, each column is a different input,
  /// In the result, each row is a different output
  /// If add measurement noise is chosen, then an independent measurement noise
  /// with the provided standard deviation is applied on each measurement
  Eigen::VectorXd generateValues(const Eigen::MatrixXd & requested_inputs,
                                 std::default_random_engine & engine,
                                 bool add_measurement_noise = false);

  /// Compute the log likelihood of the current distribution
  /// i.e. p(observations|inputs,theta) with theta the parameters of the covariance function
  double getLogMarginalLikelihood();

  /// Return the partial derivatives of the marginal likelihood with respect to the
  /// parameters of the covariance function
  Eigen::VectorXd getMarginalLikelihoodGradient();

  /// Solve all internal computations
  void updateInternal();

private:
  /// Update the covariance matrix if required
  void updateCov();

  /// Update the inverse covariance matrix if required
  void updateInverse();

  /// Update the cholesky matrix if required
  void updateCholesky();

  /// Update the alpha matrix if required
  void updateAlpha();

  /// Signal that internal data have changed and that it is required to update internal data
  void setDirty();

  /// The covariance function used. By using unique_ptr, we make 'sure' that the function
  /// cannot be modified from outside
  std::unique_ptr<CovarianceFunction> covar_func;
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
  /// The alpha matrix: alpha = L^T \ (L \ y)
  Eigen::VectorXd alpha;
  /// Is it necessary to update inverse of the covariance matrix
  bool dirty_cov;
  /// Is it necessary to update inverse of the covariance matrix
  bool dirty_inv;
  /// Is it necessary to update the cholesky matrix
  bool dirty_cholesky;
  /// Is it necessary to update the alpha matrix
  bool dirty_alpha;
};

}
