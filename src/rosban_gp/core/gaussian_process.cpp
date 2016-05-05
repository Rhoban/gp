#include "rosban_gp/core/gaussian_process.h"

#include "rosban_gp/multivariate_gaussian.h"

// Required for inverse computation
#include <Eigen/LU>

namespace rosban_gp
{

GaussianProcess::GaussianProcess()
  : dirty_inv(false)
{
}

GaussianProcess::GaussianProcess(const Eigen::MatrixXd & inputs_,
                                 const Eigen::VectorXd & observations_,
                                 CovarianceFunction covar_func_)
  : covar_func(new CovarianceFunction(covar_func_)),
    inputs(inputs_),
    observations(observations_),
    dirty_inv(true)
{
}

double GaussianProcess::getPrediction(const Eigen::VectorXd & point)
{
  double mean, var;
  getDistribParameters(point, mean, var);
  return mean;
}

double GaussianProcess::getVariance(const Eigen::VectorXd & point)
{
  double mean, var;
  getDistribParameters(point, mean, var);
  return var;
}

void GaussianProcess::getDistribParameters(const Eigen::VectorXd & point,
                                           double & mean,
                                           double & var)
{
  updateInverse();

  Eigen::MatrixXd cov_x_xstar = buildCovarianceMatrix(inputs, point, *covar_func);
  Eigen::MatrixXd cov_xstar_x = buildCovarianceMatrix(point, inputs, *covar_func);
  Eigen::MatrixXd cov_xstar_xstar = buildCovarianceMatrix(point, *covar_func);

  Eigen::MatrixXd tmp_mu = cov_xstar_x * inv_cov * observations;
  Eigen::MatrixXd tmp_var = cov_xstar_xstar - cov_xstar_x * inv_cov * cov_x_xstar;

  if (tmp_mu.cols() != 1 || tmp_mu.rows() != 1)
  {
    throw std::logic_error("Unexpected dimension for mu");
  }
  if (tmp_var.cols() != 1 || tmp_var.rows() != 1)
  {
    throw std::logic_error("Unexpected dimension for var");
  }

  mean = tmp_mu(0,0);
  var = tmp_var(0,0);
}

Eigen::VectorXd
GaussianProcess::generateValues(const Eigen::MatrixXd & requested_inputs,
                                std::default_random_engine & engine)
{
  updateInverse();

  Eigen::MatrixXd cov_x_xstar = buildCovarianceMatrix(inputs, requested_inputs, *covar_func);
  Eigen::MatrixXd cov_xstar_x = buildCovarianceMatrix(requested_inputs, inputs, *covar_func);
  Eigen::MatrixXd cov_xstar_xstar = buildCovarianceMatrix(requested_inputs, *covar_func);

  Eigen::MatrixXd tmp_mu = cov_xstar_x * inv_cov * observations;

  if (tmp_mu.cols() != 1)
  {
    throw std::runtime_error("Unexpected dimension for mu");
  }

  Eigen::VectorXd mu = tmp_mu.col(0);
  Eigen::MatrixXd sigma = cov_xstar_xstar - cov_xstar_x * inv_cov * cov_x_xstar;

  MultiVariateGaussian distrib(mu,sigma);

  return distrib.getSample(engine);
}

void GaussianProcess::updateInverse()
{
  if (!dirty_inv) return;

  if (!covar_func)
  {
    throw std::runtime_error("GaussianProcess::updateInverse: covar_func is not defined yet");
  }

  Eigen::MatrixXd cov = buildCovarianceMatrix(inputs, *covar_func);
  inv_cov = cov.inverse();
  dirty_inv = false;
}

}
