#include "rosban_gp/core/gaussian_process.h"

#include "rosban_gp/multivariate_gaussian.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>       // Required for inverse computation
#include <Eigen/SVD>      // Required for jacobiSVD

namespace rosban_gp
{

GaussianProcess::GaussianProcess()
  : dirty_inv(false),
    dirty_cholesky(false),
    dirty_cov(false)
{
}

GaussianProcess::GaussianProcess(const Eigen::MatrixXd & inputs_,
                                 const Eigen::VectorXd & observations_,
                                 CovarianceFunction covar_func_)
  : GaussianProcess()
{
  covar_func = std::shared_ptr<CovarianceFunction>(new CovarianceFunction(covar_func_));
  inputs = inputs_;
  observations = observations_;
  setDirty();
}

void GaussianProcess::setMeasurementNoise(double noise_stddev)
{
  measurement_noise = noise_stddev;
  setDirty();
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

// Uses Algorithm 2.1 from Rasmussen 2006 (page 19)
void GaussianProcess::getDistribParameters(const Eigen::VectorXd & point,
                                           double & mean,
                                           double & var)
{
  // Precomputations
  Eigen::MatrixXd k_star = buildCovarianceMatrix(inputs, point, *covar_func);
  Eigen::MatrixXd k_star_t = k_star.transpose();
  Eigen::MatrixXd point_cov = buildCovarianceMatrix(point, point, *covar_func);// 1x1 Matrix

  // Temporary matrix
  Eigen::MatrixXd tmp, alpha, v, tmp_mu, tmp_var;

  // Line 2
  updateCholesky();

  // Line 3
  tmp = cholesky.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(observations);
  alpha = cholesky.transpose().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp);
  // Line 4
  tmp_mu = k_star_t * alpha;
  // Line 5
  v = cholesky.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(k_star);
  // Line 6
  tmp_var = point_cov - (v.transpose() * v);

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

void GaussianProcess::updateCov()
{
  if (!dirty_cov) return;

  if (!covar_func)
  {
    throw std::runtime_error("GaussianProcess::updateCov: covar_func is not defined yet");
  }

  cov = buildCovarianceMatrix(inputs, *covar_func);
}

void GaussianProcess::updateInverse()
{
  if (!dirty_inv) return;

  updateCov();

  inv_cov = cov.inverse();
  dirty_inv = false;
}

void GaussianProcess::updateCholesky()
{
  if (!dirty_cholesky) return;

  updateCov();

  double epsilon = std::pow(measurement_noise,2);
  // minimal noise is required for numerical stability (cf Rasmussen2006: page 201)
  epsilon = std::max(epsilon, std::pow(10, -10));
  Eigen::MatrixXd I;
  I = Eigen::MatrixXd::Identity(cov.rows(), cov.rows());
  cholesky = Eigen::LLT<Eigen::MatrixXd>(cov + epsilon * I).matrixL();
  dirty_cholesky = false;
}

void GaussianProcess::setDirty()
{
  dirty_cov = true;
  dirty_inv = true;
  dirty_cholesky = true;
}

}
