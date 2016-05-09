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
    dirty_cov(false),
    dirty_alpha(false)
{
}

GaussianProcess::GaussianProcess(const Eigen::MatrixXd & inputs_,
                                 const Eigen::VectorXd & observations_,
                                 CovarianceFunction covar_func_)
  : GaussianProcess()
{
  setCovarFunc(covar_func_);
  inputs = inputs_;
  observations = observations_;
  setDirty();
}

void GaussianProcess::setCovarFunc(CovarianceFunction f)
{
  covar_func = std::shared_ptr<CovarianceFunction>(new CovarianceFunction(f));
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
  Eigen::VectorXd k_star = buildCovarianceMatrix(inputs, point, *covar_func);
  Eigen::MatrixXd k_star_t = k_star.transpose();
  double point_cov = (*covar_func)(point, point);

  // Temporary matrix
  Eigen::VectorXd v;

  // Line 2
  updateCholesky();
  // Line 3
  updateAlpha();
  // Line 4
  mean = k_star.dot(alpha);
  // Line 5
  v = cholesky.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(k_star);
  // Line 6
  var = point_cov - v.dot(v);
}

// TODO: check if it is possible to use Cholesky instead
//       (problems to use the 'solve' since k_star is not a vector)
void GaussianProcess::getDistribParameters(const Eigen::MatrixXd & points,
                                           Eigen::VectorXd & mu,
                                           Eigen::MatrixXd & sigma)
{
  // Precomputations
  Eigen::MatrixXd k_star = buildCovarianceMatrix(inputs, points, *covar_func);
  Eigen::MatrixXd k_star_t = k_star.transpose();
  Eigen::MatrixXd points_cov = buildCovarianceMatrix(points, points, *covar_func);

  // Temporary matrix
  Eigen::MatrixXd tmp, alpha, v, tmp_mu, tmp_var;

  updateInverse();

  tmp_mu = k_star_t * inv_cov * observations;

  if (tmp_mu.cols() != 1)
  {
    throw std::logic_error("Unexpected dimension for mu");
  }

  mu = tmp_mu. col(0);
  sigma = points_cov - k_star_t * inv_cov * k_star;
}


Eigen::VectorXd
GaussianProcess::generateValues(const Eigen::MatrixXd & requested_inputs,
                                std::default_random_engine & engine)
{
  Eigen::VectorXd mu;
  Eigen::MatrixXd sigma;

  getDistribParameters(requested_inputs, mu, sigma);

  MultiVariateGaussian distrib(mu,sigma);

  return distrib.getSample(engine);
}

double GaussianProcess::getLogMarginalLikelihood()
{
  updateAlpha();

  //double det = 
}

void GaussianProcess::updateCov()
{
  if (!dirty_cov) return;

  if (!covar_func)
  {
    throw std::runtime_error("GaussianProcess::updateCov: covar_func is not defined yet");
  }

  cov = buildCovarianceMatrix(inputs, *covar_func);

  double epsilon = std::pow(measurement_noise,2);
  // minimal noise is required for numerical stability (cf Rasmussen2006: page 201)
  epsilon = std::max(epsilon, std::pow(10, -10));
  Eigen::MatrixXd I = Eigen::MatrixXd::Identity(cov.rows(), cov.rows());
  cov = cov + epsilon * I;
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

  cholesky = Eigen::LLT<Eigen::MatrixXd>(cov).matrixL();
  dirty_cholesky = false;
}

void GaussianProcess::updateAlpha()
{
  if (!dirty_alpha) return;

  updateCholesky();

  Eigen::MatrixXd tmp;
  // tmp = L \ y
  tmp = cholesky.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(observations);
  // alpha = L^T \ tmp = L^T \ (L \ y)
  alpha = cholesky.transpose().jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(tmp);
  dirty_alpha = false;
}

void GaussianProcess::setDirty()
{
  dirty_cov = true;
  dirty_inv = true;
  dirty_cholesky = true;
  dirty_alpha = true;
}

}
