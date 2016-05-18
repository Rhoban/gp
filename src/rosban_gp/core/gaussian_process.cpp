#include "rosban_gp/core/gaussian_process.h"

#include "rosban_random/multivariate_gaussian.h"

#include <Eigen/Cholesky>
#include <Eigen/LU>       // Required for inverse computation
#include <Eigen/SVD>      // Required for jacobiSVD

#include <sstream>
#include <stdexcept>

using rosban_random::MultiVariateGaussian;

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
                                 std::unique_ptr<CovarianceFunction> covar_func_)
  : GaussianProcess()
{
  inputs = inputs_;
  observations = observations_;
  setCovarFunc(std::move(covar_func_));
  setDirty();
}

GaussianProcess::GaussianProcess(const GaussianProcess & other)
  : covar_func(std::unique_ptr<CovarianceFunction>(other.covar_func->clone())),
    measurement_noise(other.measurement_noise),
    inputs(other.inputs),
    observations(other.observations),
    cov(other.cov),
    inv_cov(other.inv_cov),
    cholesky(other.cholesky),
    alpha(other.alpha),
    dirty_cov(other.dirty_cov),
    dirty_inv(other.dirty_inv),
    dirty_cholesky(other.dirty_cholesky),
    dirty_alpha(other.dirty_alpha)
{
}

GaussianProcess& GaussianProcess::operator=(const GaussianProcess& other)
{
  this->covar_func = std::unique_ptr<CovarianceFunction>(other.covar_func->clone());
  this->measurement_noise = other.measurement_noise;
  this->inputs = other.inputs;
  this->observations = other.observations;
  this->cov = other.cov;
  this->inv_cov = other.inv_cov;
  this->cholesky = other.cholesky;
  this->alpha = other.alpha;
  this->dirty_cov = other.dirty_cov;
  this->dirty_inv = other.dirty_inv;
  this->dirty_cholesky = other.dirty_cholesky;
  this->dirty_alpha = other.dirty_alpha;
  return *this;
}

void GaussianProcess::setParameters(const Eigen::VectorXd & parameters)
{
  int nb_parameters = 1 + getCovarFunc().getNbParameters();
  if (parameters.rows() != nb_parameters)
  {
    std::ostringstream oss;
    oss << "GaussianProcess::setParameters: " << parameters.rows()
        << " parameters received, while expecting " << nb_parameters << " parameters";
    throw std::runtime_error(oss.str());
  }
  measurement_noise = parameters(0);
  covar_func->setParameters(parameters.segment(1, nb_parameters -1));
  setDirty();
}

Eigen::VectorXd GaussianProcess::getParameters() const
{
  const CovarianceFunction & f = getCovarFunc();
  Eigen::VectorXd parameters(1 + f.getNbParameters());
  parameters(0) = measurement_noise;
  parameters.segment(1, f.getNbParameters()) = f.getParameters();
  return parameters;
}

Eigen::VectorXd GaussianProcess::getParametersGuess() const
{
  const CovarianceFunction & f = getCovarFunc();
  Eigen::VectorXd guess(1 + f.getNbParameters());
  guess(0) = 1;
  guess.segment(1, f.getNbParameters()) = f.getParametersGuess();
  return guess;
}

Eigen::VectorXd GaussianProcess::getParametersStep() const
{
  const CovarianceFunction & f = getCovarFunc();
  Eigen::VectorXd step(1 + f.getNbParameters());
  // Since this value has an impact, its default value should be chosen automatically
  step(0) = std::pow(10,-5);
  step.segment(1, f.getNbParameters()) = f.getParametersStep();
  return step;
}

Eigen::MatrixXd GaussianProcess::getParametersLimits() const
{
  const CovarianceFunction & f = getCovarFunc();
  Eigen::MatrixXd limits(1 + f.getNbParameters(), 2);
  limits(0,0) = std::pow(10,-10);
  limits(0,1) = std::numeric_limits<double>::max();
  limits.block(1,0, f.getNbParameters(), 2) = f.getParametersLimits();
  return limits;
}

const CovarianceFunction & GaussianProcess::getCovarFunc() const
{
  if (!covar_func)
    throw std::runtime_error("GaussianProcess::getCovarFunc(): covar_func is not defined");
  return *covar_func;
}

void GaussianProcess::setCovarFunc(std::unique_ptr<CovarianceFunction> covar_func_)
{
  covar_func = std::move(covar_func_);
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

double GaussianProcess::getPrediction(const Eigen::VectorXd & point) const
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

double GaussianProcess::getVariance(const Eigen::VectorXd & point) const
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
  // Line 2
  updateCholesky();
  // Line 3
  updateAlpha();
  // Use the const version since necessary content has been updated
  ((const GaussianProcess *)this)->getDistribParameters(point, mean, var);
}

void GaussianProcess::getDistribParameters(const Eigen::VectorXd & point,
                                           double & mean,
                                           double & var) const
{
  // Precomputations
  Eigen::VectorXd k_star = getCovarFunc().buildMatrix(inputs, point);
  double point_cov = getCovarFunc().compute(point);

  // Check that line 2 and 3 have been computed or throw error
  if (dirty_cholesky || dirty_alpha) {
    throw std::runtime_error("GaussianProcess::getDistribParameters: precomputations missing");
  }

  // Line 4
  mean = k_star.dot(alpha);
  // Line 5
  Eigen::VectorXd v = cholesky.jacobiSvd(Eigen::ComputeThinU | Eigen::ComputeThinV).solve(k_star);
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
  Eigen::MatrixXd k_star = getCovarFunc().buildMatrix(inputs, points);
  Eigen::MatrixXd k_star_t = k_star.transpose();
  Eigen::MatrixXd points_cov = getCovarFunc().buildMatrix(points);

  // Temporary matrix
  Eigen::MatrixXd tmp_mu;

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
                                std::default_random_engine & engine,
                                bool add_measurement_noise)
{
  Eigen::VectorXd mu;
  Eigen::MatrixXd sigma;

  getDistribParameters(requested_inputs, mu, sigma);

  MultiVariateGaussian distrib(mu,sigma);

  Eigen::VectorXd values = distrib.getSample(engine);

  // Adding measurement noise
  if (add_measurement_noise)
  {
    std::normal_distribution<double> measurement_distrib(0, measurement_noise);
    for (int i = 0; i < values.rows(); i++)
    {
      values(i) += measurement_distrib(engine);
    }
  }

  return values;
}

double GaussianProcess::getLogMarginalLikelihood()
{
  updateAlpha();
  
  // compute second term
  double det = cholesky.diagonal().array().log().sum();

  return -0.5 * observations.dot(alpha) - 0.5 * det - 0.5 * log(2 * M_PI);
}

Eigen::VectorXd GaussianProcess::getMarginalLikelihoodGradient()
{
  updateInverse();
  updateAlpha();

  // Dims: measurement_noise, covar_function_parameters
  int gradient_dim = getCovarFunc().getNbParameters() + 1;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(gradient_dim);

  Eigen::MatrixXd weights = alpha * alpha.transpose() - inv_cov;

  // Since we compute the trace: Each couple i, j of the matrix with i < j  is counted twice
  int size = weights.cols();
  for(int row = 0; row < size; row++) {
    for (int col = 0; col < row; col++) {
      Eigen::VectorXd tmp_grad = getCovarFunc().computeGradient(inputs.col(row), inputs.col(col));
      // measurement noise has no effect if row != col
      gradient.segment(1, gradient_dim - 1) += tmp_grad * weights(row, col);
    }
  }
  // Since we compute the trace: Each couple i, j of the matrix with i == j  is counted once
  for(int i = 0; i < size; i++) {
    Eigen::VectorXd tmp_grad = getCovarFunc().computeGradient(inputs.col(i), inputs.col(i));
    // measurement_noise is squared, therefore, dev is 2x
    gradient(0) += measurement_noise * weights(i, i);
    gradient.segment(1, gradient_dim - 1) += 0.5 * tmp_grad * weights(i, i);
  }
  return gradient;
}

void GaussianProcess::updateInternal()
{
  updateCov();
  updateInverse();
  updateAlpha();
  updateCholesky();
}

void GaussianProcess::updateCov()
{
  if (!dirty_cov) return;

  cov = getCovarFunc().buildMatrix(inputs);

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
