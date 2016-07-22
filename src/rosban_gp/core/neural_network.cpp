#include "rosban_gp/core/neural_network.h"


namespace rosban_gp
{

NeuralNetwork::NeuralNetwork()
  : NeuralNetwork(1,1)
{
}

NeuralNetwork::NeuralNetwork(int nb_dimensions)
  : NeuralNetwork(1,1)
{
  (void) nb_dimensions;
}

NeuralNetwork::NeuralNetwork(double sf, double l)
  : process_noise(sf), length_scale(l)
{
}

NeuralNetwork::~NeuralNetwork() {}

CovarianceFunction * NeuralNetwork::clone() const
{
  return new NeuralNetwork(*this);
}

int NeuralNetwork::getClassID() const
{
  return 2;
}

int NeuralNetwork::getNbParameters() const
{
  return 2;
}

Eigen::VectorXd NeuralNetwork::getParameters() const
{
  Eigen::VectorXd params(getNbParameters());
  params(0) = process_noise;
  params(1) = length_scale;
  return params;
}

void NeuralNetwork::setParameters(const Eigen::VectorXd & parameters)
{
  process_noise = parameters(0);
  length_scale  = parameters(1);
}

Eigen::MatrixXd NeuralNetwork::getParametersLimits() const
{
  Eigen::MatrixXd limits(getNbParameters(), 2);
  // Warning, if we let the minimum be too close to 0, the 'gradient' becomes so
  // small that it is represented by 0
  limits <<
    std::pow(10,-10), std::pow(10,10),// signal_noise
    std::pow(10, -3), std::pow(10,3);// length_scale
  return limits;
}

double NeuralNetwork::compute(const Eigen::VectorXd & x1_tmp,
                              const Eigen::VectorXd & x2_tmp) const
{
  if (x1_tmp.rows() != x2_tmp.rows()) {
    std::ostringstream oss;
    oss << "NeuralNetwork::compute: size mismatch: " << std::endl
        << "\tx1.rows() = " << x1_tmp.rows() << std::endl
        << "\tx2.rows() = " << x2_tmp.rows();
    throw std::runtime_error(oss.str());
  }
  int dim = x1_tmp.rows();
  /// first start by augmenting x1 and x2
  Eigen::VectorXd x1(dim + 1), x2(dim + 1);
  x1(0) = 1;
  x2(0) = 1;
  x1.segment(1,dim) = x1_tmp;
  x2.segment(1,dim) = x2_tmp;
  // Preload values
  double sf2  = std::pow(process_noise, 2);
  double ell2 = std::pow(length_scale, 2);
  double sx1 = 1 + x1.dot(x1);
  double sx2 = 1 + x2.dot(x2);
  // Compute 
  double numerator = 1 + x1.dot(x2);
  double denominator = std::sqrt(ell2 + sx1) * std::sqrt(ell2 + sx2);
  double K = numerator / denominator;
  // Ensure that asin will not fail
  if (K > 1 || K < -1) {
    std::ostringstream oss;
    oss << "NeuralNetwork::compute(): Invalid value for numerator/denominator: " << K
        << " expecting a value in [-1,1]";
    throw std::runtime_error(oss.str());
  }
  // Return covariance
  return sf2 * asin(K);
}

Eigen::VectorXd NeuralNetwork::computeGradient(const Eigen::VectorXd & x1_tmp,
                                               const Eigen::VectorXd & x2_tmp) const
{
  if (x1_tmp.rows() != x2_tmp.rows()) {
    std::ostringstream oss;
    oss << "NeuralNetwork::compute: size mismatch: " << std::endl
        << "\tx1.rows() = " << x1_tmp.rows() << std::endl
        << "\tx2.rows() = " << x2_tmp.rows();
    throw std::runtime_error(oss.str());
  }
  int dim = x1_tmp.rows();
  /// first start by augmenting x1 and x2
  Eigen::VectorXd x1(dim + 1), x2(dim + 1);
  x1(0) = 1;
  x2(0) = 1;
  x1.segment(1,dim) = x1_tmp;
  x2.segment(1,dim) = x2_tmp;
  // Preload values
  double sf2  = std::pow(process_noise, 2);
  double ell2 = std::pow(length_scale, 2);
  double sx1 = 1 + x1.dot(x1);
  double sx2 = 1 + x2.dot(x2);
  // Compute 
  double numerator = 1 + x1.dot(x2);
  double denominator = std::sqrt(ell2 + sx1) * std::sqrt(ell2 + sx2);
  double K = numerator / denominator;
  // Ensure that asin will not fail
  if (K >= 1 || K <= -1) {
    std::ostringstream oss;
    oss << "NeuralNetwork::computeGradient(): Invalid value for numerator/denominator: " << K
        << " expecting a value in ]-1,1[";
    throw std::runtime_error(oss.str());
  }
  // Computing gradient values
  double vx1 = sx1 / (ell2 + sx1);
  double vx2 = sx2 / (ell2 + sx2);
  double V = (vx1 + vx2) / 2;
  double length_scale_grad = - 2  * sf2 * (K - K * V) / std::sqrt(1 - K * K);
  double process_noise_grad = 2 * sf2 * asin(K);
  // Affectation and returning result
  Eigen::Vector2d grad;
  grad << process_noise_grad, length_scale_grad;
  return grad;
}

Eigen::MatrixXd NeuralNetwork::computeInputGradient(const Eigen::VectorXd & input,
                                                    const Eigen::MatrixXd & points) const
{
  (void) input; (void) points;
  throw std::logic_error("NeuralNetwork::computeInputGradient: unimplemented");
}

}
