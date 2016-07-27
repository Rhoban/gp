#include "rosban_gp/core/neural_network2.h"

#include <iostream>

namespace rosban_gp
{

NeuralNetwork2::NeuralNetwork2()
  : NeuralNetwork2(1,1)
{
}

NeuralNetwork2::NeuralNetwork2(int nb_dimensions)
{
  setDim(nb_dimensions);
}

NeuralNetwork2::NeuralNetwork2(double u0, double u1)
{
  u = Eigen::Vector2d();
  u << u0, u1;
}

NeuralNetwork2::NeuralNetwork2(const Eigen::VectorXd & parameters)
  : u(parameters)
{
}

NeuralNetwork2::~NeuralNetwork2() {}

CovarianceFunction * NeuralNetwork2::clone() const
{
  return new NeuralNetwork2(*this);
}

void NeuralNetwork2::setDim(int dim)
{
  // If dimension has changed, reset
  if ( u.rows() != dim + 1) {
    setParameters(Eigen::VectorXd::Constant(dim + 1, 1));
  }
}

int NeuralNetwork2::getNbParameters() const
{
  return u.rows();
}

Eigen::VectorXd NeuralNetwork2::getParameters() const
{
  return u;
}

void NeuralNetwork2::setParameters(const Eigen::VectorXd & parameters)
{
  u = parameters;
}

Eigen::MatrixXd NeuralNetwork2::getParametersLimits() const
{
  Eigen::MatrixXd limits(getNbParameters(), 2);
  for (int param = 0; param < getNbParameters(); param++) {
    limits(param, 0) = std::pow(10, -8);
    limits(param, 1) = std::pow(10, 8);
  }
  return limits;
}

double NeuralNetwork2::compute(const Eigen::VectorXd & x1_tmp,
                               const Eigen::VectorXd & x2_tmp) const
{
  int dim = u.rows() - 1;
  if (x1_tmp.rows() != dim ||
      x2_tmp.rows() != dim)
  {
    std::ostringstream oss;
    oss << "NeuralNetwork2::compute: size mismatch: " << std::endl
        << "\tu.rows()  = " << u.rows() << " (D + 1)" << std::endl
        << "\tx1.rows() = " << x1_tmp.rows() << std::endl
        << "\tx2.rows() = " << x2_tmp.rows();
    throw std::runtime_error(oss.str());
  }
  /// first start by augmenting x1 and x2
  Eigen::VectorXd x1(dim + 1), x2(dim + 1);
  x1(0) = 1;
  x2(0) = 1;
  x1.segment(1,dim) = x1_tmp;
  x2.segment(1,dim) = x2_tmp;
  // Compute 
  double numerator = x1.cwiseProduct(x2).dot(u);
  double denominator = std::sqrt((1 + x1.cwiseProduct(x1).dot(u)) *
                                 (1 + x2.cwiseProduct(x2).dot(u)));
  double K = numerator / denominator;
  // Ensure that asin will not fail
  if (K > 1 || K < -1) {
    std::ostringstream oss;
    oss << "NeuralNetwork2::compute(): Invalid value for numerator/denominator: " << K
        << " expecting a value in [-1,1]";
    throw std::runtime_error(oss.str());
  }
  // Return covariance
  return 2 / M_PI * asin(K);
}

Eigen::VectorXd NeuralNetwork2::computeGradient(const Eigen::VectorXd & x1_tmp,
                                                const Eigen::VectorXd & x2_tmp) const
{
  // 1. Check input
  int dim = u.rows() - 1;
  if (x1_tmp.rows() != dim ||
      x2_tmp.rows() != dim)
  {
    std::ostringstream oss;
    oss << "NeuralNetwork2::compute: size mismatch: " << std::endl
        << "\tu.rows()  = " << u.rows() << " (D + 1)" << std::endl
        << "\tx1.rows() = " << x1_tmp.rows() << std::endl
        << "\tx2.rows() = " << x2_tmp.rows();
    throw std::runtime_error(oss.str());
  }
  // 2. Augmenting x1 and x2
  Eigen::VectorXd x1(dim + 1), x2(dim + 1);
  x1(0) = 1;
  x2(0) = 1;
  x1.segment(1,dim) = x1_tmp;
  x2.segment(1,dim) = x2_tmp;
  // 3. Computing essential values (cf symbolic_computations/neural_network2_diff.mac)
  double xPx = x1.cwiseProduct(x1).dot(u);
  double xPz = x1.cwiseProduct(x2).dot(u);
  double zPz = x2.cwiseProduct(x2).dot(u);
  double sxx = xPx + 1;
  double szz = zPz + 1;
  // 4. Computing A = sqrt(sxx * szz - xPz^2);
  double A2 = sxx * szz - xPz * xPz;
  if (A2 < 0) {
    std::ostringstream oss;
    oss << "NeuralNetwork2::computeGradient: invalid value for A2: " << A2;
    throw std::runtime_error(oss.str());
  }
  double A = std::sqrt(A2);
  // 5. Compute basis for gradient
  Eigen::VectorXd x1_2 = x1.cwiseProduct(x1);
  Eigen::VectorXd x2_2 = x2.cwiseProduct(x2);
  Eigen::VectorXd x1_x2 = x1.cwiseProduct(x2);
  // 6. Compute gradient numerator
  Eigen::VectorXd numerator = 2 * x1_x2 - xPz * (x1_2 / sxx + x2_2 / szz);
  return numerator / (M_PI * A);
}

Eigen::MatrixXd NeuralNetwork2::computeInputGradient(const Eigen::VectorXd & input,
                                                     const Eigen::MatrixXd & points) const
{
  (void) input; (void) points;
  throw std::logic_error("NeuralNetwork2::computeInputGradient: unimplemented");
}

int NeuralNetwork2::getClassID() const
{
  return ID::NeuralNetwork2;
}

}
