#include "rosban_gp/core/squared_exponential.h"


namespace rosban_gp
{

SquaredExponential::SquaredExponential()
  : SquaredExponential(1,1)
{
}

SquaredExponential::SquaredExponential(double l, double sf)
  : SquaredExponential(Eigen::VectorXd::Constant(1,l), sf)
{
}

SquaredExponential::SquaredExponential(const Eigen::VectorXd & l, double sf)
  : length_scales(l),
    process_noise(sf)
{
}

int SquaredExponential::getNbParameters() const
{
  return length_scales.rows() + 1;
}

Eigen::VectorXd SquaredExponential::getParameters() const
{
  Eigen::VectorXd params(getNbParameters());
  params(0) = process_noise;
  params.segment(1, length_scales.rows()) = length_scales;
  return params;
}

void SquaredExponential::setParameters(const Eigen::VectorXd & parameters)
{
  process_noise = parameters(0);
  length_scales = parameters.segment(1, parameters.rows() - 1);
}

double SquaredExponential::compute(const Eigen::VectorXd & x1,
                                   const Eigen::VectorXd & x2) const
{
  double s2 = std::pow(process_noise, 2);
  double z = (x1 - x2).cwiseQuotient(length_scales).squaredNorm();
  return s2 * std::exp(-0.5 * z);
}

Eigen::VectorXd SquaredExponential::computeGradient(const Eigen::VectorXd & x1,
                                                    const Eigen::VectorXd & x2) const
{
  int input_dim = length_scales.rows();
  Eigen::VectorXd gradient(input_dim + 1);
  // From libgp
  Eigen::VectorXd z = (x1-x2).cwiseQuotient(length_scales).array().square();  
  double s2 = std::pow(process_noise, 2);
  double k = s2*exp(-0.5*z.sum());
  gradient.head(input_dim) = z * k;
  gradient(input_dim) = 2.0 * k;
  return gradient;
}

}
