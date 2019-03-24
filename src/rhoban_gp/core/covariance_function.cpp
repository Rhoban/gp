#include "rhoban_gp/core/covariance_function.h"

#include "rhoban_utils/io_tools.h"

namespace rhoban_gp
{
double CovarianceFunction::compute(const Eigen::VectorXd& x) const
{
  return compute(x, x);
}

Eigen::VectorXd CovarianceFunction::getParametersGuess() const
{
  return Eigen::VectorXd::Constant(getNbParameters(), 1);
}

Eigen::VectorXd CovarianceFunction::getParametersStep() const
{
  // Since this value has an impact, its default value should be chosen automatically
  double step_default_size = std::pow(10, -2);
  return Eigen::VectorXd::Constant(getNbParameters(), 1) * step_default_size;
}

Eigen::MatrixXd CovarianceFunction::buildMatrix(const Eigen::MatrixXd& inputs) const
{
  return buildMatrix(inputs, inputs);
}

Eigen::MatrixXd CovarianceFunction::buildMatrix(const Eigen::MatrixXd& inputs_a, const Eigen::MatrixXd& inputs_b) const
{
  Eigen::MatrixXd result(inputs_a.cols(), inputs_b.cols());
  for (int p1 = 0; p1 < inputs_a.cols(); p1++)
  {
    for (int p2 = 0; p2 < inputs_b.cols(); p2++)
    {
      result(p1, p2) = compute(inputs_a.col(p1), inputs_b.col(p2));
    }
  }
  return result;
}

int CovarianceFunction::writeInternal(std::ostream& out) const
{
  int bytes_written = 0;
  Eigen::VectorXd params = getParameters();
  int nb_params = params.rows();
  bytes_written += rhoban_utils::write<int>(out, nb_params);
  bytes_written += rhoban_utils::writeArray<double>(out, nb_params, params.data());
  return bytes_written;
}

int CovarianceFunction::read(std::istream& in)
{
  int bytes_read = 0;
  int nb_params;
  bytes_read += rhoban_utils::read<int>(in, &nb_params);
  Eigen::VectorXd params(nb_params);
  bytes_read += rhoban_utils::readArray<double>(in, nb_params, params.data());
  setParameters(params);
  return bytes_read;
}

}  // namespace rhoban_gp
