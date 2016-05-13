#include "rosban_gp/core/covariance_function.h"

namespace rosban_gp
{

double CovarianceFunction::compute(const Eigen::VectorXd & x) const
{
  return compute(x,x);
}
 
Eigen::VectorXd CovarianceFunction::getParametersGuess() const
{
  return Eigen::VectorXd::Constant(getNbParameters(), 1);
}
  
Eigen::VectorXd CovarianceFunction::getParametersStep() const
{
  return Eigen::VectorXd::Constant(getNbParameters(), 1);
}

Eigen::MatrixXd CovarianceFunction::buildMatrix(const Eigen::MatrixXd & inputs) const
{
  return buildMatrix(inputs, inputs);
}

Eigen::MatrixXd CovarianceFunction::buildMatrix(const Eigen::MatrixXd & inputs_a,
                                                const Eigen::MatrixXd & inputs_b) const
{
  Eigen::MatrixXd result(inputs_a.cols(), inputs_b.cols());
  for (int p1 = 0; p1 < inputs_a.cols(); p1++)
  {
    for (int p2 = 0; p2 < inputs_b.cols(); p2++)
    {
      result(p1,p2) = compute(inputs_a.col(p1), inputs_b.col(p2));
    }
  }
  return result;
}

}
