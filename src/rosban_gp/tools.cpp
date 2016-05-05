#include "rosban_gp/tools.h"

namespace rosban_gp
{

Eigen::MatrixXd buildCovarianceMatrix(const Eigen::MatrixXd & inputs,
                                      CovarianceFunction covar_func)
{
  return buildCovarianceMatrix(inputs, inputs, covar_func);
}

Eigen::MatrixXd buildCovarianceMatrix(const Eigen::MatrixXd & inputs_a,
                                      const Eigen::MatrixXd & inputs_b,
                                      CovarianceFunction covar_func)
{
  Eigen::MatrixXd result(inputs_a.cols(), inputs_b.cols());
  for (int p1 = 0; p1 < inputs_a.cols(); p1++)
  {
    for (int p2 = 0; p2 < inputs_b.cols(); p2++)
    {
      double value = covar_func(inputs_a.col(p1), inputs_b.col(p2));
      result(p1,p2) = value;
    }
  }
  return result;
}

}
