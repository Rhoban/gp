#include "rosban_gp/scoring.h"

namespace rosban_gp
{

double computeMSE(const Eigen::VectorXd & observations,
                  const Eigen::VectorXd & predictions)
{
  return (observations - predictions).squaredNorm() / predictions.rows();
}

/// Compute the standardized mean squared error
/// SMSE = MSE / Var(observations) 
double computeSMSE(const Eigen::VectorXd & observations,
                   const Eigen::VectorXd & predictions)
{
  double mean_obs = predictions.sum() / predictions.rows();
  double var = (predictions - Eigen::VectorXd::Constant(predictions.size(), mean_obs)).squaredNorm();
  return computeMSE(observations, predictions) / var;
}


/// Compute the mean standardized log loss
/// MSLL = avg(SLL)
double computeMSLL(const Eigen::MatrixXd & inputs,
                   const Eigen::VectorXd & observations,
                   const Eigen::VectorXd & predictions)
{
  throw std::runtime_error("unimplemented function");
}

}
