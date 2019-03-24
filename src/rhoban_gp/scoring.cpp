#include "rhoban_gp/scoring.h"

namespace rhoban_gp
{
double computeMSE(const Eigen::VectorXd& observations, const Eigen::VectorXd& predictions)
{
  return (observations - predictions).squaredNorm() / predictions.rows();
}

/// Compute the standardized mean squared error
/// SMSE = MSE / Var(observations)
double computeSMSE(const Eigen::VectorXd& observations, const Eigen::VectorXd& predictions)
{
  double mean_obs = observations.sum() / observations.rows();
  double var =
      (observations - Eigen::VectorXd::Constant(observations.size(), mean_obs)).squaredNorm() / observations.rows();
  double mse = computeMSE(observations, predictions);
  return mse / var;
}

/// Compute the mean standardized log loss
/// MSLL = avg(SLL)
double computeMSLL(const Eigen::MatrixXd& inputs, const Eigen::VectorXd& observations,
                   const Eigen::VectorXd& predictions)
{
  (void)inputs;
  (void)observations;
  (void)predictions;
  throw std::runtime_error("unimplemented function");
}

}  // namespace rhoban_gp
