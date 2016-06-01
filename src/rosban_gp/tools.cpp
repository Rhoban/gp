#include "rosban_gp/tools.h"

#include "rosban_random/tools.h"
#include <iostream>

namespace rosban_gp
{

Eigen::VectorXd generateObservations(const Eigen::MatrixXd & inputs,
                                     std::function<double(const Eigen::VectorXd &)> f,
                                     double measurement_noise,
                                     std::default_random_engine * engine)
{
  bool cleanup = false;
  if (engine == NULL)
  {
    engine = rosban_random::newRandomEngine();
    cleanup = true;
  }
  std::normal_distribution<double> noise_distrib(0, measurement_noise);
  Eigen::VectorXd observations(inputs.cols());
  for (int i = 0; i < observations.size(); i++)
  {
    observations(i) = f(inputs.col(i)) + noise_distrib(*engine);
  }
  // Cleaning if required
  if (cleanup) delete(engine);

  return observations;
}

Eigen::VectorXd generateObservations(const Eigen::MatrixXd & inputs,
                                     std::function<double(double)> f,
                                     double measurement_noise,
                                     std::default_random_engine * engine)
{
  return generateObservations(inputs,
                              [f](const Eigen::VectorXd & input)
                              {
                                return f(input(0));
                              },
                              measurement_noise,
                              engine);
}

void getDistribParameters(const Eigen::VectorXd & input,
                          const std::vector<GaussianProcess> & gps,
                          double & mean,
                          double & var)
{
  int nb_predictors = gps.size();
  Eigen::VectorXd means(nb_predictors);
  Eigen::VectorXd precisions(nb_predictors);
  // Compute values for each predictor
  for (size_t i = 0; i < gps.size(); i++)
  {
    const GaussianProcess & gp = gps[i];
    // compute values
    double tmp_mean, tmp_var;
    gp.getDistribParameters(input, tmp_mean, tmp_var);
    // Avoiding 0 variance cases
    if (tmp_var == 0) {
      tmp_var = std::pow(10,-20);
    }
    // Store values
    means(i) = tmp_mean;
    precisions(i) = 1.0 / tmp_var;
  }
  // Mix predictors
  Eigen::VectorXd weights = precisions;
  double total_weight = weights.sum();
  mean = weights.dot(means) / total_weight;
  // Since we artificially create nb_predictions, we cannot simply sum the precisions
  double final_precision = total_weight / nb_predictors;
  var = 1.0 / final_precision;
}

}
