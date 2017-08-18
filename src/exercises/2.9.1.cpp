#include "rosban_gp/core/gaussian_process.h"
#include "rosban_gp/core/squared_exponential.h"

#include "rosban_random/multivariate_gaussian.h"
#include "rosban_random/tools.h"

#include <functional>
#include <fstream>
#include <chrono>

using namespace rosban_gp;
using rosban_random::MultivariateGaussian;

int main()
{
  // Setting problem properties
  Eigen::MatrixXd limits(1,2);
  limits(0,0) = -5;
  limits(0,1) = 5;
  // Parameters
  int nb_points = 1000;
  int nb_func = 5;

  // Generating input
  Eigen::MatrixXd input(1, nb_points);
  double x = limits(0,0);
  double x_step = (limits(0,1) - limits(0,0)) / nb_points;
  for (int i = 0; i < nb_points; i++)
  {
    input(0,i) = x;
    x += x_step;
  }

  std::unique_ptr<CovarianceFunction> covar_func(new SquaredExponential());

  Eigen::VectorXd mu = Eigen::VectorXd::Zero(nb_points);
  Eigen::MatrixXd sigma = covar_func->buildMatrix(input);

  std::default_random_engine engine = rosban_random::getRandomEngine();
  MultivariateGaussian distrib(mu, sigma);

  // Samples used
  Eigen::MatrixXd samples(1,5);
  Eigen::VectorXd observations(5);
  samples << -4, -3, 0, 2, 3;
  observations << -1.5, -1, 1, 0.5, 0;

  GaussianProcess gp(samples, observations, std::move(covar_func));

  std::ofstream prior_out, posterior_out, prediction_out;

  // PRIOR
  prior_out.open("prior.csv");
  prior_out << "func,input,output" << std::endl;
  for (int func_id = 1; func_id <= nb_func; func_id++)
  {
    Eigen::VectorXd func_values = distrib.getSample(&engine);
    for (int i = 0; i < nb_points; i++)
    {
      prior_out << "f" << func_id << "," << input(0,i) << "," << func_values(i) << std::endl;
    }
  }
  prior_out.close();

  // POSTERIOR
  posterior_out.open("posterior.csv");
  posterior_out << "func,input,output" << std::endl;
  for (int func_id = 1; func_id <= nb_func; func_id++)
  {
    // Not adding noise for estimating the posterior
    Eigen::VectorXd func_values = gp.generateValues(input, engine);
    for (int i = 0; i < nb_points; i++)
    {
      posterior_out << "f" << func_id << "," << input(0,i) << "," << func_values(i) << std::endl;
    }
  }
  posterior_out.close();

  // PREDICTION
  prediction_out.open("prediction.csv");
  prediction_out << "input,mean,min,max" << std::endl;
  for (int func_id = 1; func_id <= nb_func; func_id++)
  {
    for (int i = 0; i < nb_points; i++)
    {
      double mean, var;
      gp.getDistribParameters(input.col(i), mean, var);
      double interval = 2 * std::sqrt(var);
      double min = mean - interval;
      double max = mean + interval;
      prediction_out << input(0,i) << "," << mean << "," << min << "," << max << std::endl;
    }
  }
  prediction_out.close();

}
