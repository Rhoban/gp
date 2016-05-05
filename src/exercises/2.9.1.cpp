#include "rosban_gp/multivariate_gaussian.h"
#include "rosban_gp/tools.h"
#include "rosban_gp/core/gaussian_process.h"

#include <functional>
#include <fstream>
#include <chrono>

using rosban_gp::MultiVariateGaussian;

std::default_random_engine get_random_engine()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  return std::default_random_engine(seed);
}

int main(int argc, char ** argv)
{
  // Setting problem properties
  Eigen::MatrixXd limits(1,2);
  limits(0,0) = -5;
  limits(0,1) = 5;
  // Parameters
  int nb_points = 1000;
  int nb_func = 3;

  // Generating input
  Eigen::MatrixXd input(1, nb_points);
  double x = limits(0,0);
  double x_step = (limits(0,1) - limits(0,0)) / nb_points;
  for (int i = 0; i < nb_points; i++)
  {
    input(0,i) = x;
    x += x_step;
  }

  rosban_gp::CovarianceFunction covar_func = [](const Eigen::VectorXd & p1,
                                                const Eigen::VectorXd & p2)
    {
      double norm2 = (p1 - p2).squaredNorm();
      return std::exp(-0.5 * norm2);
    };

  Eigen::VectorXd mu = Eigen::VectorXd::Zero(nb_points);
  Eigen::MatrixXd sigma = rosban_gp::buildCovarianceMatrix(input, covar_func);

  std::default_random_engine engine = get_random_engine();
  MultiVariateGaussian distrib(mu, sigma);

  std::ofstream prior_out, posterior_out;

  prior_out.open("prior.csv");
  posterior_out.open("posterior.csv");

  prior_out << "func,input,output" << std::endl;

  for (int func_id = 1; func_id <= nb_func; func_id++)
  {
    Eigen::VectorXd func_values = distrib.getSample(engine);
    for (int i = 0; i < nb_points; i++)
    {
      prior_out << "f" << func_id << "," << input(0,i) << "," << func_values(i) << std::endl;
    }
  }

  prior_out.close();

  Eigen::MatrixXd samples(1,5);
  Eigen::VectorXd observations(5);

  samples << -4, -3, 0, 2, 3;
  observations << -1.5, -1, 1, 0.5, 0;

  rosban_gp::GaussianProcess gp(samples, observations, covar_func);


  posterior_out << "func,input,output" << std::endl;

  for (int func_id = 1; func_id <= nb_func; func_id++)
  {
    Eigen::VectorXd func_values = gp.generateValues(input, engine);
    for (int i = 0; i < nb_points; i++)
    {
      posterior_out << "f" << func_id << "," << input(0,i) << "," << func_values(i) << std::endl;
    }
  }

  posterior_out.close();
}
