#include "rosban_gp/multivariate_gaussian.h"

#include <functional>
#include <iostream>
#include <chrono>

using rosban_gp::MultiVariateGaussian;

typedef std::function<double(const Eigen::VectorXd &, const Eigen::VectorXd &)> CovarianceFunction;

std::default_random_engine get_random_engine()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  return std::default_random_engine(seed);
}

/// for inputs
/// - each row is a different dimension
/// - each col is a different point
Eigen::MatrixXd buildCovarianceMatrix(const Eigen::MatrixXd & inputs,
                                      CovarianceFunction covar_func)
{
  int nb_points = inputs.cols();
  Eigen::MatrixXd result(nb_points, nb_points);
  for (int p1 = 0; p1 < nb_points; p1++)
  {
    for (int p2 = p1+1; p2 < nb_points; p2++)
    {
      double value = covar_func(inputs.col(p1), inputs.col(p2));
      result(p1,p2) = value;
      result(p2,p1) = value;
    }
  }
  return result;
}

int main(int argc, char ** argv)
{
  // Setting problem properties
  Eigen::MatrixXd limits(1,2);
  limits(0,0) = -5;
  limits(0,1) = 5;
  // Parameters
  int nb_points = 5000;
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

  Eigen::VectorXd mu = Eigen::VectorXd::Zero(nb_points);
  Eigen::MatrixXd sigma = buildCovarianceMatrix(input,
                                                [](const Eigen::VectorXd & p1,
                                                   const Eigen::VectorXd & p2)
                                                {
                                                  double norm2 = (p1 - p2).squaredNorm();
                                                  return std::exp(-0.5 * norm2);
                                                }
    );

  std::default_random_engine engine = get_random_engine();
  MultiVariateGaussian distrib(mu, sigma);

  std::cout << "func,input,output" << std::endl;

  for (int func_id = 1; func_id <= nb_func; func_id++)
  {
    Eigen::VectorXd func_values = distrib.getSample(engine);
    for (int i = 0; i < nb_points; i++)
    {
      std::cout << func_id << "," << input(0,i) << "," << func_values(i) << std::endl;
    }
  }
}
