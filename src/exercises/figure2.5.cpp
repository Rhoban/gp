/// This file implement the production of the data presented at Figure 2.5 in Rasmussen 2006

#include "rosban_gp/multivariate_gaussian.h"
#include "rosban_gp/tools.h"
#include "rosban_gp/core/gaussian_process.h"

#include <functional>
#include <fstream>
#include <chrono>

using rosban_gp::GaussianProcess;
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
  limits(0,0) = -8;
  limits(0,1) = 8;
  // Parameters
  int nb_test_points = 20;

  // Generating points
  GaussianProcess gp;
  // Sampling points with noise require update of GP.cpp
  Eigen::VectorXd func_values = gp.generateValues(input, engine);

  auto engine = get_random_engine();
}
