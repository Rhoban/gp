#include "rhoban_random/tools.h"

#include "rhoban_gp/core/gaussian_process.h"
#include "rhoban_gp/core/neural_network2.h"
#include "rhoban_gp/gradient_ascent/randomized_rprop.h"

#include <fstream>

using namespace rhoban_gp;

int main()
{
  auto engine = rhoban_random::getRandomEngine();

  // Setting problem properties
  double x_min = -5;
  double x_max = 5;
  int nb_points = 500;
  double sigma0 = 2;
  std::vector<double> sigmas = {1,3,10};

  // Computing file path
  std::ostringstream path;
  path << "functions_4_5b.csv";
  // Opening stream
  std::ofstream out;
  out.open(path.str());
  // Writing header
  out << "func,input,output" << std::endl;

  // Initializing inputs
  Eigen::MatrixXd inputs(1, nb_points);
  for (int i = 0; i < nb_points; i++)
  {
    double x = x_min + i * (x_max - x_min) / (nb_points - 1);
    inputs(0,i) = x;
  }

  for (double sigma : sigmas) {
    // Initializing GP
    std::unique_ptr<CovarianceFunction> covar_func(new NeuralNetwork2(std::pow(sigma0,2),
                                                                      std::pow(sigma ,2)));
    GaussianProcess gp(Eigen::MatrixXd(), Eigen::VectorXd(), std::move(covar_func));

    Eigen::VectorXd observations = gp.generateValues(inputs, engine, false);

    // Writing predictions
    for (int i = 0; i < nb_points; i++)
    {
      out << "sigma=" << sigma << "," << inputs(0,i) << "," << observations(i) << std::endl;
    }
  }
}
