#include "rosban_gp/multivariate_gaussian.h"
#include "rosban_gp/core/gaussian_process.h"
#include "rosban_gp/core/squared_exponential.h"

#include <functional>
#include <fstream>
#include <iostream>
#include <chrono>

using namespace rosban_gp;

std::default_random_engine get_random_engine()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  return std::default_random_engine(seed);
}

/// This code uses a very simple gradient ascent to find the maximum of the parameters
int main(int argc, char ** argv)
{
  // Setting problem properties
  double x_min = -8;
  double x_max = 8;
  // Parameters
  int nb_points = 50;// Points used to sample function
  int nb_prediction_points = 1000;
  double length_scale = 1;
  double sn = 0.1;
  double sf = 1;

  auto engine = get_random_engine();

  // Generating inputs
  std::uniform_real_distribution<double> input_distrib(x_min, x_max);
  Eigen::MatrixXd inputs(1,nb_points);
  for (int p = 0; p < nb_points; p++)
  {
    inputs(0,p) = input_distrib(engine);
  }

  // Generating noisy observations
  std::unique_ptr<CovarianceFunction> generative_func(new SquaredExponential(length_scale, sf));
  GaussianProcess generative_gp;
  generative_gp.setCovarFunc(std::move(generative_func));
  generative_gp.setMeasurementNoise(sn);
  Eigen::VectorXd observations = generative_gp.generateValues(inputs, engine, true);

  // Optimizing using gradient ascent
  Eigen::VectorXd guess, last_guess;
  guess      = Eigen::VectorXd::Constant(3,1);
  last_guess = Eigen::VectorXd::Constant(3,0);
  double epsilon = std::pow(10, -6);
  double gamma = 0.0001;

  GaussianProcess gp(inputs, observations,
                     std::unique_ptr<CovarianceFunction>(new SquaredExponential()));
  while ((last_guess - guess).squaredNorm() > epsilon)
  {
    last_guess = guess;
    gp.setParameters(guess);
    Eigen::VectorXd gradient = gp.getMarginalLikelihoodGradient();
    std::cout << "guess: " << guess.transpose() << std::endl
              << "\tgradient: " << gradient.transpose() << std::endl;
    // Hack: avoid to get out of limits
    double gain = gamma;
    for (int i = 0; i < gradient.rows(); i++)
    {
      if (last_guess(i) + gradient(i) * gain < 0)
      {
        gain = - last_guess(i) / gradient(i);
      }
    }
    guess = last_guess + gain * gradient;
  }

  // Use final guess
  gp.setParameters(guess);

  // Writing predictions + points
  std::ofstream out;
  out.open("gradient_predictions.csv");
  out << "type,input,mean,min,max" << std::endl;

  // Writing Ref points
  for (int i = 0; i < inputs.cols(); i++)
  {
    // write with the same format but min and max carry no meaning
    out << "observation," << inputs(0,i) << "," << observations(i) << ",0,0" << std::endl;
  }
  
  // Writing predictions
  for (int point = 0; point < nb_prediction_points; point++)
  {
    // Computing input
    double delta = x_max - x_min;
    double x = x_min + delta * point / nb_prediction_points;
    Eigen::VectorXd prediction_input(1);
    prediction_input(0) = x; 
    // Retrieving distrib parameters
    double mean, var;
    gp.getDistribParameters(prediction_input, mean, var);
    // Getting +- 2 stddev
    double interval = 2 * std::sqrt(var);
    double min = mean - interval;
    double max = mean + interval;
    // Writing line
    out << "prediction," << x << ","
        << mean << "," << min << "," << max << std::endl;
  }

  out.close();

}
