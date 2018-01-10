#include "rosban_random/tools.h"

#include "rosban_gp/core/gaussian_process.h"
#include "rosban_gp/core/squared_exponential.h"
#include "rosban_gp/core/neural_network.h"
#include "rosban_gp/core/neural_network2.h"
#include "rosban_gp/gradient_ascent/randomized_rprop.h"

#include <fstream>
#include <iostream>

using namespace rosban_gp;

int main()
{
  // Setting problem properties
  double x_min = -1;
  double x_max = 1;
  int nb_points = 64;
  int nb_prediction_points = 1000;
  double noise_stddev = 0.1;

  auto engine = rosban_random::getRandomEngine();
  std::uniform_real_distribution<double> x_distrib(x_min, x_max);
  std::normal_distribution<double> noise_distrib(0, noise_stddev);

  // Computing file path
  std::ostringstream path;
  path << "prediction_5_10.csv";
  // Opening stream
  std::ofstream out;
  out.open(path.str());
  // Writing header
  out << "type,input,mean,min,max" << std::endl;

  // Generating input
  Eigen::MatrixXd inputs(1, nb_points);
  Eigen::VectorXd observations(nb_points);
  for (int p = 0; p < nb_points; p++) {
    double x = x_distrib(engine);
    double y = 1;
    if (x < 0) { y = -1; }
    y += noise_distrib(engine);
    out << "observation," << x << "," << y << ",0,0" << std::endl;
    inputs(0,p) = x;
    observations(p) = y;
  }

  // Training GP
  //std::unique_ptr<CovarianceFunction> covar_func(new SquaredExponential());
  //std::unique_ptr<CovarianceFunction> covar_func(new NeuralNetwork());
  std::unique_ptr<CovarianceFunction> covar_func(new NeuralNetwork2());
  GaussianProcess gp(inputs, observations, std::move(covar_func));
  RandomizedRProp::Config ga_conf;
  ga_conf.nb_trials = 5;
  ga_conf.rprop_conf->max_iterations = 100;
  ga_conf.rprop_conf->tuning_space = RProp::TuningSpace::Log;
  gp.autoTune(ga_conf);
  //Eigen::VectorXd manual_parameters(3);
  //manual_parameters << 0.1, 100, 100;
  //gp.setParameters(manual_parameters);

  // Writing predictions
  for (int i = 0; i < nb_prediction_points; i++)
  {
    double x = x_min + i * (x_max - x_min) / (nb_prediction_points - 1);
    Eigen::VectorXd prediction_input(1);
    prediction_input(0) = x;
    double mean, var;
    gp.getDistribParameters(prediction_input, mean, var);
    double interval = 2 * std::sqrt(var);
    double min = mean - interval;
    double max = mean + interval;
    out << "prediction," << x << ","
        << mean << "," << min << "," << max << std::endl;
  }

  std::cout << "Log marginal likelihood: " << gp.getLogMarginalLikelihood() << std::endl;
}
