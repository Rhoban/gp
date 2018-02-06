/// This file implement the production of the data presented at Figure 2.5 in Rasmussen 2006

#include "rhoban_random/multivariate_gaussian.h"
#include "rhoban_gp/core/gaussian_process.h"
#include "rhoban_gp/core/squared_exponential.h"

#include <functional>
#include <fstream>
#include <chrono>

using namespace rhoban_gp;

std::default_random_engine getRandomEngine()
{
  unsigned long seed = std::chrono::system_clock::now().time_since_epoch().count();
  return std::default_random_engine(seed);
}

int main()
{
  // Setting problem properties
  double x_min = -8;
  double x_max = 8;
  // Parameters
  int nb_test_points = 30;
  int nb_prediction_points = 1000;

  auto engine = getRandomEngine();

  // Generating inputs
  std::uniform_real_distribution<double> input_distrib(x_min, x_max);
  Eigen::MatrixXd inputs(1,nb_test_points);
  for (int p = 0; p < nb_test_points; p++)
  {
    inputs(0,p) = input_distrib(engine);
  }

  // Generating noisy observations
  std::unique_ptr<CovarianceFunction> generative_func(new SquaredExponential());
  GaussianProcess generative_gp;
  generative_gp.setCovarFunc(std::move(generative_func));
  Eigen::VectorXd observations = generative_gp.generateValues(inputs, engine, true);

  // Evaluation parameters
  std::vector<double> l_values  = {1  , 0.3    , 3   }; // Length-scale
  std::vector<double> sf_values = {1  , 1.08   , 1.16}; // Signal stddev
  std::vector<double> sn_values = {0.1, 0.00005, 0.89}; // Noise stddev

  // Generating prediction inputs
  Eigen::MatrixXd prediction_inputs(1, nb_prediction_points);
  double x = x_min;
  double x_step = (x_max - x_min) / nb_prediction_points;
  for (int i = 0; i < nb_prediction_points; i++)
  {
    prediction_inputs(0,i) = x;
    x += x_step;
  }
  for (size_t i = 0; i < l_values.size(); i++)
  {
    // Reading hyperparameters
    double l  = l_values[i];
    double sf = sf_values[i];
    double sn = sn_values[i];
    // Creating GP with appropriate parameters
    std::unique_ptr<CovarianceFunction> covar_func(new SquaredExponential(l,sf));
    GaussianProcess gp(inputs, observations, std::move(covar_func));
    gp.setMeasurementNoise(sn);
    // Computing file path
    std::ostringstream path;
    path << "prediction" << i << ".csv";
    // Opening stream
    std::ofstream out;
    out.open(path.str());
    // Writing header
    out << "type,input,mean,min,max" << std::endl;
    // Writing Ref points
    for (int i = 0; i < inputs.cols(); i++)
    {
      // write with the same format but min and max carry no meaning
      out << "observation," << inputs(0,i) << "," << observations(i) << ",0,0" << std::endl;
    }
    // Writing predictions
    for (int i = 0; i < nb_prediction_points; i++)
    {
      double mean, var;
      gp.getDistribParameters(prediction_inputs.col(i), mean, var);
      double interval = 2 * std::sqrt(var);
      double min = mean - interval;
      double max = mean + interval;
      out << "prediction," << prediction_inputs(0,i) << ","
          << mean << "," << min << "," << max << std::endl;
    }
    // Close output file
    out.close();
  }
}
