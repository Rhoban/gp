#include "rhoban_gp/core/gaussian_process.h"
#include "rhoban_gp/core/squared_exponential.h"
#include "rhoban_gp/auto_tuning.h"
#include "rhoban_gp/tools.h"

#include "rhoban_random/multivariate_gaussian.h"
#include "rhoban_random/tools.h"

#include <functional>
#include <fstream>
#include <iostream>
#include <chrono>

using namespace rhoban_gp;

/// This code uses a very simple gradient ascent to find the maximum of the parameters
int main()
{
  // Setting problem properties
  Eigen::MatrixXd limits(1, 2);
  limits(0, 0) = -8;
  limits(0, 1) = 8;
  // Parameters
  int nb_points = 50;  // Points used to sample function
  int nb_prediction_points = 1000;
  double length_scale = 1;
  double sn = 0.1;
  double sf = 1;

  auto engine = rhoban_random::getRandomEngine();

  // Generating inputs
  Eigen::MatrixXd inputs = rhoban_random::getUniformSamplesMatrix(limits, nb_points, &engine);

  // Generating noisy observations
  std::unique_ptr<CovarianceFunction> generative_func(new SquaredExponential(length_scale, sf));
  GaussianProcess generative_gp;
  generative_gp.setCovarFunc(std::move(generative_func));
  generative_gp.setMeasurementNoise(sn);
  Eigen::VectorXd observations = generative_gp.generateValues(inputs, engine, true);
  // tmp override
  // observations = generateObservations(inputs, [](double x)
  //                                    {
  //                                      return
  //                                      0.01 * x * x * x * x
  //                                      - 0.2 * x * x * x
  //                                      + 2 * x * x
  //                                      +  3 * x
  //                                      - 2;
  //                                    }, 0.05);
  observations = generateObservations(inputs,
                                      [](double x) {
                                        if (x > 2)
                                          return 1;
                                        return -1;
                                      },
                                      0.05);

  GaussianProcess gp(inputs, observations, std::unique_ptr<CovarianceFunction>(new SquaredExponential()));
  // Gradient ascent
  double epsilon = std::pow(10, -6);
  rProp(gp, gp.getParametersGuess(), gp.getParametersStep(), gp.getParametersLimits(), epsilon);

  // Writing predictions + points
  std::ofstream out;
  out.open("gradient_predictions.csv");
  out << "type,input,mean,min,max" << std::endl;

  // Writing Ref points
  for (int i = 0; i < inputs.cols(); i++)
  {
    // write with the same format but min and max carry no meaning
    out << "observation," << inputs(0, i) << "," << observations(i) << ",0,0" << std::endl;
  }

  // Writing predictions
  for (int point = 0; point < nb_prediction_points; point++)
  {
    // Computing input
    double delta = limits(0, 1) - limits(0, 0);
    double x = limits(0, 0) + delta * point / nb_prediction_points;
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
    out << "prediction," << x << "," << mean << "," << min << "," << max << std::endl;
  }

  out.close();
}
