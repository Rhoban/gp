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

/// This code produces figure 5.3.b at page 113 in Rasmussen-2006
int main()
{
  // Setting problem properties
  double x_min = -8;
  double x_max = 8;
  // Parameters
  std::vector<int> nb_points_vec = { 8, 21, 55 };
  double length_scale = 1;
  double sn = 0.1;
  double sf = 1;

  auto engine = getRandomEngine();

  std::ofstream out;
  out.open("fig_5_3_b.csv");

  out << "nbPoints,lengthScale,logMarginalLikelihood,derivateSN,derivateSF,derivateL" << std::endl;

  for (int nb_points : nb_points_vec)
  {
    // Generating inputs
    std::uniform_real_distribution<double> input_distrib(x_min, x_max);
    Eigen::MatrixXd inputs(1, nb_points);
    for (int p = 0; p < nb_points; p++)
    {
      inputs(0, p) = input_distrib(engine);
    }

    // Generating noisy observations
    std::unique_ptr<CovarianceFunction> generative_func(new SquaredExponential(length_scale, sf));
    GaussianProcess generative_gp;
    generative_gp.setCovarFunc(std::move(generative_func));
    generative_gp.setMeasurementNoise(sn);
    Eigen::VectorXd observations = generative_gp.generateValues(inputs, engine, true);

    // Generating data
    int nb_plot_points = 1000;
    double min_l_exp = -5;
    double max_l_exp = 1;
    for (int point = 0; point < nb_plot_points; point++)
    {
      double delta = max_l_exp - min_l_exp;
      double exp = min_l_exp + delta * point / nb_plot_points;
      double l = std::pow(10, exp);
      // Create GP with the given parameter
      std::unique_ptr<CovarianceFunction> f(new SquaredExponential(l, sf));
      GaussianProcess gp(inputs, observations, std::move(f));
      gp.setMeasurementNoise(sn);

      double log_marginal_likelihood = gp.getLogMarginalLikelihood();
      Eigen::VectorXd gradient = gp.getMarginalLikelihoodGradient();

      out << nb_points << "," << l << "," << log_marginal_likelihood << "," << gradient(0) << "," << gradient(1) << ","
          << gradient(2) << std::endl;
    }
  }

  out.close();
}
