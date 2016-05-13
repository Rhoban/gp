#include "rosban_gp/core/gaussian_process.h"
#include "rosban_gp/core/squared_exponential.h"

#include "rosban_gp/auto_tuning.h"
#include "rosban_gp/tools.h"

#include "rosban_random/tools.h"

#include <fstream>

using rosban_gp::CovarianceFunction;
using rosban_gp::GaussianProcess;
using rosban_gp::SquaredExponential;

// Only valid for uni-dimensional input
class MultiGP
{
public:
  MultiGP(const Eigen::MatrixXd & inputs,
          const Eigen::VectorXd & observations,
          double split_)
    : split(split_)
    {
      // Splitting indexes
      std::vector<int> idx_lower;
      std::vector<int> idx_upper;
      for (int col = 0; col < inputs.cols(); col++)
      {
        if (inputs(0, col) > split) {
          idx_upper.push_back(col);
        }
        else {
          idx_lower.push_back(col);
        }
      }
      // Building new inputs / observetions
      Eigen::MatrixXd lower_inputs(1, idx_lower.size());
      Eigen::MatrixXd upper_inputs(1, idx_upper.size());
      Eigen::VectorXd lower_obs(idx_lower.size());
      Eigen::VectorXd upper_obs(idx_upper.size());
      int lower_i = 0;
      int upper_i = 0;
      for (int col = 0; col < inputs.cols(); col++)
      {
        if (inputs(0,col) > split) {
          upper_inputs(0, upper_i) = inputs(0, col);
          upper_obs(upper_i) = observations(col);
          upper_i++;
        }
        else {
          lower_inputs(0, lower_i) = inputs(0, col);
          lower_obs(lower_i) = observations(col);
          lower_i++;
        }
      }
      // Building GP + training
      lower_gp = GaussianProcess(lower_inputs, lower_obs,
                                 std::unique_ptr<CovarianceFunction>(new SquaredExponential()));
      upper_gp = GaussianProcess(upper_inputs, upper_obs,
                                 std::unique_ptr<CovarianceFunction>(new SquaredExponential()));
      // Perform gradients
      double epsilon = std::pow(10, -6);
      rProp(lower_gp, lower_gp.getParametersGuess(), lower_gp.getParametersStep(),
            lower_gp.getParametersLimits(), epsilon);
      rProp(upper_gp, upper_gp.getParametersGuess(), upper_gp.getParametersStep(),
            upper_gp.getParametersLimits(), epsilon);
    }

  double split;
  GaussianProcess lower_gp;
  GaussianProcess upper_gp;
};

void getDistribParameters(const Eigen::VectorXd & input,
                          std::vector<MultiGP> & multi_gps,
                          double & mean,
                          double & var)
{
  int nb_predictors = multi_gps.size();
  Eigen::VectorXd means(nb_predictors);
  Eigen::VectorXd precisions(nb_predictors);
  // Compute values for each predictor
  for (int i = 0; i < multi_gps.size(); i++)
  {
    MultiGP & gps = multi_gps[i];
    // Choose appropriate gp
    GaussianProcess * gp = &gps.lower_gp;
    if (input(0) > gps.split) gp = &gps.upper_gp;
    // compute values
    double tmp_mean, tmp_var;
    gp->getDistribParameters(input, tmp_mean, tmp_var);
    // Store values
    means(i) = tmp_mean;
    precisions(i) = 1.0 / tmp_var;
  }
  // Mix predictors
  Eigen::VectorXd weights = precisions;
  double total_weight = weights.sum();
  mean = weights.dot(means) / total_weight;
  // Since we artificially create nb_predictions, we cannot simply sum the precisions
  double final_precision = total_weight / nb_predictors;
  var = 1.0 / final_precision;
}

int main(int argc, char ** argv)
{
  // getting random tool
  auto engine = rosban_random::getRandomEngine();

  // Setting problem properties
  Eigen::MatrixXd limits(1,2);
  limits(0,0) = -8;
  limits(0,1) = 8;
  int nb_samples = 50;
  int nb_prediction_points = 1000;
  int nb_predictors = 25;

//  std::function<double(const Eigen::VectorXd &)> f =
//    [](const Eigen::VectorXd & input)
//    {
//      return std::fabs(input(0));
//    };
  std::function<double(const Eigen::VectorXd &)> f =
    [](const Eigen::VectorXd & input)
    {
      if (input(0) > 0) return 1;
      return -1;
    };
//  std::function<double(const Eigen::VectorXd &)> f =
//    [](const Eigen::VectorXd & input)
//    {
//      return sin(input(0));
//    };

  // Generating random input
  Eigen::MatrixXd samples = rosban_random::getUniformSamplesMatrix(limits, nb_samples, &engine);

  // Generating output
  Eigen::VectorXd observations = rosban_gp::generateObservations(samples, f, 0.05, &engine);
  
  // Generating random splits
  double min_input = samples.minCoeff();
  double max_input = samples.maxCoeff();
  std::vector<double> splits = rosban_random::getUniformSamples(min_input,
                                                                max_input,
                                                                nb_predictors,
                                                                &engine);
  // Generate all the MultiGP
  std::vector<MultiGP> multi_gps;
  for (double split : splits)
  {
    multi_gps.push_back(MultiGP(samples, observations, split));
  }

   // Writing predictions + points
  std::ofstream out;
  out.open("randomized_gp_predictions.csv");
  out << "type,input,mean,min,max" << std::endl;

  // Writing Ref points
  for (int i = 0; i < samples.cols(); i++)
  {
    // write with the same format but min and max carry no meaning
    out << "observation," << samples(0,i) << "," << observations(i) << ",0,0" << std::endl;
  }
  
  // Writing predictions
  for (int point = 0; point < nb_prediction_points; point++)
  {
    // Computing input
    double delta = limits(0,1) - limits(0,0);
    double x = limits(0,0) + delta * point / nb_prediction_points;
    Eigen::VectorXd prediction_input(1);
    prediction_input(0) = x; 
    // Retrieving distrib parameters
    double mean, var;
    getDistribParameters(prediction_input, multi_gps, mean, var);
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
