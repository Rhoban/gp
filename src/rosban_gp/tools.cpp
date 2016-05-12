#include "rosban_gp/tools.h"

#include "rosban_random/tools.h"

namespace rosban_gp
{

Eigen::VectorXd generateObservations(const Eigen::MatrixXd & inputs,
                                     std::function<double(const Eigen::VectorXd &)> f,
                                     double measurement_noise,
                                     std::default_random_engine * engine)
{
  bool cleanup = false;
  if (engine == NULL)
  {
    engine = rosban_random::newRandomEngine();
    cleanup = true;
  }
  std::normal_distribution<double> noise_distrib(0, measurement_noise);
  Eigen::VectorXd observations(inputs.cols());
  for (int i = 0; i < observations.size(); i++)
  {
    observations(i) = f(inputs.col(i)) + noise_distrib(*engine);
  }
  // Cleaning if required
  if (cleanup) delete(engine);

  return observations;
}

Eigen::VectorXd generateObservations(const Eigen::MatrixXd & inputs,
                                     std::function<double(double)> f,
                                     double measurement_noise,
                                     std::default_random_engine * engine)
{
  return generateObservations(inputs,
                              [f](const Eigen::VectorXd & input)
                              {
                                return f(input(0));
                              },
                              measurement_noise,
                              engine);
}
                                     

}
