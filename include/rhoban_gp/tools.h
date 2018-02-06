#pragma once

#include <Eigen/Core>

#include "rhoban_gp/core/gaussian_process.h"

#include <random>

namespace rhoban_gp
{

/// Return a vector v such as:
/// v(i) = f(inputs.col(i)) + w
/// With w a normal noise
/// If engine is NULL, then create its own engine (and free it before returning)
/// otherwise, it uses the provided engine
Eigen::VectorXd generateObservations(const Eigen::MatrixXd & inputs,
                                     std::function<double(const Eigen::VectorXd &)> f,
                                     double measurement_noise,
                                     std::default_random_engine * engine = NULL);

Eigen::VectorXd generateObservations(const Eigen::MatrixXd & inputs,
                                     std::function<double(double)> f,
                                     double measurement_noise,
                                     std::default_random_engine * engine = NULL);

/// aggregate several gaussian processes to find the parameters of the corresponding
/// distribution. This methods consider that all the gaussian process where build from
/// the same set of data, therefore the final variance does not correspond to the one
/// which should be obtained through convolution
/// If an output stream is provided, then print debug information on it
void getDistribParameters(const Eigen::VectorXd & input,
                          const std::vector<GaussianProcess> & gps,
                          double & mean,
                          double & var,
                          std::ostream * output_pointer = NULL);
}
