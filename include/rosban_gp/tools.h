#pragma once

#include <Eigen/Core>

namespace rosban_gp
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
}
