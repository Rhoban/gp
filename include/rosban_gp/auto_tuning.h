#pragma once

#include "rosban_gp/core/gaussian_process.h"

namespace rosban_gp
{

void runSimpleGradientAscent(GaussianProcess & gp,
                             const Eigen::VectorXd & initial_guess,
                             const Eigen::VectorXd & gradient,
                             double epsilon);
                             

}
