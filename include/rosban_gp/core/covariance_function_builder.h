#pragma once

#include "rosban_gp/core/covariance_function.h"

#include <map>

namespace rosban_gp
{

class CovarianceFunctionBuilder
{
public:
  CovarianceFunctionBuilder();

  /// Build a covariance function with the given class id and the chosen input_dim
  CovarianceFunction * build(int class_id, int input_dim);
};

}
