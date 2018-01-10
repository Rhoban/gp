#pragma once

#include "rosban_gp/core/covariance_function.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rosban_gp
{

class CovarianceFunctionFactory : public rhoban_utils::Factory<CovarianceFunction>
{
public:
  CovarianceFunctionFactory();
};

}
