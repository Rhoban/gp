#pragma once

#include "rhoban_gp/core/covariance_function.h"

#include "rhoban_utils/serialization/factory.h"

#include <map>

namespace rhoban_gp
{
class CovarianceFunctionFactory : public rhoban_utils::Factory<CovarianceFunction>
{
public:
  CovarianceFunctionFactory();
};

}  // namespace rhoban_gp
