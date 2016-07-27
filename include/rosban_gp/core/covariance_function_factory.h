#pragma once

#include "rosban_gp/core/covariance_function.h"

#include "rosban_utils/factory.h"

#include <map>

namespace rosban_gp
{

class CovarianceFunctionFactory : public rosban_utils::Factory<CovarianceFunction>
{
public:
  CovarianceFunctionFactory();
};

}
