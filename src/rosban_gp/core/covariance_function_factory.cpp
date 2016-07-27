#include "rosban_gp/core/covariance_function_factory.h"

#include "rosban_gp/core/neural_network.h"
#include "rosban_gp/core/neural_network2.h"
#include "rosban_gp/core/squared_exponential.h"

namespace rosban_gp
{

CovarianceFunctionFactory::CovarianceFunctionFactory()
{
  registerBuilder(CovarianceFunction::SquaredExponential, []()
                  {
                    return std::unique_ptr<CovarianceFunction>(new SquaredExponential);
                  });
  registerBuilder(CovarianceFunction::NeuralNetwork, []()
                  {
                    return std::unique_ptr<CovarianceFunction>(new NeuralNetwork);
                  });
  registerBuilder(CovarianceFunction::NeuralNetwork2, []()
                  {
                    return std::unique_ptr<CovarianceFunction>(new NeuralNetwork2);
                  });
}

}
