#include "rhoban_gp/core/covariance_function_factory.h"

#include "rhoban_gp/core/neural_network.h"
#include "rhoban_gp/core/neural_network2.h"
#include "rhoban_gp/core/squared_exponential.h"

namespace rhoban_gp
{
CovarianceFunctionFactory::CovarianceFunctionFactory()
{
  registerBuilder(CovarianceFunction::SquaredExponential,
                  []() { return std::unique_ptr<CovarianceFunction>(new SquaredExponential); });
  registerBuilder(CovarianceFunction::NeuralNetwork,
                  []() { return std::unique_ptr<CovarianceFunction>(new NeuralNetwork); });
  registerBuilder(CovarianceFunction::NeuralNetwork2,
                  []() { return std::unique_ptr<CovarianceFunction>(new NeuralNetwork2); });
}

}  // namespace rhoban_gp
