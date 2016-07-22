#include "rosban_gp/core/covariance_function_builder.h"

#include "rosban_gp/core/neural_network.h"
#include "rosban_gp/core/neural_network2.h"
#include "rosban_gp/core/squared_exponential.h"

namespace rosban_gp
{

CovarianceFunctionBuilder::CovarianceFunctionBuilder() {}

CovarianceFunction * CovarianceFunctionBuilder::build(int class_id, int input_dim)
{
  if (class_id == 1) {
    return new SquaredExponential(input_dim);
  }
  if (class_id == 2) {
    return new NeuralNetwork(input_dim);
  }
  if (class_id == 3) {
    return new NeuralNetwork2(input_dim);
  }
  std::ostringstream oss;
  oss << "CovarianceFunctionBuilder::build: unknown class_id: " << class_id;
  throw std::runtime_error(oss.str());
}

}
