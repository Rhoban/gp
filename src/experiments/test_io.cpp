#include "rosban_gp/core/covariance_function_factory.h"
#include "rosban_gp/core/gaussian_process.h"

#include <fstream>
#include <iostream>

using rosban_gp::CovarianceFunction;
using rosban_gp::CovarianceFunctionFactory;
using rosban_gp::GaussianProcess;

int main()
{
  int nb_cells = 5;
  int nb_cells2 = nb_cells * nb_cells;
  Eigen::MatrixXd inputs(2,nb_cells2);
  Eigen::VectorXd observations(nb_cells2);
  // Creating sample from x - 2y = z
  int sample = 0;
  for (int x = 0; x < nb_cells; x++) {
    for (int y = 0; y < nb_cells; y++) {
      inputs(0, sample) = x;
      inputs(1, sample) = y;
      observations(sample) = x - 2 * y;
      sample++;
    }
  }
  std::vector<CovarianceFunction::ID> test_types =
    {
      CovarianceFunction::SquaredExponential,
      CovarianceFunction::NeuralNetwork,
      CovarianceFunction::NeuralNetwork2
    };
  CovarianceFunctionFactory factory;
  for (CovarianceFunction::ID cov_id : test_types)
  {
    std::cout << "-------------------------------------------" << std::endl
              << "Running test for cov: " << cov_id << std::endl;

    // Creating first Gaussian Process
    std::unique_ptr<CovarianceFunction> cov_func = factory.build(cov_id);
    cov_func->setDim(2);
    GaussianProcess original_gp(inputs, observations, std::move(cov_func));
    GaussianProcess copy_gp;//Not initialized

    // Autotuning original_gp
    rosban_gp::RandomizedRProp::Config autoTuneConf;
    original_gp.autoTune(autoTuneConf);

    // writing original
    std::ofstream ofs("gp.bin", std::ios::binary);
    if (!ofs) {
      std::cerr << "Failed to open original file for writing" << std::endl;
      return -1;
    }
    int original_bytes_written = original_gp.write(ofs);
    ofs.close();

    std::cout << "Original bytes written: " << original_bytes_written << std::endl;

    // Reading from original
    std::ifstream ifs("gp.bin", std::ios::binary);
    if (!ifs) {
      std::cerr << "Failed to open original file for reading" << std::endl;
      return -1;
    }
    int original_bytes_read = copy_gp.read(ifs);
    ifs.close();

    std::cout << "Original bytes read   : " << original_bytes_read    << std::endl;

    // writing copy
    std::ofstream ofs_copy("gp_copy.bin", std::ios::binary);
    if (!ofs_copy) {
      std::cerr << "Failed to open copy file for writing" << std::endl;
      return -1;
    }
    int copy_bytes_written = copy_gp.write(ofs);
    ofs_copy.close();

    std::cout << "Copy bytes written    : " << copy_bytes_written     << std::endl;

    Eigen::VectorXd test_input(2);
    test_input << 1.5, 2.5;

    double original_prediction = original_gp.getPrediction(test_input);
    double original_variance   = original_gp.getVariance(test_input);
    double copy_prediction     = copy_gp.getPrediction(test_input);
    double copy_variance       = copy_gp.getVariance(test_input);

    // Outputting some messages:
    std::cout << "For test input: " << test_input.transpose() << std::endl
              << "\toriginal prediction : " << original_prediction << std::endl
              << "\toriginal variance   : " << original_variance   << std::endl
              << "\tcopy prediction     : " << copy_prediction     << std::endl
              << "\tcopy variance       : " << copy_variance       << std::endl;
  }
}
