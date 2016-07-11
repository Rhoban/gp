#include "rosban_gp/core/gaussian_process.h"
#include "rosban_gp/core/squared_exponential.h"

#include <fstream>
#include <iostream>

using rosban_gp::CovarianceFunction;
using rosban_gp::GaussianProcess;
using rosban_gp::SquaredExponential;

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
  // Creating first Gaussian Process
  GaussianProcess original_gp(inputs, observations,
                              std::unique_ptr<CovarianceFunction>(new SquaredExponential(2)));
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

  // Reading from original
  std::ifstream ifs("gp.bin", std::ios::binary);
  if (!ifs) {
    std::cerr << "Failed to open original file for reading" << std::endl;
    return -1;
  }
  int original_bytes_read = copy_gp.read(ifs);
  ifs.close();

  // writing copy
  std::ofstream ofs_copy("gp_copy.bin", std::ios::binary);
  if (!ofs_copy) {
    std::cerr << "Failed to open copy file for writing" << std::endl;
    return -1;
  }
  int copy_bytes_written = copy_gp.write(ofs);
  ofs_copy.close();

  Eigen::VectorXd test_input(2);
  test_input << 1.5, 2.5;

  double original_prediction = original_gp.getPrediction(test_input);
  double original_variance   = original_gp.getVariance(test_input);
  double copy_prediction     = copy_gp.getPrediction(test_input);
  double copy_variance       = copy_gp.getVariance(test_input);

  // Outputting some messages:
  std::cout << "Original bytes written: " << original_bytes_written << std::endl
            << "Original bytes read   : " << original_bytes_read    << std::endl
            << "Copy bytes written    : " << copy_bytes_written     << std::endl
            << "For test input: " << test_input.transpose() << std::endl
            << "\toriginal prediction : " << original_prediction << std::endl
            << "\toriginal variance   : " << original_variance   << std::endl
            << "\tcopy prediction     : " << copy_prediction     << std::endl
            << "\tcopy variance       : " << copy_variance       << std::endl;
}
