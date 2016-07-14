#pragma once

#include "rosban_utils/serializable.h"

#include <Eigen/Core>

#include <functional>
#include <memory>

namespace rosban_gp
{

/// Resilient backpropagation algorithm. It uses momentum and steps are based
/// only on the sign of the gradient
class RProp
{
public:

  /// For some problems, it is more convenient to tune the parameters in a modified space
  /// Normal: the user space is conserved
  /// Log: from user space 'x' to tuning space 'y': y = log(x)
  enum class TuningSpace
  { Normal, Log };

  /// A gradient func maps an input to the gradient at this input
  typedef std::function<Eigen::VectorXd(const Eigen::VectorXd)> GradientFunc;

  class Config : public rosban_utils::Serializable {
  public:
    /// eta_pos and eta_neg are set accrording to the default value suggested in
    /// the wikipedia page
    Config();

    virtual std::string class_name() const override;
    virtual void to_xml(std::ostream &out) const override;
    virtual void from_xml(TiXmlNode *node) override;

    /// Minimal difference with previous guess to continue exploration
    /// - Warning: this value is provided in tuning_space
    double epsilon;
    /// Maximal number of iterations of the algorithm
    int max_iterations;
    /// Gain when sign is similar to previous sign
    double eta_pos;
    /// Gain when sign is different from previous sign
    double eta_neg;
    /// Which space transformation is used to tune the parameters
    TuningSpace tuning_space;
  };

  static Eigen::VectorXd run(GradientFunc gradient_func,
                             const Eigen::VectorXd & initial_guess,
                             const Eigen::VectorXd & initial_step_size,
                             const Eigen::MatrixXd & limits,
                             std::shared_ptr<Config> conf);

  /// Conversion from the tuning space to user space
  static Eigen::VectorXd cvtFromTuningSpace(const Eigen::VectorXd & v,
                                            std::shared_ptr<Config> conf);
  /// Conversion from user space to tuning Space
  static Eigen::VectorXd cvtToTuningSpace(const Eigen::VectorXd & v,
                                          std::shared_ptr<Config> conf);

};

std::string to_string(RProp::TuningSpace tuning_space);
RProp::TuningSpace loadTuningSpace(const std::string & tuning_space);

}
