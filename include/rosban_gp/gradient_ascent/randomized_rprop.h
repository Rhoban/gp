#pragma once

#include "rosban_gp/gradient_ascent/rprop.h"

namespace rosban_gp
{

/// Run several gradient ascent and keep the best
class RandomizedRProp {
public:
  typedef std::function<double(const Eigen::VectorXd)> ScoringFunc;

  class Config : public rosban_utils::Serializable {
  public:
    Config();

    virtual std::string class_name() const override;
    virtual void to_xml(std::ostream &out) const override;
    virtual void from_xml(TiXmlNode *node) override;

    /// Number of gradient ascent performed
    int nb_trials;
    /// Configuration used for rProp
    std::shared_ptr<RProp::Config> rprop_conf;
  };

  static Eigen::VectorXd run(RProp::GradientFunc gradient_func,
                             ScoringFunc scoring_func,
                             const Eigen::MatrixXd & limits,
                             const Config & conf);
};

}
