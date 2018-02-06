#pragma once

#include "rhoban_gp/gradient_ascent/rprop.h"

namespace rhoban_gp
{

/// Run several gradient ascent and keep the best
class RandomizedRProp {
public:
  typedef std::function<double(const Eigen::VectorXd)> ScoringFunc;

  class Config : public rhoban_utils::JsonSerializable {
  public:
    Config();

    virtual std::string getClassName() const override;
    virtual Json::Value toJson() const override;
    virtual void fromJson(const Json::Value & v, const std::string & dir_name) override;

    /// Avoiding that the functions from Serializable get hidden
    using rhoban_utils::JsonSerializable::read;

    /// Write a binary stream saving the configuration of the node and all its children
    /// Return the number of bytes written
    int write(std::ostream & out) const;
    /// Read the configuration of the node and all its children from the provided binary stream
    /// Return the number of bytes read
    int read(std::istream & in);

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
