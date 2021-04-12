#ifndef SRC_RAISIMGYMENV_HPP
#define SRC_RAISIMGYMENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include "raisim/World.hpp"
#include "yaml-cpp/yaml.h"
#include "raisim/OgreVis.hpp"

#define __RSG_MAKE_STR(x) #x
#define _RSG_MAKE_STR(x) __RSG_MAKE_STR(x)
#define RSG_MAKE_STR(x) _RSG_MAKE_STR(x)

#define READ_YAML(a, b, c) RSFATAL_IF(!c, "Node "<<RSG_MAKE_STR(c)<<" doesn't exist") \
                           b = c.as<a>();

namespace raisimgym_env {

using Dtype=float;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;

class RaisimGymEnvBase {

 public:
  explicit RaisimGymEnvBase (std::string resourceDir, const YAML::Node& cfg) :
      resourceDir_(std::move(resourceDir)),
      cfg_(cfg) {
    world_ = std::make_unique<raisim::World>();
  }

  virtual ~RaisimGymEnvBase() = default;

  /////// implement these methods ///////
  virtual void init() = 0;
  virtual void reset() = 0;
  virtual void setSeed(int seed) = 0;
  virtual void observe(Eigen::Ref<EigenVec> ob) = 0;
  virtual float step(const Eigen::Ref<EigenVec>& action) = 0;
  virtual bool isTerminalState(float& terminalReward) = 0;
  virtual void setInfo(const std::unordered_map<std::string, EigenVec>& info) = 0;
  ////////////////////////////////////////

  /////// optional methods ///////
  virtual void close() {};
  virtual void updateInfo() {};
  ////////////////////////////////

  void setSimulationTimeStep(double dt) {
    simulation_dt_ = dt;
    world_->setTimeStep(dt);
  }

  void setControlTimeStep(double dt) { 
    control_dt_ = dt;
  }

  virtual void startRecordingVideo(const std::string& fileName) {
    raisim::OgreVis::get()->startRecordingVideo(fileName);
  }

  virtual void stopRecordingVideo() {
    raisim::OgreVis::get()->stopRecordingVideoAndSave();
  }

  std::unordered_map<std::string, EigenVec> getInfo() {
    return info_;
  }

  raisim::World* getWorld() { return world_.get(); }
  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  double getControlTimeStep() { return control_dt_; }
  double getSimulationTimeStep() { return simulation_dt_; }
  int getInfoDim() { return info_.size(); }
  void turnOnVisualization() { visualizeThisStep_ = true; }
  void turnOffvisualization() { visualizeThisStep_ = false; }

 protected:
  std::unordered_map<std::string, EigenVec> info_;
  std::unique_ptr<raisim::World> world_;
  double simulation_dt_ = 0.001;
  double control_dt_ = 0.01;
  std::string resourceDir_;
  bool visualizeThisStep_ = false;
  YAML::Node cfg_;
  int obDim_ = 0, actionDim_ = 0;
};

}

#endif //SRC_RAISIMGYMENV_HPP
