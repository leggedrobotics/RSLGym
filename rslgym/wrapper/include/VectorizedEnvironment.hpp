#ifndef SRC_RAISIMGYMVECENV_HPP
#define SRC_RAISIMGYMVECENV_HPP

#include "RaisimGymEnvBase.hpp"
#include "omp.h"
#include "yaml-cpp/yaml.h"

namespace raisimgym_env {

template<class ChildEnvironment>
class VectorizedEnvironment {

 public:

  explicit VectorizedEnvironment(std::string resourceDir, std::string cfg_str)
      : resourceDir_(resourceDir) {
    cfg_ = YAML::Load(cfg_str);
  }

  ~VectorizedEnvironment() {
    for (auto *ptr: environments_)
      delete ptr;
  }

  void setActivationKey(std::string path) {
    raisim::World::setActivationKey(raisim::Path(path).getString());
  }

  void init() {
    omp_set_num_threads(cfg_["num_threads"].template as<int>());
    num_envs_ = cfg_["num_envs"].template as<int>();
    render_ = cfg_["render"].template as<bool>();

    for (int i = 0; i < num_envs_; i++) {
      environments_.push_back(new ChildEnvironment(resourceDir_, cfg_, render_ && i == 0));
      environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template as<double>());
      environments_.back()->setControlTimeStep(cfg_["control_dt"].template as<double>());
    }

    setSeed(cfg_["seed"].template as<double>());

    if(render_) raisim::OgreVis::get()->hideWindow();

#pragma omp parallel for schedule(static)
    for (int i = 0; i < num_envs_; i++) {
      // only the first environment is visualized
      environments_[i]->init();
      environments_[i]->reset();
    }

    obDim_ = environments_[0]->getObDim();
    actionDim_ = environments_[0]->getActionDim();
    RSFATAL_IF(obDim_ == 0 || actionDim_ == 0, "Observation/Action dimension must be defined in the constructor of each environment!")

    /// generate reward names
    /// compute it once to get reward names. actual value is not used
    environments_[0]->updateInfo();
    for (auto &kv: environments_[0]->getInfo()) {
      infoNames_.push_back(kv.first);
      infoDims_.push_back(kv.second.size());
      info_[kv.first] = EigenRowMajorMat::Zero(num_envs_, kv.second.size());
    }
  }

  std::vector<std::string> &getInfoNames() {
    return infoNames_;
  }

  std::vector<int> &getInfoDims() {
    return infoDims_;
  }

  // resets all environments and returns observation
  void reset(Eigen::Ref<EigenRowMajorMat>& ob) {
    for (auto env: environments_)
      env->reset();

    observe(ob);
  }

  void observe(Eigen::Ref<EigenRowMajorMat> &ob) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++)
      environments_[i]->observe(ob.row(i));
  }

  void step(Eigen::Ref<EigenRowMajorMat> &action,
            Eigen::Ref<EigenRowMajorMat> &ob,
            Eigen::Ref<EigenVec> &reward,
            Eigen::Ref<EigenBoolVec> &done) {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++) {
      perAgentStep(i, action, ob, reward, done);
    }
  }

  void testStep(Eigen::Ref<EigenRowMajorMat> &action,
                Eigen::Ref<EigenRowMajorMat> &ob,
                Eigen::Ref<EigenVec> &reward,
                Eigen::Ref<EigenBoolVec> &done) {
    if(render_) environments_[0]->turnOnVisualization();
    perAgentStep(0, action, ob, reward, done);
    if(render_) environments_[0]->turnOffvisualization();
  }

  void startRecordingVideo(const std::string& fileName) {
    if(render_) environments_[0]->startRecordingVideo(fileName);
  }

  void stopRecordingVideo() {
    if(render_) environments_[0]->stopRecordingVideo();
  }

  void showWindow() {
    raisim::OgreVis::get()->showWindow();
  }

  void hideWindow() {
    raisim::OgreVis::get()->hideWindow();
  }

  void setSeed(int seed) {
    int seed_inc = seed;
    for (auto *env: environments_)
      env->setSeed(seed_inc++);
  }

  void close() {
    for (auto *env: environments_)
      env->close();
  }

  void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
    for (int i = 0; i < num_envs_; i++) {
      float terminalReward;
      terminalState[i] = environments_[i]->isTerminalState(terminalReward);
    }
  }

  void setSimulationTimeStep(double dt) {
    for (auto *env: environments_)
      env->setSimulationTimeStep(dt);
  }

  void setControlTimeStep(double dt) {
    for (auto *env: environments_)
      env->setControlTimeStep(dt);
  }

  void setInfo(std::unordered_map<std::string, Eigen::Ref<EigenRowMajorMat>> &info) {
// #pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++) {
      std::unordered_map<std::string, EigenVec> info_i;
      for (auto& kv : info) {
        auto& key = kv.first;
        info_i[key] = kv.second.row(i);
      }
      environments_[i]->setInfo(info_i);
    }
  }

  std::unordered_map<std::string, EigenRowMajorMat> getInfo() {
#pragma omp parallel for schedule(dynamic)
    for (int i = 0; i < num_envs_; i++) {
      environments_[i]->updateInfo();
      for (auto& kv : environments_[i]->getInfo()) {
        // transpose is to convert to rowmajor matrix
        info_[kv.first].row(i) << kv.second.transpose();
      }
    }
    return info_;
  }

  int getObDim() { return obDim_; }
  int getActionDim() { return actionDim_; }
  int getNumOfEnvs() { return num_envs_; }

 private:

  inline void perAgentStep(int agentId,
                           Eigen::Ref<EigenRowMajorMat> &action,
                           Eigen::Ref<EigenRowMajorMat> &ob,
                           Eigen::Ref<EigenVec> &reward,
                           Eigen::Ref<EigenBoolVec> &done) {
    reward[agentId] = environments_[agentId]->step(action.row(agentId));

    float terminalReward = 0;
    done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

    if (done[agentId]) {
      environments_[agentId]->reset();
      reward[agentId] += terminalReward;
    }

    environments_[agentId]->observe(ob.row(agentId));
  }

  std::vector<ChildEnvironment *> environments_;
  std::vector<std::string> infoNames_;
  std::vector<int> infoDims_;

  int num_envs_ = 1;
  int obDim_ = 0, actionDim_ = 0;
  bool recordVideo_=false, render_=false;
  std::string resourceDir_;
  YAML::Node cfg_;
  std::unordered_map<std::string, EigenRowMajorMat> info_;
};

}

#endif //SRC_RAISIMGYMVECENV_HPP
