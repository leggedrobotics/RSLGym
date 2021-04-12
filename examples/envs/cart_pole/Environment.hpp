#include <stdlib.h>
#include <cstdint>
#include <set>
#include <raisim/OgreVis.hpp>
#include "RaisimGymEnvBase.hpp"

#include "visualizer/visSetupCallback.hpp"
#include "visualizer/raisimKeyboardCallback.hpp"
#include "visualizer/helper.hpp"
#include "visualizer/guiState.hpp"
#include "visualizer/raisimBasicImguiPanel.hpp"

// Python module will be compiled to this name
#define ENVIRONMENT_NAME cart_pole_example_env

namespace raisimgym_env {

class ENVIRONMENT : public RaisimGymEnvBase {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const YAML::Node& cfg, bool visualizable) :
      RaisimGymEnvBase(resourceDir, cfg), visualizable_(visualizable), uni01_(0.0, 1.0) {

    /// add objects
    robot_ = world_->addArticulatedSystem(resourceDir_+"/cart_pole.urdf");

    /// get robot data
    gcDim_ = robot_->getGeneralizedCoordinateDim();
    gvDim_ = robot_->getDOF();

    /// MUST BE DONE FOR ALL ENVIRONMENTS
    obDim_ = 4; // cart pos, pole ang, cart vel, pol ang vel
    actionDim_ = 1; // force to cart

    /// initialize containers
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    scaledActionClipped_.setZero(actionDim_);

    /// action & observation scaling
    actionMean_.setZero(actionDim_); actionStd_.setOnes(actionDim_);
    actionStd_ *= 5.0;
    obMean_.setZero(obDim_); obStd_.setOnes(obDim_);
    obStd_(2) = 2.0; // cart vel
    obStd_(3) = 2.0; // pend vel
    obStd_(1) = 0.5; // pend pos
    obStd_(0) = 2.0; // cart pos


    /// Reward coefficients
    READ_YAML(double, balancingRewardCoeff_, cfg["balancing_reward_coeff"])
    READ_YAML(double, actionRewardCoeff_, cfg["action_reward_coeff"])
    READ_YAML(double, terminalRewardCoeff_, cfg["terminal_reward_coeff"])

    raisim::gui::rewardLogger.init({"balancing", "action"});
    auto ground = world_->addGround();

    updateInfo();

    /// visualize if it is the first environment
    if (visualizable_) {
      auto vis = raisim::OgreVis::get();

      /// these method must be called before initApp
      vis->setWorld(world_.get());
      vis->setWindowSize(1280, 720);
      vis->setImguiSetupCallback(imguiSetupCallback);
      vis->setImguiRenderCallback(imguiRenderCallBack);
      vis->setKeyboardCallback(raisimKeyboardCallback);
      vis->setSetUpCallback(setupCallback);
      vis->setAntiAliasing(2);

      /// starts visualizer thread
      vis->initApp();
      vis->setDesiredFPS(visDesiredFps_);

      /// create robot visualization
      robotVisual_ = vis->createGraphicalObject(robot_, "cart_pole");

      /// set camera
      vis->select(robotVisual_->at(0), false);
      vis->getCameraMan()->setYawPitchDist(Ogre::Radian(3.14), Ogre::Radian(-1.6), 8, true);
      vis->createGraphicalObject(ground, 20, "floor", "checkerboard_green");
      // vis->getSceneManager()->setSkyBox(true, "white", 20);
    }

  }

  ~ENVIRONMENT() final = default;

  void init() final { }

  void reset() final {
    gc_init_(0) = 3.0*(uni01_(gen_) - 0.5); // +/- 1.0m from center
    gc_init_(1) = uni01_(gen_) - 0.3; // +/- 15° from balancing point
    robot_->setState(gc_init_, gv_init_);
    updateObservation();
    updateInfo();
    if(visualizable_)
      raisim::gui::rewardLogger.clean();
  }

  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    scaledAction_ = action.cast<double>();
    scaledAction_ = scaledAction_.cwiseProduct(actionStd_) + actionMean_;

    /// apply & clip action
    scaledActionClipped_ = scaledAction_.array().min(20.0).max(-20.0);
    Eigen::Vector2d genForce;
    genForce << scaledActionClipped_, 0.0; // pendulum joint is not actuated
    robot_->setGeneralizedForce(genForce);

    /// run sim
    auto loopCount = int(control_dt_ / simulation_dt_ + 1e-10);
    auto visDecimation = int(1. / (visDesiredFps_ * simulation_dt_) + 1e-10);

    for(int i=0; i<loopCount; i++) {
      world_->integrate();
      if (visualizable_ && visualizeThisStep_ && visualizationCounter_ % visDecimation == 0)
        raisim::OgreVis::get()->renderOneFrame();

      visualizationCounter_++;
    }

    updateObservation();
    updateInfo();

//    balancingReward_ = errorToUpright < 0.1 ? 0.01 : 0.0;
    balancingReward_ = balancingRewardCoeff_ * errorToUpright_ * errorToUpright_;
    actionReward_ = actionRewardCoeff_ * scaledAction_(0)*scaledAction_(0); // not the clipped action!

    if(visualizeThisStep_) {
      raisim::gui::rewardLogger.log("balancing", balancingReward_);
      raisim::gui::rewardLogger.log("action", actionReward_);
    }

    return balancingReward_ + actionReward_ + 0.5;
  }

  void updateInfo() final {
    /// set any additional infomation here
    EigenVec rewardTerms(2), cartState(2), poleState(2);
    rewardTerms <<  balancingReward_, actionReward_;
    cartState << gc_(0), gv_(0);
    poleState << gc_(1), gv_(1);

    info_["rewards"] = rewardTerms;
    info_["cart_state"] = cartState;
    info_["pole_state"] = poleState;
    info_["action"] = scaledActionClipped_.cast<Dtype>();
    info_["error_to_upright"] = EigenVec(1).setOnes() * (errorToUpright_) ;
  }

  void setInfo(const std::unordered_map<std::string, EigenVec>& info) {
  }

  void updateObservation() {
    robot_->getState(gc_, gv_);
    errorToUpright_ = abs(gc_(1));

    obDouble_.setZero(obDim_); obScaled_.setZero(obDim_);
    obDouble_ << gc_, gv_;

    obScaled_ = (obDouble_-obMean_).cwiseQuotient(obStd_);
  }

  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to Dtype (float)
    ob = obScaled_.cast<Dtype>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    // if cart runs away
    if (abs(gc_(0)) > 2.5 )
      return true;

    if (errorToUpright_> 0.5 ) // terminate if +/- 45° from balancing point
      return true;

    terminalReward = 0.f;
    return false;
  }

  void setSeed(int seed) final {
    std::srand(seed);
    gen_.seed(seed);
  }

  void close() final {
  }

 private:
  int gcDim_, gvDim_;
  bool visualizable_ = false;
  raisim::ArticulatedSystem* robot_;
  std::vector<raisim::GraphicObject>* robotVisual_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, scaledActionClipped_ , scaledAction_;
  double terminalRewardCoeff_ = 0.;
  double balancingReward_ = 0., balancingRewardCoeff_ = 0.;
  double actionReward_ = 0., actionRewardCoeff_ = 0.;
  double errorToUpright_ = 0.0;
  double visDesiredFps_ = 50.;
  int visualizationCounter_ = 0;
  Eigen::VectorXd actionMean_, actionStd_, obMean_, obStd_;
  Eigen::VectorXd obDouble_, obScaled_;
  std::uniform_real_distribution<double> uni01_;
  std::mt19937 gen_;
};
}
