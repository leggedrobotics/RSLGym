#ifndef RAISIMOGREVISUALIZER_GUISTATE_HPP
#define RAISIMOGREVISUALIZER_GUISTATE_HPP
#include "RewardLogger.hpp"

namespace raisim{
namespace gui {
static bool manualStepping = false;
static bool showBodies = true;
static bool showCollision = false;
static bool showContacts = false;
static bool showForces = false;
static RewardLogger rewardLogger;

}
}

#endif //RAISIMOGREVISUALIZER_GUISTATE_HPP
