#ifndef RAISIMOGREVISUALIZER_RAISIMKEYBOARDCALLBACK_HPP
#define RAISIMOGREVISUALIZER_RAISIMKEYBOARDCALLBACK_HPP

#include "raisim/OgreVis.hpp"
#include "raisimKeyboardCallback.hpp"
#include "guiState.hpp"

bool raisimKeyboardCallback(const OgreBites::KeyboardEvent &evt) {
  auto &key = evt.keysym.sym;
  // termination gets the highest priority
  switch (key) {
    case '1':
      raisim::gui::showBodies = !raisim::gui::showBodies;
      break;
    case '2':
      raisim::gui::showCollision = !raisim::gui::showCollision;
      break;
    case '3':
      raisim::gui::showContacts = !raisim::gui::showContacts;
      break;
    case '4':
      raisim::gui::showForces = !raisim::gui::showForces;
      break;
    default:
      break;
  }
  return false;
}

#endif //RAISIMOGREVISUALIZER_RAISIMKEYBOARDCALLBACK_HPP
