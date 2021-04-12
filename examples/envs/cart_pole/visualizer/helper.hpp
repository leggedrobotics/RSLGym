#ifndef RAISIMOGREVISUALIZER_HELPER_HPP
#define RAISIMOGREVISUALIZER_HELPER_HPP

#include <string>
#define MAKE_STR(x) #x

namespace raisim {

inline std::string loadResource (const std::string& file) {
  return std::string(MAKE_STR(EXAMPLE_ROBOT_RESOURCE_DIR))+file;
}

}


#endif //RAISIMOGREVISUALIZER_HELPER_HPP
