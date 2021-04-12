#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "Environment.hpp"
#include "VectorizedEnvironment.hpp"

namespace py = pybind11;
using namespace raisimgym_env;

#ifndef ENVIRONMENT_NAME
  #define ENVIRONMENT_NAME VectorizedEnvironment
#endif

#ifndef MODULE_NAME
  #define MODULE_NAME _raisim_gym_wrapper
#endif

PYBIND11_MODULE(MODULE_NAME, m) {
  py::class_<VectorizedEnvironment<ENVIRONMENT>>(m, RSG_MAKE_STR(ENVIRONMENT_NAME))
    .def(py::init<std::string, std::string>())
    .def("init", &VectorizedEnvironment<ENVIRONMENT>::init)
    .def("setActivationKey", &VectorizedEnvironment<ENVIRONMENT>::setActivationKey)
    .def("getInfoNames", &VectorizedEnvironment<ENVIRONMENT>::getInfoNames)
    .def("getInfoDims", &VectorizedEnvironment<ENVIRONMENT>::getInfoDims)
    .def("reset", &VectorizedEnvironment<ENVIRONMENT>::reset)
    .def("observe", &VectorizedEnvironment<ENVIRONMENT>::observe)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("testStep", &VectorizedEnvironment<ENVIRONMENT>::testStep)
    .def("setSeed", &VectorizedEnvironment<ENVIRONMENT>::setSeed)
    .def("step", &VectorizedEnvironment<ENVIRONMENT>::step)
    .def("close", &VectorizedEnvironment<ENVIRONMENT>::close)
    .def("isTerminalState", &VectorizedEnvironment<ENVIRONMENT>::isTerminalState)
    .def("setSimulationTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setSimulationTimeStep)
    .def("setControlTimeStep", &VectorizedEnvironment<ENVIRONMENT>::setControlTimeStep)
    .def("getObDim", &VectorizedEnvironment<ENVIRONMENT>::getObDim)
    .def("getActionDim", &VectorizedEnvironment<ENVIRONMENT>::getActionDim)
    .def("getNumOfEnvs", &VectorizedEnvironment<ENVIRONMENT>::getNumOfEnvs)
    .def("startRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::startRecordingVideo)
    .def("stopRecordingVideo", &VectorizedEnvironment<ENVIRONMENT>::stopRecordingVideo)
    .def("showWindow", &VectorizedEnvironment<ENVIRONMENT>::showWindow)
    .def("hideWindow", &VectorizedEnvironment<ENVIRONMENT>::hideWindow)
    .def("getInfo", &VectorizedEnvironment<ENVIRONMENT>::getInfo)
    .def("setInfo", &VectorizedEnvironment<ENVIRONMENT>::setInfo);

}
