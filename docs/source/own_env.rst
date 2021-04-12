Create Your Own Environment
============================

You need to implement a single environment instance. The frameworks parallelizes this single environment by creating a **vectorized environment**.

Your environment needs to have at least Environment.hpp which inherits from **rslgym/rslgym/wrapper/include/RaisimGymEnvBase.hpp**:

.. code-block:: cpp

   virtual void init() = 0;
   virtual void reset() = 0;  // resets environment at the beginning or when terminated
   virtual void setSeed(int seed) = 0;  // sets random number generator(s) seed(s)
   virtual void observe(Eigen::Ref<EigenVec> ob) = 0;  // returns environment observations
   virtual float step(const Eigen::Ref<EigenVec>& action) = 0;  // returns reward
   virtual bool isTerminalState(float& terminalReward) = 0;  //substitutes terminal reward
   virtual void setInfo(const std::unordered_map<std::string, EigenVec>& info) = 0;  // transfer info to the environment
   virtual void updateInfo() {};  //transfer info from the environment


**setInfo()** and **updateInfo()** allow you to transfer information from the python code to the environment and the other way around. This can be useful to monitor variables in the environment or to update curriculum factors in the environment.

You can define any additional info to this variable like below. This will be a dictionary on the python side:

.. code-block:: cpp

   void updateInfo() final {
     info_["gc"] = gc_.cast<float>();
     info_["gv"] = gv_.cast<float>();
     info_["rewards"] = Eigen::Vector2d(torqueReward_, forwardVelReward_).cast<float>();
   }

In the same way, you can get python dictionary data from python side in Environment.hpp:

.. code-block:: cpp

   void setInfo(const std::unordered_map<std::string, EigenVec>& info) {
     for (auto &kv: info) {
       const auto& key = kv.first;
       const auto& value = kv.second;
       // do whatever based on info from python
       // if(key == "curriculum")
       //    updateCurriculum(value);
     }
   }

Building, Naming and Usage
^^^^^^^^^^^^^^^^^^^^^^^^^^
You can define the name under which the python module will be built in the **environment.hpp**.

.. code-block:: cpp

   #define ENVIRONMENT_NAME <my-env-name>

When building the environment, you can pass the name of the python package which will be called rslgym_wrapper_<your_package_name>::

   rslgym build . --name <your_package_name> --CMAKE_PREFIX_PATH $LOCAL_INSTALL

Then, in python, you can include the environment using:

.. code-block:: python

   from rslgym_wrapper_<your_package_name> import <my-env-name>

For more information about the **build** function, call::

    rslgym build --help

Debugging
^^^^^^^^^
To debug your environment.hpp and catch e.g. nasty segmentation faults you can build your environment using the **--debug** flag::

    rslgym build . --name cartpole --debug --CMAKE_PREFIX_PATH $LOCAL_INSTALL

This will create a c++ executable which you can debug with **valgrind** using this command::

    rslgym debug <render/no_render> --resource <relative-path-to-rsc-folder-default:/rsc> --cfg <relative-path_to_cfg-default:/cfg.yaml>

For more information about the **debug** function, call::

    rslgym debug --help

Additional Libraries
^^^^^^^^^^^^^^^^^^^^^

RSLGym includes and links **raisim**, **eigen3** and **OpenMP** libraries.
If you want to use **additional libraries** in your environment, you can add prebuilt or custom libraries to the CMake variable `EXTRA_LIBS` through a CMake include file.
Note that the libraries must be built with **-fPIC** option::

   rslgym build . --name <your_package_name> --CMAKE_INCLUDE_FILE <path_to_my_cmake_include_file>

An example of a CMake include file to build custom libraries can be found below. 

.. code-block:: cmake

   add_library(<MY_LIBRARY_NAME> ${CMAKE_CURRENT_LIST_DIR}/<path_to_my_source_code_relative_to_this_file>.cpp)
   target_include_directories(<MY_LIBRARY_NAME> PUBLIC ${CMAKE_CURRENT_LIST_DIR}/<path_to_my_include_directories>)
   target_compile_options(<MY_LIBRARY_NAME> PRIVATE -mtune=native -fPIC -O3)
   set(EXTRA_LIBS <MY_LIBRARY_NAME>)
