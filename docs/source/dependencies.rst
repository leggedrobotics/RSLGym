Install Dependencies
====================

Folder Setup
------------
To avoid conflicts with other libs it is recommended to install everything locally. Therefore you will need two folders:

*  **WORKSPACE**: where you clone all the git repos (e.g. ~/rslgym_ws)
*  **LOCAL_INSTALL**: where you install all the libs (e.g. ~/rslgym_build)

Define the LOCAL_INSTALL variable in your .bashrc and add it to the library search path, i.e.
add the following lines to **~/.bashrc**::

    export LOCAL_INSTALL=/home/<user_account>/rslgym_build
    export LD_LIBRARY_PATH=$LOCAL_INSTALL/lib:$LD_LIBRARY_PATH

RaiSimLib
---------
Install the RaiSim rigid body simulator.

*  Clone repo and install additional dependencies::

      cd WORKSPACE
      git clone https://github.com/raisimTech/raisimLib.git
      sudo apt install cmake

*  Install python3 depencency::

    sudo apt install python3-dev


*  Follow RaiSim installation instructions: https://raisim.com/sections/Installation.html

*  To use RaiSim you need a valid license. You can apply for it here: https://raisim.com/sections/License.html

*  Place the license in **~/.raisim** and rename the file to **activation.raisim** (create the folder if necessary)

    *  The license can also be placed at another location. This needs to be specified when instantiating the vectorized environment in python, see rslgym/ wrapper/ script/ RaisimGymVecEnv.py.

    
RaiSimOgre
----------
Install the ogre visualizer for RaiSim.

* Follow the instructions here: https://github.com/raisimTech/raisimOgre

PyBind11
--------
Install the PyBind11 library::

    cd WORKSPACE/raisimLib/thirdParty/pybind11
    mkdir build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$LOCAL_INSTALL -DPYBIND11_TEST=FALSE
    make install