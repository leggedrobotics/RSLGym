Install RSLGym
===============

Virtualenv
^^^^^^^^^^^
To avoid conflicts with other libs it is recommended to install everything locally in a **virtual environment**::

    pip3 install virtualenv
    mkdir ~/.virtualenvs
    pip3 install virtualenvwrapper

Add the following lines to **~/.bashrc**::

     export WORKON_HOME=~/.virtualenvs
     export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
     source ~/.local/bin/virtualenvwrapper.sh

Open a new terminal tab and create your virtual environment::

    source ~/.profile
    mkvirtualenv --system-site-packages rslgym

PyTorch
^^^^^^^^^^^^
Activate the virtualenvironment (if not already active)::

    workon rslgym

Install the latest stable version of PyTorch using pip following the instructions here https://pytorch.org/get-started/locally/


RSLGym
^^^^^^^^^^^^
Clone and install RSLGym (inside the virtual environment)::

    cd $WORKSPACE
    git clone <rslgym_repo>
    cd rslgym
    pip3 install -e .


Other Dependencies
^^^^^^^^^^^^^^^^^^^
Yaml-cpp for hyperparameter loading::

    sudo apt install libyaml-cpp-dev

Dependencies for openAI examples::

    sudo apt install ubuntu-restricted-extras swig

Valgrind for debugging **environment.hpp**::

    sudo apt install valgrind
