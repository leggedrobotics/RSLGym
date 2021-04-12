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

    mkvirtualenv --system-site-packages rslgym

Pytorch
^^^^^^^^^^^^
Activate the virtualenvironment (if not already active)::

    workon rslgym

If you have a GPU::
    
    pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html

For CPU only::
    
    pip install torch==1.7.1+cpu torchvision==0.8.2+cpu torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html


RSLGym
^^^^^^^^^^^^
Install RSLGym (inside the virtual environment)::

    cd rslgym
    pip3 install .


Other Dependencies
^^^^^^^^^^^^^^^^^^^
Yaml-cpp for hyperparameter loading::

    sudo apt install libyaml-cpp-dev

Codecs for openAI video playing in standard ubuntu player::

    sudo apt install ubuntu-restricted-extras

Valgrind for debugging **environment.hpp**::

    sudo apt install valgrind
