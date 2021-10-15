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

Pytorch
^^^^^^^^^^^^
Activate the virtualenvironment (if not already active)::

    workon rslgym

If you have a GPU::
    
    pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

For CPU only::
    
    pip3 install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html


RSLGym
^^^^^^^^^^^^
Install RSLGym (inside the virtual environment)::

    cd rslgym
    pip3 install -e .


Other Dependencies
^^^^^^^^^^^^^^^^^^^
Yaml-cpp for hyperparameter loading::

    sudo apt install libyaml-cpp-dev

Codecs for openAI video playing in standard ubuntu player::

    sudo apt install ubuntu-restricted-extras

Valgrind for debugging **environment.hpp**::

    sudo apt install valgrind
