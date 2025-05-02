#!/bin/bash

# PY_SCRIPT="lab2.py"
PY_SCRIPT="lab2.py"

#
## List of dependencies:
##
MODULEs="collections nltk pandas numpy pprintpp"

## Clear the screen 
clear

#
## Install MODULEs
#
install_moduels_2() {
  for MODULE in $MODULES; do
    if(conda list | grep $MODULE); then
      echo -e " > $MODULE already installed"
    else
      echo -e " > Installing MODULE: $MODULE"
  	  conda install $MODULE
    fi
  done
}

install_modules() {
  for module in nltk pandas; do
		echo -e "Installing module: ${module}"
    # pipenv run pip install $module
    pip install $module
  done
}

echo -e "Installing python modules"
install_moduels_2
	
echo -e "Running script: ${PY_SCRIPT}"
python ${PY_SCRIPT}

#
## Run in debugger
# pipenv run python -m pdb ${PY_SCRIPT}
