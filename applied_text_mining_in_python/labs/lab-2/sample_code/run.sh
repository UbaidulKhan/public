#!/bin/bash

PY_SCRIPT="$1"
# PY_SCRIPT="geeks_for_geeks.py"

#
## List of dependencies:
##
MODULEs="nltk pandas numpy pprintpp"

## Clear the screen 
clear

#
## Install MODULEs
#
for MODULE in $MODULES; do
  if(conda list | grep $MODULE); then
    echo -e " > $MODULE already installed"
  else
    echo -e " > Installing MODULE: $MODULE"
	  conda install $MODULE
  fi
done

	
echo -e "Running script: ${PY_SCRIPT}"
pipenv run python ${PY_SCRIPT}

#
## Run in debugger
# pipenv run python -m pdb ${PY_SCRIPT}
