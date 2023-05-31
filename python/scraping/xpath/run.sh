#!/bin/bash

if [ "$1" ]; then
  SCRIPT="$1"
	
  echo -e "Activating VENV"
  VENV_DIR="/Users/ukhan/Development/python-public.git/.venv/github-public"
  source "${VENV_DIR}/bin/activate"

  echo -e "Installing required modules"
  pip3 install -r requirements.txt

  clear
  echo -e "Running python script: ${SCRIPT}\n"
  python3 $SCRIPT
	
else
  echo -e "\n No script specified, exiting\n"
  exit
fi


