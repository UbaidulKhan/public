#!/bin/bash

if [ "$1" ]; then
  SCRIPT="$1"
	
  user="${USER}"
  REQ_FILE="requirements.txt"
  GIT_DIR="github-public.git"
  
  VENV_DIR="/Users/${user}/Development/${GIT_DIR}/.venv"
  VENV_NAME="python-venv"
  VENV="${VENV_DIR}/${VENV_NAME}"
  PYTHON="python3.9"

  if [ ! -e "${VENV}" ]; then
    echo -e "\n Creating new python virtual env: ${VENV}"
    mkdir -p ${VENV_DIR}
    cd ${VENV_DIR}
    ${PYTHON} -m venv ${VENV_NAME}
  else
    echo -e "\n  Activating python virtual env: ${VENV}"
    source "${VENV}/bin/activate"
  fi 
  

  echo -e "  Installing required modules"
  if [ -e "${REQ_FILE}" ]; then
    echo -e "\n Installing requirements"
    pip3 install -r requirements.txt
  fi

  # clear
  echo -e "  Running python script: ${SCRIPT}"
  echo -e "--------------------------------------------------------------------------"
  python3 $SCRIPT
	
else
  echo -e "\n  No script specified, exiting\n"
  exit
fi


