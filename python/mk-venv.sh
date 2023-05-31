#!/bin/bash

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
fi 

