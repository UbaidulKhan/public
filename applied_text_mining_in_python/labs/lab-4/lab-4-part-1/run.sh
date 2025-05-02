#!/bin/bash

clear
echo -e "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"


user="${USER}"
REQ_FILE="requirements.txt"
GIT_DIR="education.git/coursera/applied_text_mining_in_python/labs/lab-4/lab-4-part-1"

VENV_DIR="/Users/${user}/Development/github/${GIT_DIR}/.venv"
VENV_NAME="lab-4"
VENV="${VENV_DIR}/${VENV_NAME}"
PYTHON="python3.9"



if [ "$2" ]; then
	SCRIPT="$2"
else
	SCRIPT="lab4p1.py"
fi


# Create virtual environment
function init_venv() {

	if [ ! -e "${VENV}" ]; then
		echo -e "\n Creating new python virtual env: ${VENV}"
		mkdir -p ${VENV_DIR}
		cd ${VENV_DIR}
		${PYTHON} -m venv ${VENV_NAME}
		echo -e "Now run: source ${VENV}/bin/activate"

	else
		# echo -e "\n  Activating python virtual env: ${VENV}"
		# source "${VENV}/bin/activate"
		echo -e "\n  Now run: source ${VENV}/bin/activate\n"
	fi 
}

# Install requirements
function install_reqs() {
  if [ -e "${REQ_FILE}" ]; then
     echo -e "\n Installing requirements"
		 activate
     pip3 install -r requirements.txt
   fi
}


# Run python script
function run_script() {
  source "${VENV}/bin/activate"
  echo -e "Running python script: ${SCRIPT}\n"
  python3 $SCRIPT
}


# Activate python virtual environment
function activate() {
	echo -e "\n To activate environmnetn, run:\n\n   source ${VENV}/bin/activate\n"
  
  
}


# Show help function
function help() {
	echo -e "\n To activate environmnetn, run:\n\n   source ${VENV}/bin/activate\n"
  echo " Invalid argument. Please provide a value of {init | reqs | run }"
  
}


# Call the appropriate function based on the argument value
case "$1" in
    "--act")
        activate
        ;;
    "--init")
        init_venv
        ;;
    "--reqs")
        install_reqs
        ;;
    "--run")
        run_script
        ;;
    "--help")
        help
        ;;
     *)
        echo "Invalid argument. Please provide a value of { --act | --init | --reqs | --run | --help }"
        exit 1
        ;;
esac