#!/bin/bash

clear
echo -e "\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n"


user="${USER}"
REQ_FILE="requirements.txt"
GIT_DIR="public.git/python"

HOME_DIR=""
if [ "`uname`" == "Linux" ]; then
	HOME_DIR="/home"
else
	HOME_DIR="/Users"
fi

VENV_DIR="${HOME_DIR}/${user}/Development/github/${GIT_DIR}/.venv"
VENV_NAME="python-auto-install"
VENV="${VENV_DIR}/${VENV_NAME}"
PYTHON="python"

SCRIPT=""
PYTHON_SCRIPT_ARGS="${@:3}"


if [ "$2" ]; then
	SCRIPT="$2"
	# echo -e "Python script to run: ${SCRIPT}"
else
	SCRIPT="make_boot_iso.py"
fi

# if [ "${@:3}" ]; then
#   PYTHON_SCRIPT_ARGS="${@:3}"
# 	echo -e "Python script arguments: ${PYTHON_SCRIPT_ARGS}"
# fi


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
		 # poetry now installed via pip - see requirements.txt
		 # curl -sSL https://install.python-poetry.org | python3 -
   fi
}


# Run python script
function run_script() {
  source "${VENV}/bin/activate"
	PWD="`pwd`"
	export PYTHONPATH="${PYTHONPATH}:${PWD}"
  echo -e "Setting PYTHONPATH to: ${PYTHONPATH}\n"
  echo -e "Running python script: ${SCRIPT}\n"
  python3 $SCRIPT
}


# Run python script
function run_script_new() {
	# echo -e "Run script...."
	# echo -e "  Invoking python script:  ${SCRIPT}"
	# echo -e "  With parameters: ${PYTHON_SCRIPT_ARGS}"
	# echo -e "........................................."
  source "${VENV}/bin/activate"
	PWD="`pwd`"
	export PYTHONPATH="${PYTHONPATH}:${PWD}"
  echo -e "Setting PYTHONPATH to: ${PYTHONPATH}\n"
  echo -e "Running python script: ${SCRIPT}\n"
	
  # python3 $SCRIPT "${@:2}"
  # python3 $SCRIPT "${@:3}"
	
	python3 ${SCRIPT} ${PYTHON_SCRIPT_ARGS}

}


# Activate python virtual environment
function activate() {
	echo -e "\n To activate environmnetn, run:\n\n   source ${VENV}/bin/activate\n"
  
  
}


# Show help function
function help() {
	echo -e "\n To activate environmnetn, run:\n\n   source ${VENV}/bin/activate\n"
  echo " Invalid argument. Please provide a value of {--init | --reqs | --run }"
	echo " "
  
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
        # run_script
				run_script_new
        ;;
    "--help")
        help
        ;;
     *)
        echo "Invalid argument. Please provide a value of { --act | --init | --reqs | --run | --help }"
        exit 1
        ;;
esac