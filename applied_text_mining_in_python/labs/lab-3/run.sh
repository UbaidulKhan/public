#!/bin/bash

if [ "$1" ]; then
	SCRIPT="$1"
else
	SCRIPT="lab3.py"
# SCRIPT="lab3-working.py"
fi


clear
echo -e "Activating VENV"
source /Users/ukhan/Development/github/education.git/coursera/applied_text_mining_in_python/venv/bin/activate

# echo -e "Installing modules\n"
# for module in numpy pandas sklearn; do
#   pip3 install $module
# done

echo -e "Unzipping sample data\n"
# unzip -f data/archive.zip

echo -e "Running python script: ${SCRIPT}\n"
python3 $SCRIPT
