#!/bin/bash

# SCRIPT="case-study-amazon-review-bruce-zhang.py"
SCRIPT="case-study-amazon-review.py"

echo -e "Activating VENV"
source /Users/ukhan/Development/github/education.git/coursera/applied_text_mining_in_python/venv/bin/activate

echo -e "Installing modules\n"
for module in numpy pandas sklearn; do
  pip3 install $module
done

echo -e "Unzipping sample data\n"
# unzip -f data/archive.zip
unzip data/archive.zip -d data

echo -e "Running python script\n"
python3 $SCRIPT
