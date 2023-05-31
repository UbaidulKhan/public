#!/bin/bash

DIR="/Users/ukhan/Development/python-public.git/.venv"
VENV_NAME="github-public"

mkdir -p ${DIR}
cd ${DIR}
python3.9 -m venv ${VENV_NAME}
