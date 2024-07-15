#!/bin/bash

python3 -m venv ./venv
source ./venv/bin/activate

pip install torch

find / -name "TorchConfig.cmake" 2>/dev/null

sudo apt-get update
sudo apt-get install libssl-dev ccache libboost-all-dev python3.10-venv

# Define the hardcoded directory
TARGET_DIR="/path/to/your/directory"

# Define the repository URL
REPO_URL="https://github.com/Kitware/CMake.git"

# Navigate to the target directory
cd $TARGET_DIR

# Clone the repository
git clone $REPO_URL

# Navigate to the cloned repository directory
cd CMake

mkdir Build && cd Build

# Run the bootstrap script
../bootstrap

# Make and make install
make
sudo make install

echo "CMake has been successfully installed."