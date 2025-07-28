#!/bin/bash
# setup_env.sh - Bash script to set up your Python environment

# Exit on error
set -e

# Detect current python3 version
PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
VENV_PKG="python${PYTHON_VERSION}-venv"

# Ensure venv for current python3 version is installed
if ! python3 -m venv --help &> /dev/null; then
    echo "Installing $VENV_PKG..."
    sudo apt-get update
    sudo apt-get install -y $VENV_PKG
fi

# Create virtual environment for current python3 if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "Virtual environment .venv (python3) created."
fi

# Create virtual environment for python3.10 if available
if ! command -v python3.10 &> /dev/null; then
    echo "python3.10 not found. Installing from source..."
    sudo apt-get update
    sudo apt-get install -y build-essential libssl-dev zlib1g-dev poppler-utils \
        libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
        libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev tk-dev \
        libffi-dev wget
    wget https://www.python.org/ftp/python/3.10.14/Python-3.10.14.tgz
    tar -xf Python-3.10.14.tgz
    cd Python-3.10.14
    ./configure --enable-optimizations
    make -j$(nproc)
    sudo make altinstall
    cd ..
    rm -rf Python-3.10.14 Python-3.10.14.tgz
    echo "python3.10 installed from source."
fi

if command -v python3.10 &> /dev/null; then
    if [ ! -d ".venv3.10" ]; then
        if ! python3.10 -m venv --help &> /dev/null; then
            echo "Installing python3.10-venv..."
            sudo apt-get update
            sudo apt-get install -y python3.10-venv
        fi
        python3.10 -m venv .venv3.10
        echo "Virtual environment .venv3.10 (python3.10) created."
    fi
    # Activate python3.10 environment and install requirements
    source .venv3.10/bin/activate
    echo "Virtual environment .venv3.10 activated."
    if [ -f "requirements.txt" ]; then
        pip install --upgrade pip
        pip install -r requirements.txt
        echo "Requirements installed."
    else
        echo "requirements.txt not found. Skipping pip install."
    fi
    echo "Environment setup complete."
    exit 0
fi

cat <<EOM

To activate the default Python environment:
  source .venv/bin/activate

To activate the Python 3.10 environment (if available):
  source .venv3.10/bin/activate

EOM

# Install requirements
if [ -f "requirements.txt" ]; then
    pip install --upgrade pip
    pip install -r requirements.txt
    echo "Requirements installed."
else
    echo "requirements.txt not found. Skipping pip install."
fi

echo "Environment setup complete."
source .venv3.10/bin/activate