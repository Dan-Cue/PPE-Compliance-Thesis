#!/bin/bash
# PPE Verification System - Linux/macOS Installation Script
# This script automates the installation process

set -e  # Exit on error

echo "============================================================"
echo "PPE Verification System - Automated Installation"
echo "============================================================"
echo

# Detect OS
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=macOS;;
    *)          OS_TYPE="UNKNOWN";;
esac

echo "Detected OS: ${OS_TYPE}"
echo

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python is not installed!"
    echo
    
    if [ "${OS_TYPE}" = "Linux" ]; then
        echo "Install Python with:"
        echo "  sudo apt update"
        echo "  sudo apt install python3.10 python3.10-venv python3-pip"
    elif [ "${OS_TYPE}" = "macOS" ]; then
        echo "Install Python with:"
        echo "  brew install python@3.10"
    fi
    
    exit 1
fi

# Determine Python command
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD=python3.10
elif command -v python3 &> /dev/null; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=python
fi

echo "[1/6] Python found!"
${PYTHON_CMD} --version
echo

# Check Python version
PYTHON_VERSION=$(${PYTHON_CMD} -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
PYTHON_MAJOR=$(echo ${PYTHON_VERSION} | cut -d. -f1)
PYTHON_MINOR=$(echo ${PYTHON_VERSION} | cut -d. -f2)

if [ "${PYTHON_MAJOR}" != "3" ] || [ "${PYTHON_MINOR}" -lt 8 ]; then
    echo "ERROR: Python 3.8 or higher required!"
    echo "Current version: ${PYTHON_VERSION}"
    exit 1
fi

if [ "${PYTHON_MINOR}" -ge 9 ] && [ "${PYTHON_MINOR}" -le 11 ]; then
    echo "✓ Python version is optimal!"
else
    echo "⚠ Python version ${PYTHON_VERSION} may have issues. Recommended: 3.9-3.11"
fi
echo

# Create virtual environment
echo "[2/6] Creating virtual environment..."
if [ -d "venv" ]; then
    echo "Virtual environment already exists. Removing old one..."
    rm -rf venv
fi

${PYTHON_CMD} -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment!"
    echo
    echo "Install venv with:"
    if [ "${OS_TYPE}" = "Linux" ]; then
        echo "  sudo apt install python3-venv"
    fi
    exit 1
fi

echo "✓ Virtual environment created!"
echo

# Activate virtual environment
echo "[3/6] Activating virtual environment..."
source venv/bin/activate
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to activate virtual environment!"
    exit 1
fi
echo "✓ Virtual environment activated!"
echo

# Upgrade pip
echo "[4/6] Upgrading pip..."
python -m pip install --upgrade pip
echo

# Install dependencies
echo "[5/6] Installing dependencies..."
echo "This may take 5-15 minutes depending on your internet speed..."
echo

# Install dependencies one by one for better error messages
echo "Installing OpenCV..."
pip install opencv-python>=4.8.0

echo "Installing MediaPipe..."
pip install mediapipe>=0.10.0

echo "Installing Ultralytics..."
pip install ultralytics>=8.0.0

echo "Installing TensorFlow..."
pip install tensorflow>=2.13.0

echo "Installing NumPy..."
pip install numpy>=1.24.0

if [ $? -eq 0 ]; then
    echo "✓ Dependencies installed successfully!"
else
    echo "ERROR: Some dependencies failed to install!"
    echo
    echo "Try installing manually:"
    echo "  pip install -r requirements.txt"
    exit 1
fi
echo

# Download MediaPipe models
echo "[6/6] Downloading MediaPipe models..."
python download_mediapipe_models.py
if [ $? -ne 0 ]; then
    echo "⚠ WARNING: Failed to download MediaPipe models automatically"
    echo "You'll need to download them manually."
else
    echo "✓ MediaPipe models downloaded successfully!"
fi
echo

# Verify installation
echo "============================================================"
echo "Running installation verification..."
echo "============================================================"
echo
python verify_installation.py

echo
echo "============================================================"
echo "Installation Complete!"
echo "============================================================"
echo
echo "NEXT STEPS:"
echo "1. Add your trained models to the models/ folder:"
echo "   - models/best.pt (YOLO model)"
echo "   - models/ppe_classifier.h5 (TensorFlow classifier)"
echo
echo "2. Edit config.py to match your setup"
echo
echo "3. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo
echo "4. Run the system:"
echo "   python main_ppe_system_v2.py"
echo
echo "============================================================"
