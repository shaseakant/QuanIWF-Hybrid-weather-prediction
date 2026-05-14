# Setup Requirement
Python 3.9 – 3.11
py -0 (to know all python version)
py -3.11 -m venv venv
venv\Scripts\Activate.ps1
python --version

# Create a virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies INSIDE venv
pip install --upgrade pip
python -m pip install -r requirements.txt


# Verify installation
python -c "import numpy; print('NumPy OK')"
python -c "import pandas; print('Pandas OK')"
python -c "import sklearn; print('Sklearn OK')"
python -c "import pennylane as qml; print('PennyLane OK:', qml.__version__)"
python -c "import qiskit; print('Qiskit OK:', qiskit.__version__)"
python -c "import torch; print('Torch OK:', torch.__version__)"


