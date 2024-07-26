# Define variables
VENV_DIR = venv
REQUIREMENTS = requirements.txt
PYTHON = python
PIP = $(VENV_DIR)/bin/pip
PYTHON_ENV = $(VENV_DIR)/bin/python

# Default target
all: setup install run

# Create the virtual environment
setup: $(VENV_DIR)/bin/activate

# This rule creates the virtual environment and touches the activation file
$(VENV_DIR)/bin/activate:
	$(PYTHON) -m venv $(VENV_DIR)
	touch $(VENV_DIR)/bin/activate

# Install the required packages
install: $(VENV_DIR)/bin/activate
	$(PIP) install -r $(REQUIREMENTS)

# Run the first Python file
run_tokenizer: install
	$(PYTHON_ENV) tokenizer.py

# Run the second Python file
run_train: install
	$(PYTHON_ENV) train.py

# Run all scripts
run: run_tokenizer run_train

# Clean up the virtual environment and build artifacts
clean:
	rm -rf $(VENV_DIR)
