# Makefile for HDB Resale Price Prediction Project

.PHONY: run test setup clean process-data train-models lint format

# Run the application
run:
	streamlit run app/main.py

# Run tests
test:
	python -m unittest discover tests

# Set up the environment
setup:
	pip install -e .

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Process raw data
process-data:
	python scripts/process_data.py

# Train models
train-models:
	python scripts/train_models.py

# Run linting checks
lint:
	pylint src app scripts tests

# Format code
format:
	black src app scripts tests
