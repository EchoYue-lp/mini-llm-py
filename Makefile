PYTHON ?= python

.PHONY: help install install-dev install-mlx test check

help:
	@printf '%s\n' \
		'install      Install runtime dependencies' \
		'install-dev  Install runtime and test dependencies' \
		'install-mlx  Install Apple MLX fine-tuning dependencies' \
		'test         Run the test suite' \
		'check        Compile Python sources and run tests'

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

install-mlx:
	$(PYTHON) -m pip install -r requirements-mlx.txt

test:
	$(PYTHON) -m pytest

check:
	$(PYTHON) -m compileall -q models utils scripts labs tokenization tests finetuning inference evaluation check_cloud_env.py
	$(PYTHON) -m pytest
