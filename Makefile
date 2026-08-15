PYTHON ?= python

.PHONY: help install install-dev install-mlx smoke-core test-core test-pytorch test-mlx test check

help:
	@printf '%s\n' \
		'install      Install runtime dependencies' \
		'install-dev  Install runtime and test dependencies' \
		'install-mlx  Install Apple MLX fine-tuning dependencies' \
		'smoke-core   Run every self-contained foundation and lab demo' \
		'test-core    Run tests that do not need Transformers or MLX' \
		'test-pytorch Run the complete PyTorch test suite' \
		'test-mlx     Check MLX imports and validate tool-router data' \
		'test         Run the test suite' \
		'check        Compile Python sources and run tests'

install:
	$(PYTHON) -m pip install -r requirements.txt

install-dev:
	$(PYTHON) -m pip install -r requirements-dev.txt

install-mlx:
	$(PYTHON) -m pip install -r requirements-mlx.txt

smoke-core:
	$(PYTHON) -m foundations
	$(PYTHON) -m labs

test-core:
	$(PYTHON) -m pytest \
		tests/test_foundations.py \
		tests/test_embedding_geometry.py \
		tests/test_layer_contracts.py \
		tests/test_learning_labs.py \
		tests/test_model_basics.py \
		tests/test_padding_mask.py \
		tests/test_pre_ln.py \
		tests/test_correctness_regressions.py \
		tests/test_tool_router_validation.py \
		tests/test_training_layout.py

test-pytorch:
	$(PYTHON) -m pytest

test-mlx:
	$(PYTHON) -c "import mlx, mlx_lm; print('MLX ready')"
	$(PYTHON) -m evaluation.validate_tool_router_data

test:
	$(PYTHON) -m pytest

check:
	$(PYTHON) -m compileall -q models utils scripts foundations labs tokenization tests finetuning inference evaluation check_cloud_env.py
	$(PYTHON) -m pytest
