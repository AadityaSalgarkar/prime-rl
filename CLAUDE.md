# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

PRIME-RL is a decentralized reinforcement learning training framework designed for large language models. The system uses a multi-component architecture with separate trainer, orchestrator, and inference processes that communicate asynchronously to enable scalable RL training.

## Core Architecture

The system consists of three main components that work together:

### 1. Trainer (`src/prime_rl/trainer/`)
- Handles model training using FSDP (Fully Sharded Data Parallel)
- Implements GRPO (Group Relative Policy Optimization) loss
- Saves model weights to shared filesystem for orchestrator pickup
- Configuration: `src/prime_rl/trainer/config.py:157`

### 2. Orchestrator (`src/prime_rl/orchestrator/`) 
- Manages rollout generation and training data preparation
- Coordinates between inference engines and trainer
- Handles advantage computation and reward processing
- Configuration: `src/prime_rl/orchestrator/config.py:256`

### 3. Inference Server (`src/prime_rl/inference/`)
- Provides vLLM-based inference via OpenAI-compatible API
- Supports tensor and data parallelism
- Configuration: `src/prime_rl/inference/config.py:73`

### 4. RL Coordinator (`src/prime_rl/rl.py`)
- Main entry point that orchestrates all components
- Manages process lifecycle and GPU allocation
- Handles configuration validation and shared resource setup

## Development Commands

### Environment Setup
```bash
# Quick installation (recommended)
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install.sh | bash

# Manual installation
uv sync && uv sync --all-extras

# Validate environment
uv run python -c "import flash_attn"  # Check flash attention
uv run trainer @ configs/trainer/debug.toml  # Test trainer (1 GPU)
```

### Running Experiments

**Single Node RL Training:**
```bash
# Simple reverse text task (2 GPUs, ~5 minutes)
uv run rl \
  --trainer @ configs/trainer/reverse_text.toml \
  --orchestrator @ configs/orchestrator/reverse_text.toml \
  --inference @ configs/inference/reverse_text.toml

# Math training (8 GPUs: 2 for trainer, 6 for inference)
uv run rl \
  --trainer @ configs/trainer/hendrycks_math/1b.toml \
  --orchestrator @ configs/orchestrator/hendrycks_math/1b.toml \
  --inference @ configs/inference/hendrycks_math/1b.toml \
  --trainer-gpus 2 --inference-gpus 6
```

**Individual Components:**
```bash
uv run trainer @ configs/trainer/debug.toml
uv run orchestrator @ configs/orchestrator/debug.toml  
uv run inference @ configs/inference/debug.toml
uv run eval --model.name Qwen/Qwen3-0.6B --benchmarks math500,aime24
```

### Development Workflow

**TMux Setup for Development:**
```bash
# Start tmux layout with separate panes
bash scripts/tmux.sh

# In Inference pane:
uv run inference @ configs/inference/reverse_text.toml

# In Trainer pane:
uv run rl \
  --trainer @ configs/trainer/reverse_text.toml \
  --orchestrator @ configs/orchestrator/reverse_text.toml

# Kill session when done
bash scripts/tmux.sh kill
```

### Testing
```bash
# Run all tests
uv run pytest -v

# Unit tests only
uv run pytest tests/unit -v

# Integration tests only  
uv run pytest tests/integration -v

# CPU-only tests (no GPU required)
uv run pytest -v -m "not gpu"

# Fast tests only (exclude slow tests)
uv run pytest -v -m "not slow"
```

### Linting and Code Quality
```bash
# Format code
uv run ruff format .

# Lint code  
uv run ruff check .

# Pre-commit hooks
uv run pre-commit install
uv run pre-commit run --all-files
```

### Benchmarking
```bash
# Benchmark full RL pipeline
uv run rl \
  --trainer @ configs/trainer/reverse_text.toml \
  --orchestrator @ configs/orchestrator/reverse_text.toml \
  --inference @ configs/inference/reverse_text.toml \
  --bench

# Benchmark trainer only
uv run trainer @ configs/trainer/reverse_text.toml --bench --data.fake "{'micro_batch_size': 8, 'batch_size': 128, 'seq_len': 128}"

# Benchmark orchestrator (requires running inference server)
uv run orchestrator @ configs/orchestrator/reverse_text.toml --bench
```

## Configuration System

The system uses a hierarchical configuration system with the following precedence (highest to lowest):

1. **Command-line arguments**: `--model.name Qwen/Qwen3-8B`
2. **Config files**: `@ configs/trainer/debug.toml`  
3. **Environment variables**: `PRIME_MODEL__NAME=Qwen/Qwen3-8B`
4. **Defaults**: Built-in default values

**Config File Usage:**
```bash
# Single config file
uv run trainer @ configs/trainer/debug.toml

# Multiple config files (later ones override earlier)
uv run trainer @ configs/base.toml @ configs/trainer/math.toml

# Override specific values
uv run trainer @ configs/trainer/debug.toml --model.name different-model --max_steps 100
```

## Key Implementation Details

### Asynchronous Training
- The `async_level` parameter controls how many steps inference can be ahead of training
- `async_level=0`: Fully synchronous (on-policy)
- `async_level>0`: Allows off-policy training with better throughput

### Checkpointing
- **Trainer checkpoints**: Full model + optimizer state in `checkpoints/step_{step}/trainer.pt`
- **Weight checkpoints**: Model weights only in `weights/step_{step}` (for inference updates)
- **Orchestrator checkpoints**: Progress state in `checkpoints/step_{step}/orchestrator.pt`

**Resuming from checkpoints:**
```bash
# Resume both trainer and orchestrator from step 50
uv run rl \
  --trainer @ configs/trainer/reverse_text.toml \
  --trainer.ckpt.resume-step 50 \
  --orchestrator @ configs/orchestrator/reverse_text.toml \
  --orchestrator.ckpt.resume-step 50
```

### GPU Memory and Performance
- Uses FSDP for trainer memory efficiency
- Flash Attention 2 for inference performance
- Gradient checkpointing available via `model.ac=true`
- Model compilation via `model.compile=true` (experimental)

### Logging and Monitoring
- File-based logging for each component in `logs/`
- W&B integration for experiment tracking
- Rich console output with progress tracking

## Common Development Patterns

### Adding New Environments
1. Register environment in `src/prime_rl/environments/registry.py`
2. Create config files in `configs/{trainer,orchestrator,inference}/your_env.toml`
3. Update environment args in orchestrator config

### Model Integration
1. Ensure model is compatible with transformers and vLLM
2. Update model name in config files
3. Adjust `max_model_len` and `seq_len` as needed
4. Test with debug configs first

### Multi-Node Setup
- Configure distributed training in trainer config
- Use `CUDA_VISIBLE_DEVICES` for GPU assignment
- Set unique ports for multiple experiments: `--inference.server.port 8001`

## Troubleshooting

### Common Issues
- **"Too many open files"**: Run `ulimit -n 32000`
- **vLLM startup fails**: Check GPU memory and model size compatibility
- **Training hangs**: Verify rollout directory permissions and async_level settings
- **W&B login required**: Run `uv run wandb login` or set `WANDB_API_KEY`

### Debug Mode
Use debug configs for quick testing:
```bash
uv run trainer @ configs/trainer/debug.toml  # 5 steps, fake data
uv run orchestrator @ configs/orchestrator/debug.toml
uv run inference @ configs/inference/debug.toml
```

## Development Best Practices

### Testing
- When running tests, create a standalone script
- For tests, use a standalone script and run them with ./script.py

## Example: Complete Environment Implementation

### Case Study: Thesaurus Replacement Environment

This section documents the complete process of implementing a new RLVR environment from scratch, including testing and Ollama integration.

#### Overview
The thesaurus replacement environment tests a model's ability to reconstruct original text from synonym-corrupted versions. This serves as a comprehensive example of environment development in the prime-rl framework.

#### Step-by-Step Implementation

**1. Environment Design and Planning**
```bash
# Branch created: env/thesaurus_replacement
# Task: Implement RLVR environment for text reconstruction from synonym-corrupted input
```

**2. Core Implementation**
- **ThesaurusLoader** (`envs/thesaurus_replacement/thesaurus_loader.py`):
  - Loads 23.2MB WordNet thesaurus data (17,749 words)
  - Implements synonym replacement with case preservation
  - Configurable replacement rate (default: 30%)

- **Environment Registration** (`src/prime_rl/environments/registry.py`):
  ```python
  "thesaurus-replacement": {
      "load_fn": load_thesaurus_replacement_environment,
      "type": "train",
      "tags": ["instruction-following", "text-reconstruction"],
  }
  ```

- **Reward Function**: Word-level exact matching with length penalties

**3. Configuration Files**
```toml
# trainer_config.toml
max_steps = 30
[model]
name = "Qwen/Qwen2.5-0.5B-Instruct"

# orchestrator_config.toml  
[environment]
id = "thesaurus-replacement"
[environment.args]
replacement_rate = 0.3
num_examples = 1000

# inference_config.toml
[model] 
name = "Qwen/Qwen2.5-0.5B-Instruct"
```

**4. Testing Infrastructure**
Created comprehensive test suite using uv script format:

- **Core Tests** (`tests/test_thesaurus.py`):
  ```bash
  #!/usr/bin/env -S uv run --script
  # /// script
  # dependencies = []
  # requires-python = ">=3.8"
  # ///
  ```
  - Data format validation
  - ThesaurusLoader functionality
  - Reward function logic
  - Environment integration

- **Ollama Integration Tests** (`tests/test_inference_ollama.py`):
  ```bash
  # /// script
  # dependencies = ["requests", "openai"]
  # requires-python = ">=3.8"
  # ///
  ```
  - Ollama API connection testing
  - Model availability detection
  - Real inference testing with qwen2.5:0.5b-instruct
  - Configuration compatibility validation

- **Combined Test Runner** (`tests/run_all_tests.py`):
  - Orchestrates all test suites
  - Provides comprehensive reporting

**5. Ollama Local Inference Support**
- **Configuration Template** (`inference_config_ollama.toml`):
  ```toml
  [model]
  name = "qwen2.5:0.5b-instruct"  # Ollama format
  base_url = "http://localhost:11434/v1"
  api_key = "ollama"
  ```

- **Setup Instructions**:
  ```bash
  # Install Ollama model matching training config
  ollama pull qwen2.5:0.5b-instruct
  
  # Copy template and run tests
  cp inference_config_ollama.toml inference_config.toml
  ./tests/test_inference_ollama.py
  ```

**6. Mock Inference for Training Pipeline Testing**
- **Mock Implementation** (`mock_inference.py`):
  - Identity model: input = output (perfect for testing)
  - Simple completions for general testing
  - Thesaurus-aware mode with heuristic restoration
  - OpenAI-compatible API interface

- **Mock Server** (`mock_server.py`):
  ```bash
  # Start mock inference server
  ./mock_server.py
  
  # Available at http://localhost:8888/v1/chat/completions
  ```

- **Configuration Template** (`inference_config_mock.toml`):
  ```toml
  [model]
  name = "mock-identity"
  base_url = "http://localhost:8888/v1"
  api_key = "mock"
  
  [mock]
  mode = "identity"  # Options: identity, simple_completion, thesaurus_aware
  ```

- **Testing Pipeline**:
  ```bash
  # Test mock system
  ./tests/test_mock_inference.py
  
  # Start mock server for training
  ./mock_server.py &
  
  # Use mock config for training tests
  cp inference_config_mock.toml inference_config.toml
  
  # Run training with mock inference (no GPU required)
  uv run rl --trainer @ ... --inference @ inference_config.toml
  ```

**7. Documentation and Examples**
- **Comprehensive README** with:
  - Task description and examples
  - Setup instructions with data download
  - Testing procedures for both local and cloud inference
  - Performance expectations and model recommendations

**8. Validation and Testing Results**
```bash
# Final test results:
📊 Core Environment Tests: 4/4 PASSED
📊 Ollama Integration Tests: 5/5 PASSED  
📊 Mock Inference Tests: 4/4 PASSED
📊 Overall: 3/3 test suites PASSED

# Performance metrics:
• 17,749 words loaded from thesaurus
• 67% word accuracy with Ollama inference (real models)
• 80% word accuracy with mock inference (testing)
• Complete training pipeline tested without GPU requirements
• Both cloud, local, and mock inference workflows validated
```

#### Key Development Patterns Demonstrated

**Environment Structure**:
```
envs/thesaurus_replacement/
├── README.md                    # Complete documentation
├── thesaurus_loader.py          # Core functionality
├── en_thesaurus.jsonl          # Data file (23.2MB)
├── trainer_config.toml          # Training configuration
├── orchestrator_config.toml     # Environment configuration  
├── inference_config.toml        # Inference configuration
├── inference_config_ollama.toml # Local inference template
├── inference_config_mock.toml   # Mock inference template
├── mock_inference.py            # Mock model implementation
├── mock_server.py              # Mock inference server
└── tests/
    ├── test_thesaurus.py        # Core functionality tests
    ├── test_inference_ollama.py # Local inference tests
    ├── test_mock_inference.py   # Mock inference tests
    └── run_all_tests.py         # Test orchestration
```

**Git Workflow**:
```bash
# Two main commits on env/thesaurus_replacement branch:
git commit -m "Add comprehensive testing suite for thesaurus replacement environment"
git commit -m "Add Ollama integration testing and reorganize test structure"
```

**Testing Best Practices**:
- Use uv script format for standalone executable tests
- Organize tests in dedicated tests/ directory
- Include both unit tests and integration tests
- Support both cloud and local inference testing
- Provide clear setup instructions and model recommendations

#### Usage Examples

**Basic Training**:
```bash
uv run rl \
  --trainer @ envs/thesaurus_replacement/trainer_config.toml \
  --orchestrator @ envs/thesaurus_replacement/orchestrator_config.toml \
  --inference @ envs/thesaurus_replacement/inference_config.toml
```

**Testing**:
```bash
# Core functionality
./tests/test_thesaurus.py

# Ollama integration  
./tests/test_inference_ollama.py

# Mock inference (training pipeline)
./tests/test_mock_inference.py

# Complete test suite
./tests/run_all_tests.py
```

**Local Inference with Ollama**:
```bash
# Setup
ollama pull qwen2.5:0.5b-instruct
cp inference_config_ollama.toml inference_config.toml

# Run with local model
uv run rl --trainer @ ... --inference @ inference_config.toml
```

**Mock Inference for Testing**:
```bash
# Setup mock inference (no GPU required)
./mock_server.py &
cp inference_config_mock.toml inference_config.toml

# Run training with mock model
uv run rl --trainer @ ... --inference @ inference_config.toml

# Test training pipeline without external dependencies
./tests/test_mock_inference.py
```

This implementation demonstrates the complete lifecycle of environment development in prime-rl, from initial design through testing and production deployment with cloud, local, and mock inference support.

</invoke>