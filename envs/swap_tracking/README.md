# Swap Tracking Box Prediction Environment

A complete RLVR environment for training models to track box positions through sequences of swaps and predict final arrangements.

## Overview

The swap tracking environment tests a model's ability to:
- Track multiple objects (boxes numbered 1-10) through a sequence of position swaps
- Maintain state through complex transformations
- Predict final arrangements accurately

### Task Description

1. **Initial State**: Boxes arranged in order [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
2. **Transformations**: 20 random swaps between positions (described in words)
3. **Challenge**: Predict the final arrangement after all swaps
4. **Reward**: Position-wise accuracy (0.0 to 1.0)

### Example Task

**Input**: "Boxes are arranged from 1 to n=10. Then the box at location two is swapped with the box at location seven. Then the box at location one is swapped with the box at location ten..."

**Expected Output**: "[10, 7, 3, 4, 5, 6, 2, 8, 9, 1]"

**Reward**: Fraction of positions correctly predicted

## Environment Structure

```
envs/swap_tracking/
├── README.md                           # This documentation
├── swap_tracking_loader.py             # Core environment implementation
├── trainer_config.toml                 # Training configuration
├── orchestrator_config.toml            # Environment configuration
├── inference_config.toml               # Inference configuration
├── inference_config_ollama.toml        # Local inference template
├── inference_config_mock.toml          # Mock inference template
├── mock_inference.py                   # Mock model implementation
├── mock_server.py                      # Mock inference server
├── mock_trainer.py                     # Complete training simulator
└── tests/
    ├── test_swap_tracking.py           # Core functionality tests
    ├── test_inference_ollama.py        # Local inference tests
    ├── test_mock_inference.py          # Mock inference tests
    ├── test_full_pipeline.py           # Complete pipeline tests
    └── run_all_tests.py                # Test orchestration
```

## Quick Start

### 1. Validate Environment

```bash
cd envs/swap_tracking
./tests/run_all_tests.py --required-only
```

Expected output: "🎉 ALL REQUIRED TESTS PASSED!"

### 2. Test Mock Training

```bash
./mock_trainer.py --steps 20 --batch-size 8
```

This runs a complete training simulation without GPU requirements.

### 3. Run Real Training

```bash
# Copy desired inference config
cp inference_config_mock.toml inference_config.toml

# Run training
uv run rl \\
  --trainer @ trainer_config.toml \\
  --orchestrator @ orchestrator_config.toml \\
  --inference @ inference_config.toml
```

## Configuration Options

### Training Parameters

- `max_steps`: Number of training steps (default: 30)
- `batch_size`: Training batch size (default: 8)
- `learning_rate`: Learning rate (default: 3e-6)
- `model.name`: Model to train (default: "Qwen/Qwen2.5-0.5B-Instruct")

### Environment Parameters

- `n_boxes`: Number of boxes (default: 10)
- `n_swaps`: Number of swaps per task (default: 20)
- `num_examples`: Training examples to generate (default: 1000)

### Inference Options

1. **Cloud Inference**: Use default `inference_config.toml`
2. **Local Ollama**: Copy `inference_config_ollama.toml`
3. **Mock Inference**: Copy `inference_config_mock.toml`

## Testing

### Core Tests
```bash
./tests/test_swap_tracking.py
```
Tests data generation, reward calculation, and basic environment logic.

### Ollama Integration
```bash
./tests/test_inference_ollama.py
```
Tests local inference with Ollama models (requires Ollama installation).

### Mock Inference
```bash
./tests/test_mock_inference.py
```
Tests mock inference system for training pipeline development.

### Full Pipeline
```bash
./tests/test_full_pipeline.py
```
Tests complete training workflow and prime-rl compatibility.

### Complete Suite
```bash
./tests/run_all_tests.py
```
Runs all tests with comprehensive reporting.

## Mock Training System

The environment includes a complete mock training system for development:

### Mock Inference Server
```bash
./mock_server.py
```
Provides OpenAI-compatible API with multiple mock model modes:
- `mock-identity`: Returns original order (perfect for testing)
- `mock-simple`: Basic completion responses
- `mock-swap-aware`: Attempts swap tracking with errors
- `mock-random`: Random arrangements

### Mock Trainer
```bash
./mock_trainer.py --steps 30 --batch-size 8
```
Simulates complete RL training with:
- Rich console visualization
- Progress tracking
- Realistic metrics
- Checkpoint simulation
- macOS compatibility (CPU-only)

## Performance Expectations

### Data Generation
- 10 examples: ~40,000/sec
- 1,000 examples: ~50,000/sec
- Scales linearly with number of examples

### Model Performance
- **Random baseline**: ~10% accuracy
- **Simple heuristics**: ~30% accuracy
- **Good models**: 60-80% accuracy
- **Perfect tracking**: 100% accuracy

### Training Time
- Mock training: 30 steps in ~15 seconds
- Real training (GPU): 30 steps in ~2-5 minutes
- Scales with model size and batch size

## Development Workflow

### 1. Environment Development
```bash
# Create new branch
git checkout -b env/swap_tracking

# Develop and test
./tests/run_all_tests.py

# Mock training validation
./mock_trainer.py --steps 10
```

### 2. Local Testing
```bash
# Install Ollama model
ollama pull qwen2.5:0.5b-instruct

# Test local inference
cp inference_config_ollama.toml inference_config.toml
./tests/test_inference_ollama.py
```

### 3. Production Training
```bash
# Use real models for training
cp inference_config.toml inference_config_production.toml
# Edit to use production model

# Run full training
uv run rl --trainer @ trainer_config.toml --orchestrator @ orchestrator_config.toml --inference @ inference_config_production.toml
```

## Implementation Details

### Core Algorithm
The `SwapTrackingLoader` implements:
1. **Task Generation**: Creates random swap sequences with reproducible seeds
2. **Text Formatting**: Converts swaps to natural language instructions
3. **Reward Calculation**: Position-wise exact matching
4. **Data Pipeline**: Verifiers-compatible dataset format

### Mock System
The mock inference system provides:
1. **Multiple Modes**: Identity, completion, swap-aware, random
2. **OpenAI Compatibility**: Standard chat completions API
3. **Dynamic Box Count**: Adapts to task requirements
4. **Error Simulation**: Realistic model mistakes

### Testing Philosophy
1. **Progressive Testing**: Core → Integration → Full Pipeline
2. **Multiple Inference**: Cloud → Local → Mock
3. **Cross-Platform**: Supports both GPU and CPU-only development
4. **Comprehensive Coverage**: All components validated

## Troubleshooting

### Common Issues

**Tests failing**:
```bash
./tests/run_all_tests.py --verbose
```

**Mock server not starting**:
```bash
# Check port availability
lsof -i :8888
./mock_server.py --port 8889
```

**Ollama tests failing**:
```bash
# Install required model
ollama pull qwen2.5:0.5b-instruct
ollama list
```

**Training not starting**:
```bash
# Validate configs
./tests/test_full_pipeline.py
```

### Performance Issues

**Slow data generation**:
- Reduce `num_examples` in orchestrator config
- Use smaller `n_swaps` for simpler tasks

**Mock training too slow**:
- Reduce `--steps` parameter
- Use smaller `--batch-size`

**Real training fails**:
- Check GPU availability
- Validate model compatibility
- Use debug configs first

## Success Metrics

A successful environment should achieve:
- ✅ 100% test suite pass rate
- ✅ Mock training completion
- ✅ Real inference compatibility  
- ✅ Cross-platform support
- ✅ Complete documentation

This environment demonstrates the complete lifecycle from initial design through production deployment with robust testing at every level.