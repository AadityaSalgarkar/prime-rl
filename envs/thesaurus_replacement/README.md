# Thesaurus Replacement Environment

## Overview

The Thesaurus Replacement environment is designed for Reinforcement Learning from Verifiable Rewards (RLVR) training. This environment tests a language model's ability to reconstruct original text from synonym-corrupted versions.

## Task Description

### Objective
Given a sentence where X% of words have been randomly replaced with near-synonyms, the model must reconstruct the exact original sentence.

### Environment Mechanics

1. **State**: A sentence with synonyms substituted for original words
2. **Action**: Generate the exact original sentence with correct words restored
3. **Reward**: +1 for each word that exactly matches the original (word-level exact matching)

### Example

- **Original**: "She opened the ancient door."
- **Augmented**: "She unfastened the antique door."
- **Expected Output**: "She opened the ancient door."
- **Reward**: 2/2 = 1.0 (all words match exactly)

## Setup Instructions

### Prerequisites
Before running the environment, you need to download the thesaurus data:

```bash
# Navigate to the thesaurus replacement environment directory
cd envs/thesaurus_replacement/

# Download the thesaurus data (WordNet-based synonyms)
curl -O https://raw.githubusercontent.com/zaibacu/thesaurus/master/en_thesaurus.jsonl
```

**Note**: The `en_thesaurus.jsonl` file (~23MB) contains WordNet synonym data and is not committed to the repository due to its size.

## Implementation Details

### Data Augmentation Strategy
- Uses the repo github.com/zaibacu/thesaurus to find near-synonyms
- Among the words with available synonyms, randomly selects 30% of words in each sentence for replacement
- Maintains grammatical structure and sentence length where possible
- Uses TinyStories dataset for diverse, natural sentence examples

### Reward Function
- **Word-level exact matching**: Each word position is compared exactly
- **Case-sensitive matching**: "Door" ≠ "door"
- **Normalization**: Basic whitespace normalization applied
- **Scoring**: Fraction of correctly restored words (0.0 to 1.0)

### Dataset Properties
- Built on common English sentences and passages
- Diverse vocabulary to test synonym understanding
- Varying sentence lengths and complexity levels
- Balanced distribution of word types (nouns, verbs, adjectives, etc.)

## Configuration

### Environment Parameters
- `replacement_rate`: Percentage of words to replace with synonyms (default: 0.3)
- `min_synonyms`: Minimum number of synonyms required for replacement (default: 2)
- `preserve_case`: Whether to preserve original capitalization (default: True)

### Model Requirements
- Input format: Corrupted sentence as user prompt
- Output format: Reconstructed original sentence
- No special formatting or XML tags required
- Direct text-to-text reconstruction task

## Integration

This environment integrates with the prime-rl framework through:
- Registration in `src/prime_rl/environments/registry.py`
- Compatible with verifiers library interface
- Uses `SingleTurnEnv` for single-turn question-answer format
- Supports batched evaluation for efficient training

## Expected Performance

### Success Metrics
- **Perfect Reconstruction**: 100% word-level accuracy
- **Partial Success**: 80-99% word-level accuracy  
- **Baseline Performance**: Random word selection ~20% accuracy

### Training Characteristics
- Requires understanding of semantic similarity
- Tests vocabulary knowledge and context awareness
- Benefits from large-scale language model pre-training
- Relatively fast training due to deterministic rewards

# Testing

## Running Tests

The environment includes a comprehensive test suite that validates all components:

```bash
# Run the standalone test script
./tests/test_thesaurus.py

# Or using uv directly
uv run --script tests/test_thesaurus.py
```

The test suite validates:
- ✅ **Data Format**: Thesaurus data file structure and content
- ✅ **ThesaurusLoader**: Synonym lookup and text replacement functionality  
- ✅ **Reward Function**: Word-level accuracy scoring logic
- ✅ **Environment Integration**: Configuration files and registry setup

## Testing with Ollama (Local LLM)

For testing with local Ollama models:

```bash
# Test Ollama integration
./tests/test_inference_ollama.py
```

**Prerequisites for Ollama testing:**
1. Install Ollama: https://ollama.ai
2. Start Ollama server: `ollama serve`
3. Install the matching model: `ollama pull qwen2.5:0.5b-instruct`

The Ollama test suite validates:
- 🔗 **Ollama Connection**: API availability and model detection
- 🧠 **Basic Inference**: Simple completion testing
- 🎯 **Environment Integration**: Thesaurus task with Ollama models
- ⚙️ **Config Compatibility**: Configuration file validation
- 📊 **Model Recommendations**: Best models for text reconstruction

**Using Ollama Configuration:**
```bash
# Copy the Ollama config template
cp inference_config_ollama.toml inference_config.toml

# Edit the model name to match your installed Ollama model
# [model]
# name = "qwen2.5:0.5b-instruct"  # matches training config
```

## Running All Tests

To run both core functionality and Ollama integration tests:

```bash
# Run complete test suite
./tests/run_all_tests.py
```

This will run both test suites and provide a comprehensive validation report.

## Testing with Mock Inference (Training Pipeline)

For testing the complete training pipeline without requiring actual models:

```bash
# Test mock inference system
./tests/test_mock_inference.py

# Start mock server for training tests
./mock_server.py &

# Use mock configuration for training
cp inference_config_mock.toml inference_config.toml
```

**Mock Inference Features:**
- 🔄 **Identity Model**: Returns input as output (perfect for testing)
- 📝 **Simple Completions**: Basic sentence completion for testing
- 🎯 **Thesaurus-Aware**: Attempts basic synonym restoration
- 🚀 **Fast**: No GPU required, instant responses
- 🔌 **OpenAI Compatible**: Drop-in replacement for real inference

**Mock Testing Modes:**
```python
# Available mock modes:
"identity"           # Input = Output (perfect accuracy testing)
"simple_completion"  # Basic completions for general testing  
"thesaurus_aware"    # Heuristic-based synonym restoration
```

This enables end-to-end testing of the training pipeline without requiring external models or GPU resources.

## Complete Pipeline Testing (macOS Compatible)

For testing the full trainer + orchestrator + inference workflow on macOS:

```bash
# Test individual mock trainer (simulates full RL training)
./mock_trainer.py --steps 20 --batch-size 8

# Test full pipeline integration
./tests/test_full_pipeline.py

# Test all components together
./tests/run_all_tests.py
```

**Mock Training Features:**
- 🔄 **Complete RL Simulation**: Orchestrator + Trainer + Inference in one script
- 💻 **macOS Compatible**: Works without CUDA or GPU requirements
- 📊 **Rich Console Output**: Beautiful training progress display
- 💾 **Checkpoint Simulation**: Saves training state like real trainer
- 🎯 **Realistic Metrics**: Loss, reward, gradient norms, timing

**Pipeline Components Tested:**
1. **Environment**: Thesaurus data loading and synonym replacement
2. **Orchestrator**: Batch generation and reward calculation  
3. **Inference**: Mock model responses via OpenAI-compatible API
4. **Trainer**: Simulated training loop with metrics and checkpointing

Example output:
```
🚀 Starting Mock Training for Thesaurus Replacement Environment
       Training Step 5        
┏━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┓
┃ Metric            ┃ Value  ┃
┡━━━━━━━━━━━━━━━━━━━╇━━━━━━━━┩
│ Loss              │ 0.7623 │
│ Reward            │ 0.2448 │
│ Cumulative Reward │ 0.2116 │
│ Grad Norm         │ 1.2479 │
│ Elapsed Time      │ 0.5s   │
└───────────────────┴────────┘
```

### Test Output Example
```
🚀 Testing Thesaurus Replacement Environment
==================================================
🧪 Testing thesaurus data format...
✅ Data file exists: en_thesaurus.jsonl
✅ File size: 23.2 MB
✅ Valid entries in first 1000: 1000/1000 (100.0%)

📊 TEST SUMMARY: 4/4 tests passed
🎉 All tests passed! Environment is ready for use.
```

# Instructions to Run

The environment can be run using:
```bash
uv run rl \
  --trainer @ envs/thesaurus_replacement/trainer_config.toml \
  --orchestrator @ envs/thesaurus_replacement/orchestrator_config.toml \
  --inference @ envs/thesaurus_replacement/inference_config.toml
```

## Registry Implementation

The environment is implemented in `src/prime_rl/environments/registry.py` through the `load_thesaurus_replacement_environment()` function. This function:

### Key Features
- **Environment ID**: `"thesaurus-replacement"` (registered in the REGISTRY)
- **Environment Type**: Training environment for RLVR
- **Tags**: `["instruction-following", "text-reconstruction"]`

### Function Parameters
```python
def load_thesaurus_replacement_environment(
    replacement_rate: float = 0.3,        # Fraction of words to replace (30%)
    min_synonyms: int = 2,                # Minimum synonyms required (filtering)
    preserve_case: bool = True,           # Preserve original capitalization
    num_examples: int = 1000,             # Number of training examples
    **kwargs
) -> Environment
```

### Implementation Details
- Loads thesaurus data from `envs/thesaurus_replacement/en_thesaurus.jsonl`
- Extracts sentences from TinyStories dataset (5-15 words, containing replaceable words)
- Creates training examples with original/augmented text pairs
- Uses `vf.SingleTurnEnv` with custom word-level accuracy reward function
- Reward function computes exact word matching with length penalty
- Returns a fully configured RLVR environment ready for training
