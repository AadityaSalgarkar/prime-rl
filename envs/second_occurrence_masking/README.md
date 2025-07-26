# Second Occurrence Masking RLVR Environment

A comprehensive Reinforcement Learning from Verifiable Rewards (RLVR) environment that tests a model's ability to reconstruct original text from masked second occurrences of words.

## Task Description

The Second Occurrence Masking task challenges models to fill in `[MASK]` tokens with words that appeared earlier in the text. This tests contextual understanding, memory, and pattern recognition abilities.

### How It Works

1. **Text Processing**: Take a text with repeated words
2. **Masking**: Replace all occurrences of target words after the first with `[MASK]`
3. **Model Task**: Fill each `[MASK]` with the correct original word
4. **Reward**: +1 for each correctly filled mask

### Example

```
Original:  "The cat chased the cat and the dog."
Masked:    "The cat chased the [MASK] and [MASK] dog."
Target:    ["cat", "the"]
Response:  "cat the"  → Reward: 2/2 = 1.0
```

## Features

- **Intelligent Word Selection**: Focuses on content words, avoids articles/prepositions when configured
- **Configurable Difficulty**: Adjustable text length, mask count, and word filtering
- **Diverse Text Sources**: Uses TinyStories and other datasets for varied content
- **Comprehensive Testing**: Multiple test suites for different scenarios
- **Multiple Inference Options**: Cloud, local (Ollama), and mock inference support
- **macOS Development**: Complete CPU-only development workflow

## Quick Start

### Prerequisites

```bash
# Install dependencies
uv sync

# For local inference (optional)
ollama pull qwen2.5:0.5b-instruct
```

### Basic Training

```bash
# Cloud inference (requires API keys)
uv run rl \
  --trainer @ envs/second_occurrence_masking/trainer_config.toml \
  --orchestrator @ envs/second_occurrence_masking/orchestrator_config.toml \
  --inference @ envs/second_occurrence_masking/inference_config.toml

# Local inference with Ollama
cp envs/second_occurrence_masking/inference_config_ollama.toml \
   envs/second_occurrence_masking/inference_config.toml
uv run rl --trainer @ ... --orchestrator @ ... --inference @ ...

# Mock inference for development
./envs/second_occurrence_masking/mock_server.py --mode masking_aware &
cp envs/second_occurrence_masking/inference_config_mock.toml \
   envs/second_occurrence_masking/inference_config.toml
uv run rl --trainer @ ... --orchestrator @ ... --inference @ ...
```

### Testing

```bash
# Core functionality tests
./envs/second_occurrence_masking/tests/test_second_occurrence_masking.py

# Local inference tests (requires Ollama)
./envs/second_occurrence_masking/tests/test_inference_ollama.py

# Mock inference tests
./envs/second_occurrence_masking/tests/test_mock_inference.py

# Full pipeline tests
./envs/second_occurrence_masking/tests/test_full_pipeline.py

# Complete test suite
./envs/second_occurrence_masking/tests/run_all_tests.py

# Mock training simulation (macOS compatible)
./envs/second_occurrence_masking/mock_trainer.py --steps 30 --accuracy 0.8
```

## Configuration

### Environment Parameters

```toml
[environment.args]
dataset_name = "roneneldan/TinyStories"  # HuggingFace dataset
min_length = 50                          # Minimum text length (chars)
max_length = 300                         # Maximum text length (chars)
min_masks = 1                            # Minimum number of masks
max_masks = 5                            # Maximum number of masks  
content_words_only = true                # Focus on nouns, verbs, adjectives
num_examples = 1000                      # Dataset size
```

### Model Recommendations

| Model Size | Batch Size | Expected Performance | Use Case |
|------------|------------|---------------------|----------|
| 0.5B       | 8          | 40-60% accuracy     | Testing, development |
| 1.5B       | 16         | 60-75% accuracy     | Small experiments |
| 7B         | 32         | 75-85% accuracy     | Full training |
| 14B+       | 64         | 85%+ accuracy       | Production |

## Development Workflow

### 1. Core Development with Mock Inference

```bash
# Start mock server
./mock_server.py --mode masking_aware --accuracy 0.7 &

# Test environment
./tests/test_second_occurrence_masking.py

# Test mock integration
./tests/test_mock_inference.py

# Run mock training
./mock_trainer.py --steps 20 --mode fast
```

### 2. Local Testing with Ollama

```bash
# Setup Ollama
ollama pull qwen2.5:0.5b-instruct

# Test local inference
./tests/test_inference_ollama.py

# Train with local model
cp inference_config_ollama.toml inference_config.toml
uv run rl --trainer @ trainer_config.toml --orchestrator @ orchestrator_config.toml --inference @ inference_config.toml
```

### 3. Production Training

```bash
# Configure for cloud inference
# Update model names in config files to larger models

# Scale up training
# Increase batch_size, max_steps in trainer_config.toml
# Increase num_examples in orchestrator_config.toml

# Run training
uv run rl --trainer @ trainer_config.toml --orchestrator @ orchestrator_config.toml --inference @ inference_config.toml
```

## File Structure

```
envs/second_occurrence_masking/
├── README.md                           # This documentation
├── second_occurrence_loader.py         # Core environment logic
├── trainer_config.toml                 # Training configuration
├── orchestrator_config.toml            # Environment configuration
├── inference_config.toml               # Inference configuration
├── inference_config_ollama.toml        # Local Ollama template
├── inference_config_mock.toml          # Mock inference template
├── mock_inference.py                   # Mock model implementation
├── mock_server.py                      # Mock inference server
├── mock_trainer.py                     # macOS training simulator
└── tests/
    ├── test_second_occurrence_masking.py  # Core functionality tests
    ├── test_inference_ollama.py           # Local inference tests
    ├── test_mock_inference.py             # Mock inference tests
    ├── test_full_pipeline.py              # Complete pipeline tests
    └── run_all_tests.py                   # Test orchestration
```

## Examples and Use Cases

### Example 1: Simple Repetition

```
Original: "The cat chased the cat."
Masked:   "The cat chased the [MASK]."
Target:   ["cat"]
```

This tests basic repetition recognition.

### Example 2: Multiple Word Types

```
Original: "She opened the door and walked through the door."
Masked:   "She opened the door and walked through the [MASK]."
Target:   ["door"]
```

This tests object tracking across different sentence structures.

### Example 3: Function Words

```
Original: "The student studied hard. The student passed the test."
Masked:   "The student studied hard. [MASK] student passed [MASK] test."
Target:   ["The", "the"]
```

This tests tracking of both content words and function words (when enabled).

### Example 4: Complex Text

```
Original: "In the garden, the flowers bloom. The bees visit the flowers while the gardener tends the garden."
Masked:   "In the garden, the flowers bloom. The bees visit [MASK] flowers while [MASK] gardener tends [MASK] garden."
Target:   ["the", "the", "the"]
```

This tests longer context and multiple repetitions.

## Performance Analysis

### Reward Function

The reward is calculated as the fraction of correctly filled masks:

```python
reward = correct_masks / total_masks
```

- **Perfect score (1.0)**: All masks filled correctly
- **Partial score (0.5)**: Half of masks filled correctly  
- **Zero score (0.0)**: No masks filled correctly

### Success Metrics

- **Basic Success**: >50% reward (most masks filled correctly)
- **Good Performance**: >70% reward (reliable mask filling)
- **Excellent Performance**: >85% reward (near-perfect understanding)

### Common Challenges

1. **Ambiguous References**: When multiple words could fit
2. **Long Distance**: Masks far from original occurrence
3. **Function Words**: Articles and prepositions are harder to track
4. **Complex Syntax**: Nested clauses and complex sentence structures

## Troubleshooting

### Common Issues

**Environment Loading Fails**
```bash
# Check Python path and dependencies
./tests/test_second_occurrence_masking.py
```

**Ollama Connection Issues**
```bash
# Ensure Ollama is running
ollama serve

# Check model availability
ollama list

# Pull required model
ollama pull qwen2.5:0.5b-instruct
```

**Mock Server Won't Start**
```bash
# Check port availability
lsof -i :8888

# Try different port
./mock_server.py --port 8889
```

**Training Hangs**
```bash
# Check all services are running
# Verify configuration files
# Start with mock inference for debugging
```

### Debug Mode

```bash
# Run with minimal examples for debugging
python -c "
from prime_rl.environments.registry import load_environment
env = load_environment('second-occurrence-masking', {'num_examples': 5})
print(f'Loaded {len(env.dataset)} examples')
for i, ex in enumerate(env.dataset):
    print(f'{i}: {ex[\"question\"][:50]}...')
"
```

## Contributing

When extending this environment:

1. **Test Thoroughly**: Run all test suites
2. **Maintain Compatibility**: Keep existing configuration formats
3. **Document Changes**: Update README and docstrings
4. **Follow Patterns**: Use established code patterns from other environments

### Adding New Features

- **New Masking Strategies**: Extend `SecondOccurrenceMaskingLoader`
- **Different Datasets**: Modify dataset loading in `get_sample_text()`
- **Custom Reward Functions**: Update `calculate_reward()` method
- **Additional Tests**: Add to `tests/` directory

## License

This environment is part of the prime-rl framework. See the main repository for license information.

## Citation

If you use this environment in research, please cite:

```bibtex
@misc{second-occurrence-masking-2024,
  title={Second Occurrence Masking: An RLVR Environment for Contextual Understanding},
  author={Prime-RL Team},
  year={2024},
  howpublished={\url{https://github.com/PrimeIntellect-ai/prime-rl}}
}
```