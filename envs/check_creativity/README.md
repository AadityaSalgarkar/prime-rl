# Creativity Environment for RLVR Training

A comprehensive RLVR (Reinforcement Learning from Verifiable Rewards) environment for training models on creative text generation tasks. This environment evaluates and rewards text creativity using multiple linguistic diversity metrics.

## Overview

The creativity environment provides:

- **Multi-dimensional creativity evaluation** using entropy, word diversity, uncommon words, sentence variety, and punctuation diversity
- **Comprehensive data generation** with diverse creative writing prompts across multiple categories
- **Complete RLVR integration** with the verifiers framework and prime-rl training pipeline
- **Advanced evaluation rubrics** with detailed analysis and progress tracking
- **Flexible configuration** for different training scenarios and creativity aspects

## Features

### Core Components

1. **Reward Function** (`reward.py`): Multi-metric creativity scoring
2. **Environment Loader** (`creativity_loader.py`): Verifiers framework integration
3. **Comprehensive Environment** (`creativity_env.py`): Full RL environment with state management
4. **Data Generation** (`data_utils.py`): Advanced prompt and text generation
5. **Evaluation System** (`evaluation_rubrics.py`): Detailed analysis and benchmarking
6. **Utilities** (`utils.py`): Configuration validation and helper functions

### Creativity Metrics

The reward function evaluates text across seven dimensions:

- **Entropy (w_entropy=1.0)**: Word distribution diversity
- **Distinct Ratio (w_distinct=1.0)**: Unique words vs. total words
- **Uncommon Words (w_uncommon=0.8)**: Usage of rare vocabulary
- **Bigram Diversity (w_bigrams=1.2)**: Variety in word pair combinations
- **Sentence Length Variance (w_sentence_len_var=0.6)**: Varied sentence structures
- **Word Length Variance (w_word_len_var=0.4)**: Mixed word lengths
- **Sentence Ending Variety (w_sentence_end_var=0.5)**: Diverse punctuation usage

### Prompt Categories

The environment generates prompts across five creative categories:

- **Storytelling**: Character-driven narratives and plot development
- **Poetry**: Rhythmic and metaphorical language
- **Descriptive**: Vivid imagery and sensory details
- **Philosophical**: Abstract concepts and deep reflection
- **Experimental**: Innovative forms and techniques

## Quick Start

### 1. Environment Setup

The creativity environment is already integrated into the prime-rl registry. To use it:

```python
from prime_rl.environments.registry import load_environment

# Load the creativity environment
env = load_environment('creativity', {
    'num_train_samples': 1000,
    'num_eval_samples': 100,
    'reward_weights': {
        'w_entropy': 1.0,
        'w_distinct': 1.0,
        'w_uncommon': 0.8,
        'w_bigrams': 1.2,
        'w_sentence_len_var': 0.6,
        'w_word_len_var': 0.4,
        'w_sentence_end_var': 0.5,
    }
})
```

### 2. Training Configuration

Use the provided configuration files for training:

```bash
# Trainer configuration
trainer_config.toml    # Model training parameters

# Orchestrator configuration  
orchestrator_config.toml    # Environment coordination settings

# Inference configuration
inference_config.toml    # Model inference parameters
```

### 3. Running Tests

Validate the environment setup:

```bash
# Run comprehensive functionality tests
python test_creativity_environment.py

# Run integration tests with prime-rl components
python test_integration.py
```

## Configuration

### Trainer Configuration (`trainer_config.toml`)

```toml
max_steps = 50
[model]
model_name = "Qwen/Qwen2.5-0.5B-Instruct"

[data.creativity]
num_train_samples = 1000
num_eval_samples = 100
reward_weights = { w_entropy = 1.0, w_distinct = 1.0, w_uncommon = 0.8, w_bigrams = 1.2, w_sentence_len_var = 0.6, w_word_len_var = 0.4, w_sentence_end_var = 0.5 }
```

### Orchestrator Configuration (`orchestrator_config.toml`)

```toml
[environment]
env_id = "creativity"

[environment.creativity]
num_train_samples = 1000
reward_weights = { w_entropy = 1.0, w_distinct = 1.0, w_uncommon = 0.8, w_bigrams = 1.2, w_sentence_len_var = 0.6, w_word_len_var = 0.4, w_sentence_end_var = 0.5 }

[sampling]
temperature = 0.8
top_p = 0.9
```

## Usage Examples

### Basic Reward Calculation

```python
from reward import reward_function

text = """The iridescent butterflies danced through gossamer clouds, 
          their wings painting symphonies of color across the canvas 
          of imagination."""

score = reward_function(text)
print(f"Creativity score: {score:.2f}")
```

### Advanced Environment Usage

```python
from creativity_env import CreativityEnvironment

# Initialize environment
env = CreativityEnvironment()

# Reset for new episode
obs = env.reset()

# Process model response
response = "The ethereal moonbeams whispered ancient secrets..."
action = CreativityAction(text=response)

# Get reward and next state
next_obs, reward, done, info = env.step(action)

print(f"Reward: {reward:.3f}")
print(f"Components: {info['reward_components']}")
```

### Evaluation and Analysis

```python
from evaluation_rubrics import CreativityRubric, CreativityTracker

# Analyze single text
rubric = CreativityRubric()
analysis = rubric.evaluate_text(text, prompt, category)

print(f"Score: {analysis.total_score:.2f}")
print(f"Grade: {rubric.grade_creativity_level(analysis.total_score)}")

# Track training progress
tracker = CreativityTracker()
tracker.add_analysis(analysis)

progress = tracker.get_progress_summary()
insights = tracker.generate_training_insights()
```

## Advanced Features

### Custom Reward Weights

Adjust reward weights to emphasize different creativity aspects:

```python
# Emphasize vocabulary diversity
weights = {
    'w_entropy': 1.5,      # Higher word distribution entropy
    'w_distinct': 1.5,     # More unique words
    'w_uncommon': 1.2,     # Increased rare word usage
    'w_bigrams': 1.0,      # Standard phrase variety
    'w_sentence_len_var': 0.5,
    'w_word_len_var': 0.3,
    'w_sentence_end_var': 0.4,
}

env = load_environment('creativity', {'reward_weights': weights})
```

### Data Augmentation

Generate varied prompts for training:

```python
from data_utils import DataAugmentation, CreativeTextGenerator

# Create prompt variations
base_prompts = ["Write a story", "Describe a scene"]
augmented = DataAugmentation.augment_prompt_variations(base_prompts)

# Generate creative samples
generator = CreativeTextGenerator()
creative_text = generator.generate_creative_sample(
    prompt="Write about time",
    length="medium",
    technique="synesthesia"
)
```

### Benchmarking

Compare against reference creativity levels:

```python
from evaluation_rubrics import CreativityBenchmark

benchmark = CreativityBenchmark()
result = benchmark.benchmark_text(text)

print(f"Benchmark level: {result['benchmark_level']}")
print(f"Assessment: {result['overall_assessment']}")
```

## Training Pipeline Integration

### 1. With Trainer

```bash
# Using trainer config
trainer --config envs/check_creativity/trainer_config.toml
```

### 2. With Orchestrator

```bash
# Using orchestrator config  
orchestrator --config envs/check_creativity/orchestrator_config.toml
```

### 3. With Inference Server

```bash
# Using inference config
inference --config envs/check_creativity/inference_config.toml
```

## Performance Optimization

### Batch Processing

For efficient evaluation of multiple texts:

```python
env = CreativityEnvironment()
texts = ["Text 1", "Text 2", "Text 3"]
scores = env.batch_process(texts)
```

### Metrics Tracking

Monitor training progress:

```python
# Enable detailed metrics
tracker = CreativityTracker(save_dir=Path("metrics"))

# Add analyses during training
tracker.add_analysis(analysis, iteration=step)

# Generate insights
insights = tracker.generate_training_insights()
tracker.save_analysis()
```

## Configuration Validation

Validate your configuration files:

```python
from utils import ConfigValidator

validator = ConfigValidator()
results = validator.validate_all_configs(Path("envs/check_creativity"))

if results['overall_valid']:
    print("✓ All configurations valid")
else:
    print("✗ Configuration issues found")
    for config, result in results.items():
        if not result['valid']:
            print(f"  {config}: {result['errors']}")
```

## Troubleshooting

### Common Issues

1. **NLTK Data Missing**
   ```python
   from utils import CreativityUtils
   CreativityUtils.setup_nltk_dependencies()
   ```

2. **Configuration Errors**
   ```python
   from utils import validate_environment_setup
   results = validate_environment_setup(Path("envs/check_creativity"))
   print(results['overall_status'])
   ```

3. **Import Errors**
   - Ensure you're running from the correct directory
   - Check that all dependencies are installed
   - Run the test scripts to validate setup

### Performance Issues

- **Slow reward calculation**: Consider reducing `num_train_samples`
- **Memory usage**: Use batch processing for large datasets
- **NLTK downloads**: Run setup once to cache required data

## Dependencies

- `numpy`: Numerical computations
- `nltk`: Natural language processing
- `datasets`: HuggingFace datasets
- `verifiers`: RLVR framework
- `torch`: PyTorch for model operations
- `toml` (optional): Configuration validation

## Testing and Development

The environment includes comprehensive testing infrastructure:

### Test Suites

1. **Core Functionality Tests** (`test_creativity_environment.py`)
   - Reward function validation
   - Data generation testing
   - Environment integration
   - Evaluation system validation

2. **Integration Tests** (`test_integration.py`)
   - Prime-rl registry integration
   - Configuration compatibility
   - End-to-end workflow validation
   - Performance benchmarking

3. **Complete Integration Test** (`integration_test.py`)
   - Environment loading through registry
   - Reward calculation patterns
   - Data generation and augmentation
   - Evaluation system functionality
   - Configuration validation
   - Complete pipeline testing

### Mock Systems for Development

#### Mock Inference (`mock_inference.py`)
Complete mock inference system for development and testing:

```python
from mock_inference import MockCreativityInference

# Initialize mock system
mock_system = MockCreativityInference()

# Generate creative response
response = mock_system.generate_response(
    prompt="Write about dreams",
    max_tokens=200,
    temperature=0.8
)

print(f"Generated: {response.content}")
print(f"Creativity Score: {response.metadata['total_creativity_score']}")
```

#### Mock Server (`mock_server.py`)
FastAPI-based mock server with OpenAI-compatible API:

```bash
# Start mock server
python mock_server.py --host localhost --port 8888

# Test with curl
curl -X POST http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mock-creativity-model",
    "messages": [{"role": "user", "content": "Write creatively about time"}],
    "max_tokens": 300,
    "temperature": 0.8
  }'
```

#### Mock Trainer (`mock_trainer.py`)
Complete training simulation for macOS development:

```bash
# Run mock training with configuration
python mock_trainer.py --config trainer_config.toml --max-steps 100

# Run with custom parameters
python mock_trainer.py --max-steps 50 --creativity-target 7.0 --batch-size 4
```

### Alternative Inference Configurations

#### Ollama Local Setup (`inference_config_ollama.toml`)
For local inference with Ollama:

```toml
[model]
model_name = "qwen2.5:0.5b-instruct"
base_url = "http://localhost:11434"
api_type = "ollama"

[generation]
temperature = 0.8
max_tokens = 300
stream = true
```

#### Mock Configuration (`inference_config_mock.toml`)
For development with mock responses:

```toml
[model]
model_name = "mock-creativity-model"
api_type = "mock"
mock_mode = "creative"

[mock_responses]
storytelling = [
    "Creative story responses...",
]
```

### Running Tests

```bash
# Core functionality tests
python test_creativity_environment.py

# Integration tests
python test_integration.py

# Complete integration validation
python integration_test.py

# Mock system tests
python mock_inference.py  # Demo mock inference
python mock_trainer.py --max-steps 20  # Quick training simulation
```

## Contributing

When extending the creativity environment:

1. **Add new creativity metrics**: Extend the reward function with additional components
2. **Create new prompt categories**: Add categories in `data_utils.py`
3. **Enhance evaluation**: Add new rubrics in `evaluation_rubrics.py`
4. **Improve integration**: Update configuration files as needed

## License

This creativity environment is part of the prime-rl project and follows the same licensing terms.