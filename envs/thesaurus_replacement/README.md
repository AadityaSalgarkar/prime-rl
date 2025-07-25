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

## Implementation Details

### Data Augmentation Strategy
- Uses the repo github.com/zaibacu/thesaurus to find near-synonyms
- Among the words with available synonyms, randomly selects 20% of words in each sentence for replacement
- Only replaces words that have available synonyms
- Maintains grammatical structure and sentence length where possible

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

# Instructions to run
The environment can be run using:
```
  uv run rl \
    --trainer @
  envs/thesaurus_replacement/trainer_config.toml \
    --orchestrator @
  envs/thesaurus_replacement/orchestrator_config.toml \
    --inference @
  envs/thesaurus_replacement/inference_config.toml
```
