#!/usr/bin/env python3
"""
Creativity Environment Package for RLVR Training

A comprehensive environment for training models on creative text generation
using multiple linguistic diversity metrics as rewards.
"""

from .reward import reward_function
from .creativity_loader import CreativityEnvironmentLoader, load_creativity_environment
from .creativity_env import CreativityEnvironment, CreativityAction, CreativityState
from .data_utils import CreativeTextGenerator, DataAugmentation
from .evaluation_rubrics import CreativityRubric, CreativityTracker, CreativityBenchmark
from .utils import CreativityConfig, ConfigValidator, CreativityUtils

__version__ = "1.0.0"
__author__ = "Prime RL Team"

__all__ = [
    # Core functionality
    "reward_function",
    "load_creativity_environment",
    
    # Environment classes
    "CreativityEnvironmentLoader", 
    "CreativityEnvironment",
    "CreativityAction",
    "CreativityState",
    
    # Data utilities
    "CreativeTextGenerator",
    "DataAugmentation",
    
    # Evaluation system
    "CreativityRubric",
    "CreativityTracker", 
    "CreativityBenchmark",
    
    # Configuration and utilities
    "CreativityConfig",
    "ConfigValidator",
    "CreativityUtils",
]

# Environment metadata for registry
ENVIRONMENT_INFO = {
    "name": "creativity",
    "version": __version__,
    "description": "RLVR environment for creative text generation training",
    "tags": ["creative-writing", "text-generation", "diversity", "language-modeling"],
    "reward_metrics": [
        "entropy", "distinct_ratio", "uncommon_words", "bigram_diversity",
        "sentence_variance", "word_variance", "ending_variety"
    ],
    "prompt_categories": ["storytelling", "poetry", "descriptive", "philosophical", "experimental"],
    "supported_configurations": ["trainer", "orchestrator", "inference"],
}

# Setup helper
def setup_environment():
    """Setup the creativity environment with required dependencies."""
    return CreativityUtils.setup_nltk_dependencies()

# Quick access functions
def quick_evaluate(text: str, weights: dict = None) -> float:
    """Quick creativity evaluation of text."""
    if weights is None:
        weights = {
            'w_entropy': 1.0,
            'w_distinct': 1.0, 
            'w_uncommon': 0.8,
            'w_bigrams': 1.2,
            'w_sentence_len_var': 0.6,
            'w_word_len_var': 0.4,
            'w_sentence_end_var': 0.5,
        }
    return reward_function(text, **weights)

def quick_analysis(text: str, prompt: str = "", category: str = "") -> dict:
    """Quick creativity analysis with detailed breakdown."""
    rubric = CreativityRubric()
    analysis = rubric.evaluate_text(text, prompt, category)
    
    return {
        'total_score': analysis.total_score,
        'component_scores': analysis.component_scores,
        'grade': rubric.grade_creativity_level(analysis.total_score),
        'text_stats': analysis.text_stats
    }