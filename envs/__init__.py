#!/usr/bin/env python3
"""
RLVR Environments Package

This package contains custom environments for Reinforcement Learning from Verifiable Rewards (RLVR).
Each environment provides specialized reward functions and training configurations for specific tasks.
"""

# Import available environments
try:
    from .check_creativity import (
        load_creativity_environment, 
        CreativityEnvironmentLoader,
        CreativityEnvironment,
        quick_evaluate,
        quick_analysis,
        setup_environment as setup_creativity_environment
    )
    CREATIVITY_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Creativity environment not available: {e}")
    CREATIVITY_AVAILABLE = False

# Environment registry for easy discovery
AVAILABLE_ENVIRONMENTS = {}

if CREATIVITY_AVAILABLE:
    AVAILABLE_ENVIRONMENTS['creativity'] = {
        'name': 'Creativity Text Generation',
        'description': 'Environment for training creative text generation using linguistic diversity metrics',
        'loader_function': 'load_creativity_environment',
        'config_files': ['trainer_config.toml', 'orchestrator_config.toml', 'inference_config.toml'],
        'reward_metrics': ['entropy', 'distinct_ratio', 'uncommon_words', 'bigram_diversity'],
        'tags': ['creative-writing', 'text-generation', 'diversity']
    }

__all__ = []

# Add creativity environment exports if available
if CREATIVITY_AVAILABLE:
    __all__.extend([
        'load_creativity_environment',
        'CreativityEnvironmentLoader', 
        'CreativityEnvironment',
        'quick_evaluate',
        'quick_analysis',
        'setup_creativity_environment'
    ])

# Add registry
__all__.extend(['AVAILABLE_ENVIRONMENTS'])

def list_environments():
    """List all available environments with their descriptions."""
    for env_id, env_info in AVAILABLE_ENVIRONMENTS.items():
        print(f"{env_id}: {env_info['name']}")
        print(f"  Description: {env_info['description']}")
        print(f"  Tags: {', '.join(env_info['tags'])}")
        print()

def get_environment_info(env_id: str):
    """Get detailed information about a specific environment."""
    return AVAILABLE_ENVIRONMENTS.get(env_id, None)