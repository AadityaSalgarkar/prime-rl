#!/usr/bin/env python3
"""
Utility functions and configuration validation for creativity environment.

This module provides helper functions, configuration validation,
and integration utilities for the creativity RLVR environment.
"""

import json
import logging
import warnings
from typing import Dict, List, Optional, Any, Union, Tuple
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
import tempfile

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CreativityConfig:
    """
    Configuration dataclass for creativity environment.
    
    Provides structured configuration with validation and defaults.
    """
    # Data generation settings
    num_train_samples: int = 1000
    num_eval_samples: int = 100
    max_prompt_length: int = 150
    seed: int = 42
    
    # Reward function weights
    reward_weights: Dict[str, float] = field(default_factory=lambda: {
        'w_entropy': 1.0,
        'w_distinct': 1.0,
        'w_uncommon': 0.8,
        'w_bigrams': 1.2,
        'w_sentence_len_var': 0.6,
        'w_word_len_var': 0.4,
        'w_sentence_end_var': 0.5,
    })
    
    # Environment settings
    max_episode_steps: int = 1
    batch_size: int = 8
    max_concurrent: int = 4
    
    # Generation settings
    temperature: float = 0.8
    top_p: float = 0.9
    max_tokens: int = 300
    repetition_penalty: float = 1.1
    
    # Evaluation settings
    creativity_threshold: float = 3.0
    enable_detailed_metrics: bool = True
    save_analysis: bool = True
    
    # Data paths
    external_data_dir: Optional[Path] = None
    output_dir: Optional[Path] = None
    
    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()
    
    def validate(self) -> bool:
        """
        Validate configuration parameters.
        
        Returns:
            True if valid, raises ValueError if invalid
        """
        # Validate sample counts
        if self.num_train_samples <= 0:
            raise ValueError("num_train_samples must be positive")
        if self.num_eval_samples <= 0:
            raise ValueError("num_eval_samples must be positive")
        
        # Validate reward weights
        if not isinstance(self.reward_weights, dict):
            raise ValueError("reward_weights must be a dictionary")
        
        required_weights = [
            'w_entropy', 'w_distinct', 'w_uncommon', 'w_bigrams',
            'w_sentence_len_var', 'w_word_len_var', 'w_sentence_end_var'
        ]
        
        for weight_name in required_weights:
            if weight_name not in self.reward_weights:
                warnings.warn(f"Missing reward weight: {weight_name}, using default")
                self.reward_weights[weight_name] = 1.0
            elif not isinstance(self.reward_weights[weight_name], (int, float)):
                raise ValueError(f"Reward weight {weight_name} must be numeric")
            elif self.reward_weights[weight_name] < 0:
                warnings.warn(f"Negative reward weight: {weight_name}")
        
        # Validate generation parameters
        if not 0.1 <= self.temperature <= 2.0:
            warnings.warn(f"Temperature {self.temperature} outside recommended range [0.1, 2.0]")
        
        if not 0.1 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0.1 and 1.0")
        
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        
        # Validate thresholds
        if self.creativity_threshold < 0:
            warnings.warn("Negative creativity threshold may not be meaningful")
        
        logger.info("Configuration validation completed successfully")
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        config_dict = {}
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, Path):
                config_dict[field_name] = str(field_value)
            else:
                config_dict[field_name] = field_value
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'CreativityConfig':
        """Create configuration from dictionary."""
        # Handle Path fields
        if 'external_data_dir' in config_dict and config_dict['external_data_dir']:
            config_dict['external_data_dir'] = Path(config_dict['external_data_dir'])
        if 'output_dir' in config_dict and config_dict['output_dir']:
            config_dict['output_dir'] = Path(config_dict['output_dir'])
        
        return cls(**config_dict)


class ConfigValidator:
    """
    Comprehensive configuration validator for creativity environment.
    
    Validates TOML configuration files and provides recommendations.
    """
    
    def __init__(self):
        """Initialize validator with expected configuration structure."""
        self.trainer_schema = {
            'required_sections': ['model', 'data', 'optimizer'],
            'optional_sections': ['scheduler', 'training', 'checkpointing', 'logging', 'generation'],
            'creativity_requirements': {
                'data': {
                    'creativity': ['num_train_samples', 'num_eval_samples', 'reward_weights']
                }
            }
        }
        
        self.orchestrator_schema = {
            'required_sections': ['environment', 'sampling'],
            'optional_sections': ['rewards', 'evaluation', 'metrics', 'logging'],
            'creativity_requirements': {
                'environment': {
                    'creativity': ['num_train_samples', 'reward_weights']
                }
            }
        }
        
        self.inference_schema = {
            'required_sections': ['model', 'generation'],
            'optional_sections': ['evaluation', 'output', 'performance'],
            'creativity_requirements': {
                'creativity_weights': []  # Should have reward weight keys
            }
        }
    
    def validate_trainer_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate trainer configuration.
        
        Args:
            config: Parsed TOML configuration
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required sections
        for section in self.trainer_schema['required_sections']:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Check creativity-specific requirements
        if 'data' in config:
            if 'creativity' not in config['data']:
                errors.append("Missing creativity section in data configuration")
            else:
                creativity_config = config['data']['creativity']
                for required_key in self.trainer_schema['creativity_requirements']['data']['creativity']:
                    if required_key not in creativity_config:
                        errors.append(f"Missing creativity data config: {required_key}")
                
                # Validate reward weights structure
                if 'reward_weights' in creativity_config:
                    weights = creativity_config['reward_weights']
                    if not isinstance(weights, dict):
                        errors.append("reward_weights must be a dictionary")
                    else:
                        required_weights = ['w_entropy', 'w_distinct', 'w_uncommon', 'w_bigrams']
                        for weight in required_weights:
                            if weight not in weights:
                                errors.append(f"Missing reward weight: {weight}")
        
        # Validate model configuration for creativity
        if 'model' in config:
            model_config = config['model']
            if 'model_name' not in model_config:
                errors.append("Missing model_name in model configuration")
        
        # Check generation parameters if present
        if 'generation' in config:
            gen_config = config['generation']
            if 'temperature' in gen_config:
                temp = gen_config['temperature']
                if not isinstance(temp, (int, float)) or temp <= 0:
                    errors.append("Invalid temperature value")
        
        return len(errors) == 0, errors
    
    def validate_orchestrator_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate orchestrator configuration.
        
        Args:
            config: Parsed TOML configuration
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required sections
        for section in self.orchestrator_schema['required_sections']:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Check environment configuration
        if 'environment' in config:
            env_config = config['environment']
            if 'env_id' not in env_config or env_config['env_id'] != 'creativity':
                errors.append("Environment ID must be 'creativity'")
            
            if 'creativity' in env_config:
                creativity_config = env_config['creativity']
                for required_key in self.orchestrator_schema['creativity_requirements']['environment']['creativity']:
                    if required_key not in creativity_config:
                        errors.append(f"Missing orchestrator creativity config: {required_key}")
        
        # Check sampling configuration
        if 'sampling' in config:
            sampling_config = config['sampling']
            required_sampling = ['max_tokens', 'temperature']
            for key in required_sampling:
                if key not in sampling_config:
                    errors.append(f"Missing sampling parameter: {key}")
        
        return len(errors) == 0, errors
    
    def validate_inference_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate inference configuration.
        
        Args:
            config: Parsed TOML configuration
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        # Check required sections
        for section in self.inference_schema['required_sections']:
            if section not in config:
                errors.append(f"Missing required section: {section}")
        
        # Check creativity weights
        if 'creativity_weights' not in config:
            errors.append("Missing creativity_weights section")
        else:
            weights = config['creativity_weights']
            required_weights = ['w_entropy', 'w_distinct', 'w_uncommon', 'w_bigrams']
            for weight in required_weights:
                if weight not in weights:
                    errors.append(f"Missing creativity weight: {weight}")
        
        # Check generation parameters
        if 'generation' in config:
            gen_config = config['generation']
            if 'max_tokens' not in gen_config:
                errors.append("Missing max_tokens in generation config")
        
        return len(errors) == 0, errors
    
    def validate_all_configs(self, config_dir: Path) -> Dict[str, Any]:
        """
        Validate all configuration files in a directory.
        
        Args:
            config_dir: Directory containing configuration files
            
        Returns:
            Validation results dictionary
        """
        results = {
            'trainer_config.toml': {'valid': False, 'errors': ['File not found']},
            'orchestrator_config.toml': {'valid': False, 'errors': ['File not found']},
            'inference_config.toml': {'valid': False, 'errors': ['File not found']},
            'overall_valid': False
        }
        
        try:
            import toml
        except ImportError:
            logger.error("TOML library not available for configuration validation")
            return results
        
        # Validate trainer config
        trainer_path = config_dir / 'trainer_config.toml'
        if trainer_path.exists():
            try:
                trainer_config = toml.load(trainer_path)
                valid, errors = self.validate_trainer_config(trainer_config)
                results['trainer_config.toml'] = {'valid': valid, 'errors': errors}
            except Exception as e:
                results['trainer_config.toml'] = {'valid': False, 'errors': [f"Parse error: {e}"]}
        
        # Validate orchestrator config
        orchestrator_path = config_dir / 'orchestrator_config.toml'
        if orchestrator_path.exists():
            try:
                orchestrator_config = toml.load(orchestrator_path)
                valid, errors = self.validate_orchestrator_config(orchestrator_config)
                results['orchestrator_config.toml'] = {'valid': valid, 'errors': errors}
            except Exception as e:
                results['orchestrator_config.toml'] = {'valid': False, 'errors': [f"Parse error: {e}"]}
        
        # Validate inference config
        inference_path = config_dir / 'inference_config.toml'
        if inference_path.exists():
            try:
                inference_config = toml.load(inference_path)
                valid, errors = self.validate_inference_config(inference_config)
                results['inference_config.toml'] = {'valid': valid, 'errors': errors}
            except Exception as e:
                results['inference_config.toml'] = {'valid': False, 'errors': [f"Parse error: {e}"]}
        
        # Determine overall validity
        all_valid = all(
            result['valid'] for key, result in results.items() 
            if key.endswith('.toml')
        )
        results['overall_valid'] = all_valid
        
        return results


class CreativityUtils:
    """
    Utility functions for creativity environment operations.
    
    Provides helper functions for common operations and integrations.
    """
    
    @staticmethod
    def setup_nltk_dependencies():
        """Ensure NLTK dependencies are downloaded."""
        try:
            import nltk
            
            # Required NLTK data
            required_data = ['punkt', 'words']
            
            for data_name in required_data:
                try:
                    nltk.data.find(f'tokenizers/{data_name}' if data_name == 'punkt' else f'corpora/{data_name}')
                except LookupError:
                    logger.info(f"Downloading NLTK data: {data_name}")
                    nltk.download(data_name, quiet=True)
            
            logger.info("NLTK dependencies are ready")
            return True
            
        except ImportError:
            logger.error("NLTK not available")
            return False
        except Exception as e:
            logger.error(f"Failed to setup NLTK dependencies: {e}")
            return False
    
    @staticmethod
    def validate_text_for_creativity(text: str) -> Tuple[bool, List[str]]:
        """
        Validate text for creativity evaluation.
        
        Args:
            text: Text to validate
            
        Returns:
            Tuple of (is_valid, warnings)
        """
        warnings_list = []
        
        if not isinstance(text, str):
            return False, ["Text must be a string"]
        
        text = text.strip()
        
        if len(text) == 0:
            return False, ["Text cannot be empty"]
        
        if len(text) < 20:
            warnings_list.append("Text is very short, creativity metrics may not be meaningful")
        
        if len(text) > 2000:
            warnings_list.append("Text is very long, processing may be slow")
        
        # Check for basic structure
        if '.' not in text and '!' not in text and '?' not in text:
            warnings_list.append("Text lacks sentence-ending punctuation")
        
        # Check for minimal word variety
        words = text.lower().split()
        unique_words = set(words)
        
        if len(words) > 10 and len(unique_words) / len(words) < 0.3:
            warnings_list.append("Text has low word diversity")
        
        return True, warnings_list
    
    @staticmethod
    def normalize_creativity_score(score: float, method: str = "percentile") -> float:
        """
        Normalize creativity scores to 0-1 range.
        
        Args:
            score: Raw creativity score
            method: Normalization method ("percentile", "sigmoid", "linear")
            
        Returns:
            Normalized score between 0 and 1
        """
        if method == "percentile":
            # Based on empirical score distributions
            # These thresholds are approximate and could be refined
            if score <= 1.0:
                return 0.1
            elif score <= 3.0:
                return 0.3
            elif score <= 5.0:
                return 0.5
            elif score <= 7.0:
                return 0.7
            elif score <= 9.0:
                return 0.9
            else:
                return 1.0
        
        elif method == "sigmoid":
            # Sigmoid normalization centered around score=5
            import math
            return 1 / (1 + math.exp(-(score - 5)))
        
        elif method == "linear":
            # Linear scaling assuming max reasonable score of 15
            return min(1.0, max(0.0, score / 15.0))
        
        else:
            raise ValueError(f"Unknown normalization method: {method}")
    
    @staticmethod
    def create_creativity_report(
        texts: List[str], 
        scores: List[float], 
        prompts: Optional[List[str]] = None,
        output_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive creativity evaluation report.
        
        Args:
            texts: List of texts to analyze
            scores: Corresponding creativity scores
            prompts: Optional list of prompts
            output_path: Optional path to save report
            
        Returns:
            Report dictionary
        """
        if len(texts) != len(scores):
            raise ValueError("Texts and scores lists must have same length")
        
        # Calculate statistics
        report = {
            'summary': {
                'total_texts': len(texts),
                'average_score': np.mean(scores),
                'std_score': np.std(scores),
                'min_score': np.min(scores),
                'max_score': np.max(scores),
                'median_score': np.median(scores)
            },
            'score_distribution': {
                'low_creativity': sum(1 for s in scores if s < 3.0),
                'medium_creativity': sum(1 for s in scores if 3.0 <= s < 7.0),
                'high_creativity': sum(1 for s in scores if s >= 7.0)
            },
            'text_analysis': [],
            'recommendations': []
        }
        
        # Analyze individual texts
        for i, (text, score) in enumerate(zip(texts, scores)):
            prompt = prompts[i] if prompts and i < len(prompts) else None
            
            analysis = {
                'index': i,
                'prompt': prompt,
                'score': score,
                'normalized_score': CreativityUtils.normalize_creativity_score(score),
                'text_length': len(text),
                'word_count': len(text.split()),
                'text_preview': text[:100] + "..." if len(text) > 100 else text
            }
            
            report['text_analysis'].append(analysis)
        
        # Generate recommendations
        avg_score = report['summary']['average_score']
        
        if avg_score < 3.0:
            report['recommendations'].append("Consider using more diverse vocabulary and varied sentence structures")
            report['recommendations'].append("Experiment with creative writing techniques and metaphors")
        elif avg_score < 5.0:
            report['recommendations'].append("Good baseline creativity, focus on increasing word diversity")
            report['recommendations'].append("Try incorporating more unusual words and varied punctuation")
        elif avg_score < 7.0:
            report['recommendations'].append("Strong creativity foundation, focus on advanced techniques")
            report['recommendations'].append("Experiment with complex sentence structures and original expressions")
        else:
            report['recommendations'].append("Excellent creativity levels maintained")
            report['recommendations'].append("Continue exploring innovative language use")
        
        # Save report if path provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            logger.info(f"Creativity report saved to {output_path}")
        
        return report
    
    @staticmethod
    def benchmark_environment_performance(num_samples: int = 100) -> Dict[str, float]:
        """
        Benchmark creativity environment performance.
        
        Args:
            num_samples: Number of samples to benchmark with
            
        Returns:
            Performance metrics dictionary
        """
        import time
        
        # Import environment components
        from reward import reward_function
        from creativity_loader import CreativityEnvironmentLoader
        
        results = {}
        
        # Benchmark reward function
        test_text = """The iridescent butterflies danced through gossamer clouds of possibility, 
                      their wings painting symphonies of color across the canvas of imagination."""
        
        start_time = time.time()
        for _ in range(num_samples):
            reward_function(test_text)
        reward_time = time.time() - start_time
        
        results['reward_function_time_per_call'] = reward_time / num_samples
        results['reward_function_calls_per_second'] = num_samples / reward_time
        
        # Benchmark environment loading
        start_time = time.time()
        loader = CreativityEnvironmentLoader(num_train_samples=num_samples, num_eval_samples=10)
        prompts = loader.generate_creative_prompts(num_samples)
        loading_time = time.time() - start_time
        
        results['environment_loading_time'] = loading_time
        results['prompt_generation_time_per_sample'] = loading_time / num_samples
        
        # Benchmark dataset creation
        start_time = time.time()
        dataset = loader.create_dataset(prompts)
        dataset_time = time.time() - start_time
        
        results['dataset_creation_time'] = dataset_time
        results['dataset_creation_time_per_sample'] = dataset_time / num_samples
        
        logger.info(f"Performance benchmark completed with {num_samples} samples")
        
        return results


def validate_environment_setup(env_dir: Path) -> Dict[str, Any]:
    """
    Comprehensive validation of environment setup.
    
    Args:
        env_dir: Path to environment directory
        
    Returns:
        Validation results
    """
    results = {
        'files_present': {},
        'config_validation': {},
        'dependencies': {},
        'functionality': {},
        'overall_status': 'unknown'
    }
    
    # Check required files
    required_files = [
        'reward.py',
        'creativity_loader.py', 
        'creativity_env.py',
        'data_utils.py',
        'evaluation_rubrics.py',
        'trainer_config.toml',
        'orchestrator_config.toml',
        'inference_config.toml'
    ]
    
    for file_name in required_files:
        file_path = env_dir / file_name
        results['files_present'][file_name] = file_path.exists()
    
    # Validate configurations
    validator = ConfigValidator()
    config_results = validator.validate_all_configs(env_dir)
    results['config_validation'] = config_results
    
    # Check dependencies
    dependencies = {
        'nltk': False,
        'numpy': False,
        'datasets': False,
        'verifiers': False
    }
    
    for dep in dependencies:
        try:
            __import__(dep)
            dependencies[dep] = True
        except ImportError:
            dependencies[dep] = False
    
    results['dependencies'] = dependencies
    
    # Test basic functionality
    functionality_tests = {
        'reward_function': False,
        'environment_loading': False,
        'data_generation': False
    }
    
    try:
        # Test reward function
        import sys
        sys.path.insert(0, str(env_dir))
        
        from reward import reward_function
        test_score = reward_function("Test text for creativity evaluation.")
        functionality_tests['reward_function'] = isinstance(test_score, (int, float))
        
        # Test environment loading
        from creativity_loader import load_creativity_environment
        test_env = load_creativity_environment(num_train_samples=1, num_eval_samples=1)
        functionality_tests['environment_loading'] = test_env is not None
        
        # Test data generation
        from data_utils import CreativeTextGenerator
        generator = CreativeTextGenerator()
        test_text = generator.generate_creative_sample("Write creatively", "short")
        functionality_tests['data_generation'] = len(test_text) > 10
        
    except Exception as e:
        logger.error(f"Functionality testing failed: {e}")
    
    results['functionality'] = functionality_tests
    
    # Determine overall status
    files_ok = all(results['files_present'].values())
    configs_ok = results['config_validation']['overall_valid']
    deps_ok = all(results['dependencies'].values())
    funcs_ok = all(results['functionality'].values())
    
    if files_ok and configs_ok and deps_ok and funcs_ok:
        results['overall_status'] = 'ready'
    elif files_ok and funcs_ok:
        results['overall_status'] = 'functional'
    else:
        results['overall_status'] = 'issues'
    
    return results