#!/usr/bin/env python3
"""
Comprehensive Creativity Environment Class for RLVR Training

This module provides a complete environment implementation with proper
reset/step methods, batch processing, and comprehensive integration
with the RLVR training pipeline.
"""

import json
import logging
import random
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
from pathlib import Path

from .reward import reward_function
from .data_utils import CreativeTextGenerator, DataAugmentation, load_external_creative_datasets
from .creativity_loader import CreativityEnvironmentLoader

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CreativityState:
    """State representation for creativity environment."""
    prompt: str
    prompt_id: str
    category: str
    target_metrics: Dict[str, float]
    step_count: int
    history: List[str]
    metadata: Dict[str, Any]


@dataclass
class CreativityAction:
    """Action representation (text response from model)."""
    text: str
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class CreativityObservation:
    """Observation returned to the model."""
    prompt: str
    context: str
    requirements: str
    metadata: Dict[str, Any]


class CreativityMetrics:
    """
    Comprehensive metrics tracking for creativity evaluation.
    
    Tracks various aspects of creativity and provides detailed
    analysis for training insights.
    """
    
    def __init__(self):
        """Initialize metrics tracking."""
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.scores = {
            'entropy': [],
            'distinct_ratio': [],
            'uncommon_words': [],
            'bigram_diversity': [],
            'sentence_var': [],
            'word_var': [],
            'ending_var': [],
            'total_score': []
        }
        
        self.detailed_metrics = []
        self.step_rewards = []
    
    def add_score(self, text: str, reward_weights: Dict[str, float]):
        """Add a new creativity score to tracking."""
        try:
            # Calculate individual components
            import nltk
            from collections import Counter
            from nltk.util import bigrams
            from nltk.corpus import words as nltk_words
            
            # Ensure NLTK data is available
            try:
                nltk.data.find('tokenizers/punkt')
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('words', quiet=True)
            
            COMMON_WORDS = set(nltk_words.words())
            
            # Process text
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            words = [w.lower() for w in words if w.isalpha()]
            
            if len(words) == 0:
                return 0.0
            
            # Calculate individual metrics
            word_freqs = Counter(words)
            probs = [count / len(words) for count in word_freqs.values()]
            entropy_score = -sum(p * np.log2(p) for p in probs if p > 0)
            
            distinct_score = len(set(words)) / len(words)
            
            uncommon_words = [w for w in words if w not in COMMON_WORDS]
            uncommon_score = len(uncommon_words) / len(words)
            
            word_bigrams = list(bigrams(words))
            bigram_score = len(set(word_bigrams)) / (len(word_bigrams) + 1e-8)
            
            sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
            sentence_var_score = np.std(sentence_lengths) if len(sentence_lengths) > 1 else 0
            
            word_lengths = [len(w) for w in words]
            word_var_score = np.std(word_lengths)
            
            sentence_endings = [s.strip()[-1] if s.strip() else '.' for s in sentences]
            end_counts = Counter(sentence_endings)
            end_probs = [count / len(sentences) for count in end_counts.values()]
            end_var_score = -sum(p * np.log2(p) for p in end_probs if p > 0)
            
            # Total score
            total_score = reward_function(text, **reward_weights)
            
            # Store metrics
            self.scores['entropy'].append(entropy_score)
            self.scores['distinct_ratio'].append(distinct_score)
            self.scores['uncommon_words'].append(uncommon_score)
            self.scores['bigram_diversity'].append(bigram_score)
            self.scores['sentence_var'].append(sentence_var_score)
            self.scores['word_var'].append(word_var_score)
            self.scores['ending_var'].append(end_var_score)
            self.scores['total_score'].append(total_score)
            
            # Detailed metrics
            detailed = {
                'text_length': len(text),
                'word_count': len(words),
                'sentence_count': len(sentences),
                'unique_words': len(set(words)),
                'entropy': entropy_score,
                'distinct_ratio': distinct_score,
                'uncommon_ratio': uncommon_score,
                'bigram_diversity': bigram_score,
                'sentence_variance': sentence_var_score,
                'word_variance': word_var_score,
                'ending_variety': end_var_score,
                'total_creativity': total_score
            }
            
            self.detailed_metrics.append(detailed)
            return total_score
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return 0.0
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        summary = {}
        
        for metric_name, values in self.scores.items():
            if values:
                summary[metric_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'count': len(values)
                }
        
        # Calculate correlations if we have enough data
        if len(self.detailed_metrics) > 5:
            try:
                import pandas as pd
                df = pd.DataFrame(self.detailed_metrics)
                correlations = df.corr()['total_creativity'].to_dict()
                summary['correlations_with_total'] = correlations
            except ImportError:
                pass  # pandas not available
        
        return summary


class CreativityEnvironment:
    """
    Comprehensive creativity environment for RLVR training.
    
    Provides complete environment functionality including:
    - State management and transitions
    - Reward calculation and tracking
    - Batch processing capabilities
    - Comprehensive metrics and analysis
    - Integration with RL training pipeline
    """
    
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        reward_weights: Optional[Dict[str, float]] = None,
        data_dir: Optional[Path] = None,
        seed: int = 42
    ):
        """
        Initialize the creativity environment.
        
        Args:
            config: Environment configuration
            reward_weights: Custom reward function weights
            data_dir: Directory with additional creative text data
            seed: Random seed for reproducibility
        """
        self.config = config or {}
        self.seed = seed
        self.data_dir = Path(data_dir) if data_dir else None
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize components
        self.text_generator = CreativeTextGenerator(seed=seed)
        self.loader = CreativityEnvironmentLoader(
            reward_weights=reward_weights,
            seed=seed,
            **self.config.get('loader_args', {})
        )
        
        # Load external data if available
        self.external_data = load_external_creative_datasets(self.data_dir)
        
        # Initialize metrics tracking
        self.metrics = CreativityMetrics()
        
        # Environment state
        self.current_state = None
        self.episode_count = 0
        self.total_steps = 0
        
        # Configuration
        self.max_episode_steps = self.config.get('max_episode_steps', 1)  # Single turn by default
        self.reward_weights = reward_weights or self.loader.reward_weights
        
        logger.info(f"Initialized CreativityEnvironment with {len(self.external_data.get('samples', []))} external samples")
    
    def reset(self, options: Optional[Dict[str, Any]] = None) -> CreativityObservation:
        """
        Reset environment and return initial observation.
        
        Args:
            options: Optional reset parameters
            
        Returns:
            Initial observation for the episode
        """
        self.episode_count += 1
        options = options or {}
        
        # Generate or select prompt
        if 'prompt' in options:
            prompt = options['prompt']
            category = options.get('category', 'custom')
        else:
            prompt = self._generate_episode_prompt()
            category = self._classify_prompt(prompt)
        
        # Create initial state
        self.current_state = CreativityState(
            prompt=prompt,
            prompt_id=f"episode_{self.episode_count}",
            category=category,
            target_metrics={},  # Could be set based on prompt type
            step_count=0,
            history=[],
            metadata={
                'episode_id': self.episode_count,
                'seed': self.seed,
                'generated_at_step': self.total_steps
            }
        )
        
        # Create observation
        observation = self._create_observation()
        
        logger.debug(f"Reset environment for episode {self.episode_count}, prompt: {prompt[:50]}...")
        return observation
    
    def step(self, action: Union[str, CreativityAction]) -> Tuple[CreativityObservation, float, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.
        
        Args:
            action: Action to take (text response or CreativityAction)
            
        Returns:
            Tuple of (observation, reward, done, info)
        """
        if self.current_state is None:
            raise ValueError("Environment must be reset before taking steps")
        
        # Convert action to standard format
        if isinstance(action, str):
            action = CreativityAction(text=action)
        
        # Update state
        self.current_state.step_count += 1
        self.current_state.history.append(action.text)
        self.total_steps += 1
        
        # Calculate reward
        reward = self._calculate_reward(action.text)
        
        # Update metrics
        self.metrics.add_score(action.text, self.reward_weights)
        self.metrics.step_rewards.append(reward)
        
        # Check if episode is done
        done = self._is_episode_done()
        
        # Create next observation (if not done)
        next_observation = self._create_observation() if not done else None
        
        # Create info dictionary
        info = {
            'reward_components': self._get_reward_components(action.text),
            'step_count': self.current_state.step_count,
            'episode_id': self.episode_count,
            'prompt_category': self.current_state.category,
            'text_length': len(action.text),
            'metadata': action.metadata or {}
        }
        
        logger.debug(f"Step {self.current_state.step_count}: reward={reward:.3f}, done={done}")
        
        return next_observation, reward, done, info
    
    def _generate_episode_prompt(self) -> str:
        """Generate a prompt for the episode."""
        # Use loader's prompt generation
        prompts = self.loader.generate_creative_prompts(1)
        base_prompt = prompts[0]
        
        # Optionally augment with external data
        if self.external_data.get('prompts') and random.random() < 0.2:
            external_prompt = random.choice(self.external_data['prompts'])
            return external_prompt
        
        # Apply data augmentation
        augmented = DataAugmentation.augment_prompt_variations([base_prompt])
        return random.choice(augmented)
    
    def _classify_prompt(self, prompt: str) -> str:
        """Classify prompt into category."""
        return self.loader._classify_prompt(prompt)
    
    def _create_observation(self) -> CreativityObservation:
        """Create observation from current state."""
        if self.current_state is None:
            raise ValueError("No current state available")
        
        # Create contextual information
        context = self._build_context()
        requirements = self._build_requirements()
        
        observation = CreativityObservation(
            prompt=self.current_state.prompt,
            context=context,
            requirements=requirements,
            metadata={
                'category': self.current_state.category,
                'step_count': self.current_state.step_count,
                'episode_id': self.episode_count,
                'history_length': len(self.current_state.history)
            }
        )
        
        return observation
    
    def _build_context(self) -> str:
        """Build contextual information for the observation."""
        context_parts = [
            "Focus on creating original, creative text that demonstrates:",
            "• Diverse vocabulary and unique word choices",
            "• Varied sentence structures and lengths", 
            "• Creative use of punctuation and rhythm",
            "• Original ideas and unexpected connections",
            "• Rich imagery and expressive language"
        ]
        
        # Add category-specific guidance
        category = self.current_state.category
        if category == "storytelling":
            context_parts.append("• Compelling narrative development")
        elif category == "poetry":
            context_parts.append("• Musical language and creative form")
        elif category == "descriptive":
            context_parts.append("• Vivid sensory details and imagery")
        elif category == "philosophical":
            context_parts.append("• Deep insights and thoughtful reflection")
        elif category == "experimental":
            context_parts.append("• Bold creative techniques and innovation")
        
        return "\n".join(context_parts)
    
    def _build_requirements(self) -> str:
        """Build specific requirements for the response."""
        requirements = [
            f"Respond creatively to: {self.current_state.prompt}",
            "Aim for 100-300 words",
            "Use varied vocabulary and sentence structures",
            "Include creative language choices"
        ]
        
        return "\n".join(requirements)
    
    def _calculate_reward(self, text: str) -> float:
        """Calculate reward for the given text."""
        return self.loader.creativity_reward_function(text, self.current_state.metadata)
    
    def _get_reward_components(self, text: str) -> Dict[str, float]:
        """Get breakdown of reward components."""
        try:
            # Calculate individual components (similar to metrics but focused on rewards)
            import nltk
            from collections import Counter
            from nltk.util import bigrams
            from nltk.corpus import words as nltk_words
            
            # Ensure NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('words', quiet=True)
            
            COMMON_WORDS = set(nltk_words.words())
            
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            words = [w.lower() for w in words if w.isalpha()]
            
            if len(words) == 0:
                return {k: 0.0 for k in self.reward_weights.keys()}
            
            # Individual component calculations
            word_freqs = Counter(words)
            probs = [count / len(words) for count in word_freqs.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            
            distinct = len(set(words)) / len(words)
            
            uncommon_words = [w for w in words if w not in COMMON_WORDS]
            uncommon = len(uncommon_words) / len(words)
            
            word_bigrams = list(bigrams(words))
            bigram_diversity = len(set(word_bigrams)) / (len(word_bigrams) + 1e-8)
            
            sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
            sentence_var = np.std(sentence_lengths) if len(sentences) > 1 else 0
            
            word_lengths = [len(w) for w in words]
            word_var = np.std(word_lengths)
            
            sentence_endings = [s.strip()[-1] if s.strip() else '.' for s in sentences]
            end_counts = Counter(sentence_endings)
            end_probs = [count / len(sentences) for count in end_counts.values()]
            ending_var = -sum(p * np.log2(p) for p in end_probs if p > 0)
            
            components = {
                'entropy': entropy * self.reward_weights.get('w_entropy', 1.0),
                'distinct': distinct * self.reward_weights.get('w_distinct', 1.0),
                'uncommon': uncommon * self.reward_weights.get('w_uncommon', 1.0),
                'bigrams': bigram_diversity * self.reward_weights.get('w_bigrams', 1.0),
                'sentence_var': sentence_var * self.reward_weights.get('w_sentence_len_var', 1.0),
                'word_var': word_var * self.reward_weights.get('w_word_len_var', 1.0),
                'ending_var': ending_var * self.reward_weights.get('w_sentence_end_var', 1.0),
            }
            
            return components
            
        except Exception as e:
            logger.error(f"Error calculating reward components: {e}")
            return {k: 0.0 for k in self.reward_weights.keys()}
    
    def _is_episode_done(self) -> bool:
        """Check if episode is complete."""
        return self.current_state.step_count >= self.max_episode_steps
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get comprehensive metrics summary."""
        return self.metrics.get_summary()
    
    def batch_process(self, texts: List[str]) -> List[float]:
        """
        Process multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to evaluate
            
        Returns:
            List of creativity scores
        """
        scores = []
        
        for text in texts:
            score = self._calculate_reward(text)
            scores.append(score)
            
            # Also update metrics
            self.metrics.add_score(text, self.reward_weights)
        
        return scores
    
    def save_metrics(self, filepath: Path):
        """Save metrics to file for analysis."""
        summary = self.get_metrics_summary()
        summary['detailed_metrics'] = self.metrics.detailed_metrics
        summary['step_rewards'] = self.metrics.step_rewards
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
    
    def load_checkpoint(self, filepath: Path):
        """Load environment state from checkpoint."""
        with open(filepath, 'r') as f:
            checkpoint = json.load(f)
        
        self.episode_count = checkpoint.get('episode_count', 0)
        self.total_steps = checkpoint.get('total_steps', 0)
        
        # Restore metrics if available
        if 'metrics' in checkpoint:
            metrics_data = checkpoint['metrics']
            for key, values in metrics_data.get('scores', {}).items():
                if key in self.metrics.scores:
                    self.metrics.scores[key] = values
    
    def save_checkpoint(self, filepath: Path):
        """Save environment state to checkpoint."""
        checkpoint = {
            'episode_count': self.episode_count,
            'total_steps': self.total_steps,
            'config': self.config,
            'reward_weights': self.reward_weights,
            'metrics': {
                'scores': self.metrics.scores,
                'detailed_metrics': self.metrics.detailed_metrics[-100:],  # Save last 100
                'step_rewards': self.metrics.step_rewards[-100:]
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(checkpoint, f, indent=2, default=str)
    
    def render(self, mode: str = 'human') -> Optional[str]:
        """
        Render the environment state.
        
        Args:
            mode: Rendering mode
            
        Returns:
            String representation if mode != 'human'
        """
        if self.current_state is None:
            output = "Environment not initialized"
        else:
            output = f"""
=== Creativity Environment State ===
Episode: {self.episode_count}
Step: {self.current_state.step_count}
Category: {self.current_state.category}
Prompt: {self.current_state.prompt}

Recent History:
{chr(10).join(f"  {i+1}: {h[:100]}..." for i, h in enumerate(self.current_state.history[-3:]))}

Metrics Summary:
{json.dumps(self.get_metrics_summary(), indent=2, default=str)}
"""
        
        if mode == 'human':
            print(output)
            return None
        else:
            return output