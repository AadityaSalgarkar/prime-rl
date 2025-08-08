#!/usr/bin/env python3
"""
Creativity Environment Loader for RLVR Training

This module implements a complete RLVR environment for training models on creativity tasks.
It integrates with the verifiers framework and uses the reward.py function for scoring.
"""

import json
import random
from typing import Dict, List, Optional, Any
import numpy as np
from datasets import Dataset
import verifiers as vf
from verifiers import Environment

from .reward import reward_function


class CreativityEnvironmentLoader:
    """
    Loads and manages the creativity checking environment for RLVR training.
    
    This class handles:
    - Generation of diverse text prompts for creativity evaluation
    - Integration with the reward function from reward.py
    - Batch processing for efficient RL training
    - Data loading and augmentation
    """
    
    def __init__(
        self,
        num_train_samples: int = 2000,
        num_eval_samples: int = 200,
        max_prompt_length: int = 100,
        reward_weights: Optional[Dict[str, float]] = None,
        seed: int = 42,
        **kwargs
    ):
        """
        Initialize the creativity environment loader.
        
        Args:
            num_train_samples: Number of training samples to generate
            num_eval_samples: Number of evaluation samples to generate
            max_prompt_length: Maximum length for generated prompts
            reward_weights: Custom weights for reward function components
            seed: Random seed for reproducibility
        """
        self.num_train_samples = num_train_samples
        self.num_eval_samples = num_eval_samples
        self.max_prompt_length = max_prompt_length
        self.seed = seed
        
        # Default reward weights (can be overridden)
        self.reward_weights = reward_weights or {
            'w_entropy': 1.0,
            'w_distinct': 1.0,
            'w_uncommon': 0.8,
            'w_bigrams': 1.2,
            'w_sentence_len_var': 0.6,
            'w_word_len_var': 0.4,
            'w_sentence_end_var': 0.5,
        }
        
        # Set random seeds
        random.seed(seed)
        np.random.seed(seed)
        
        # Initialize prompt categories for diverse text generation
        self.prompt_categories = self._init_prompt_categories()
    
    def _init_prompt_categories(self) -> Dict[str, List[str]]:
        """Initialize diverse categories of creative writing prompts."""
        return {
            "storytelling": [
                "Write a short story about a character who discovers they can taste colors.",
                "Tell a story from the perspective of the last book in a library.",
                "Create a narrative about a world where emotions have physical weight.",
                "Write about a society where people age backwards.",
                "Describe a day in the life of someone who can only speak in questions.",
                "Tell a story about a character who collects lost memories.",
                "Write about a place where time moves differently for everyone.",
                "Create a story about a character who can see the lifespan of objects.",
            ],
            "poetry": [
                "Write a poem about the sound of silence in different environments.",
                "Create verses about the conversation between day and night.",
                "Compose a poem from the perspective of a forgotten word.",
                "Write about the journey of a raindrop from cloud to sea.",
                "Create a poem about the secret life of shadows.",
                "Write verses about the last star in the universe.",
                "Compose a poem about the taste of different seasons.",
                "Create a piece about the dreams of ancient trees.",
            ],
            "descriptive": [
                "Describe a marketplace that exists only in people's dreams.",
                "Paint with words a landscape that changes based on the observer's mood.",
                "Describe the sensation of reading a book made of light.",
                "Write about a city built entirely from music.",
                "Describe the feeling of walking through someone else's memories.",
                "Paint a verbal picture of a garden that grows backwards in time.",
                "Describe a conversation between different types of weather.",
                "Write about the texture and taste of various abstract concepts.",
            ],
            "philosophical": [
                "Explore what it means to be the pause between musical notes.",
                "Contemplate the relationship between forgotten languages and lost thoughts.",
                "Discuss the weight of unspoken words in different cultures.",
                "Explore the concept of time from a mayfly's perspective.",
                "Contemplate what happens to ideas that are never shared.",
                "Discuss the philosophy of impermanence through everyday objects.",
                "Explore the connection between individual creativity and collective consciousness.",
                "Contemplate the nature of existence from the viewpoint of a mirror.",
            ],
            "experimental": [
                "Write a piece where each sentence has a different emotional temperature.",
                "Create text where the rhythm mimics the heartbeat of different emotions.",
                "Write using only questions, but tell a complete story.",
                "Create a piece where punctuation carries the main narrative.",
                "Write a story that reads differently forwards and backwards.",
                "Create text that represents the internal dialogue of a color.",
                "Write a piece using synesthesia - describing sounds as colors, etc.",
                "Create a narrative told entirely through lists of seemingly unrelated items.",
            ]
        }
    
    def generate_creative_prompts(self, num_samples: int) -> List[str]:
        """
        Generate diverse creative writing prompts.
        
        Args:
            num_samples: Number of prompts to generate
            
        Returns:
            List of creative writing prompts
        """
        prompts = []
        categories = list(self.prompt_categories.keys())
        
        for i in range(num_samples):
            # Select category with some randomness but ensure diversity
            category = categories[i % len(categories)]
            if random.random() < 0.3:  # 30% chance to pick random category
                category = random.choice(categories)
            
            # Select prompt from category
            base_prompt = random.choice(self.prompt_categories[category])
            
            # Add creative variations
            variations = [
                f"In exactly three paragraphs, {base_prompt.lower()}",
                f"Using vivid imagery and metaphors, {base_prompt.lower()}",
                f"With surprising word choices and unique perspectives, {base_prompt.lower()}",
                f"Employing varied sentence structures and rhythm, {base_prompt.lower()}",
                f"Through creative language and original insights, {base_prompt.lower()}",
                base_prompt,  # Original prompt
            ]
            
            # Select variation with weighted preference for more creative instructions
            weights = [0.2, 0.2, 0.2, 0.15, 0.15, 0.1]
            prompt = np.random.choice(variations, p=weights)
            prompts.append(prompt)
            
        return prompts
    
    def create_dataset(self, prompts: List[str], split_type: str = "train") -> Dataset:
        """
        Create a HuggingFace dataset from prompts.
        
        Args:
            prompts: List of creative writing prompts
            split_type: Type of split (train/eval)
            
        Returns:
            Dataset ready for RLVR training
        """
        dataset_entries = []
        
        for i, prompt in enumerate(prompts):
            entry = {
                "question": prompt,
                "info": {
                    "prompt_id": f"{split_type}_{i}",
                    "category": self._classify_prompt(prompt),
                    "expected_length": "medium",  # Could be made dynamic
                    "creativity_aspects": ["originality", "diversity", "expression"]
                },
                "task": "creativity-enhancement",
            }
            dataset_entries.append(entry)
        
        return Dataset.from_list(dataset_entries)
    
    def _classify_prompt(self, prompt: str) -> str:
        """Classify prompt into category based on content."""
        prompt_lower = prompt.lower()
        
        for category, prompts in self.prompt_categories.items():
            if any(p.lower() in prompt_lower or prompt_lower in p.lower() for p in prompts):
                return category
        
        return "general"
    
    def creativity_reward_function(self, completion: Any, info: Dict = None, **kwargs) -> float:
        """
        Wrapper for the reward function that integrates with verifiers framework.
        
        Args:
            completion: Model completion (can be string or message format)
            info: Additional information about the prompt
            **kwargs: Additional arguments
            
        Returns:
            Creativity reward score (0.0 to 1.0+ range)
        """
        # Extract text from completion
        if isinstance(completion, str):
            text = completion
        elif isinstance(completion, list) and len(completion) > 0:
            # Handle message format [{"role": "assistant", "content": "text"}]
            text = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
        else:
            return 0.0
        
        # Skip very short responses
        if len(text.strip()) < 20:
            return 0.0
        
        try:
            # Calculate creativity reward using the existing reward function
            reward = reward_function(text, **self.reward_weights)
            
            # Normalize reward to reasonable range (typically 0-10, normalize to 0-1)
            normalized_reward = min(1.0, max(0.0, reward / 10.0))
            
            return float(normalized_reward)
            
        except Exception as e:
            print(f"Error calculating creativity reward: {e}")
            return 0.0
    
    def get_format_reward_function(self) -> callable:
        """
        Get a format reward function that encourages proper response structure.
        
        Returns:
            Function that returns 1.0 for well-formatted responses
        """
        def format_reward_func(completion: Any, **kwargs) -> float:
            # Extract text from completion
            if isinstance(completion, str):
                text = completion
            elif isinstance(completion, list) and len(completion) > 0:
                text = completion[-1].get("content", "") if isinstance(completion[-1], dict) else str(completion[-1])
            else:
                return 0.0
            
            # Check basic formatting requirements
            text = text.strip()
            
            # Minimum length requirement
            if len(text) < 50:
                return 0.3
            
            # Has multiple sentences
            sentences = text.split('.')
            if len(sentences) < 2:
                return 0.5
            
            # Has some structure (paragraphs or varied punctuation)
            has_structure = '\n' in text or '?' in text or '!' in text or ';' in text
            
            return 1.0 if has_structure else 0.8
        
        return format_reward_func


def load_creativity_environment(
    num_train_samples: int = 2000,
    num_eval_samples: int = 200,
    reward_weights: Optional[Dict[str, float]] = None,
    **kwargs
) -> Environment:
    """
    Load the creativity environment for RLVR training.
    
    Args:
        num_train_samples: Number of training samples
        num_eval_samples: Number of evaluation samples  
        reward_weights: Custom weights for reward components
        **kwargs: Additional arguments
        
    Returns:
        Configured verifiers Environment for creativity training
    """
    # Initialize the loader
    loader = CreativityEnvironmentLoader(
        num_train_samples=num_train_samples,
        num_eval_samples=num_eval_samples,
        reward_weights=reward_weights,
        **kwargs
    )
    
    # Generate datasets
    train_prompts = loader.generate_creative_prompts(num_train_samples)
    eval_prompts = loader.generate_creative_prompts(num_eval_samples)
    
    train_dataset = loader.create_dataset(train_prompts, "train")
    eval_dataset = loader.create_dataset(eval_prompts, "eval")
    
    # Create parser for handling responses
    parser = vf.Parser()  # Basic parser since we handle text directly
    
    # Create rubric with creativity and format rewards
    rubric = vf.Rubric(
        funcs=[
            loader.creativity_reward_function,
            loader.get_format_reward_function(),
        ],
        weights=[1.0, 0.2],  # Creativity is primary, format is secondary
    )
    
    # System prompt to encourage creative writing
    system_prompt = """You are a creative writer tasked with producing original, diverse, and expressive text. 
    
Focus on:
- Using varied vocabulary and unique word choices
- Creating diverse sentence structures and lengths
- Employing different punctuation and sentence endings
- Generating original ideas and unexpected connections
- Building rich imagery and creative expressions
- Maintaining engaging rhythm and flow

Write thoughtfully and creatively in response to each prompt."""

    # Create and return the environment
    vf_env = vf.SingleTurnEnv(
        dataset=train_dataset,
        eval_dataset=eval_dataset,
        system_prompt=system_prompt,
        parser=parser,
        rubric=rubric,
        max_concurrent=10,  # Allow parallel processing
    )
    
    return vf_env