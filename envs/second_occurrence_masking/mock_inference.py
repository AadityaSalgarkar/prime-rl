#!/usr/bin/env python3
"""
Mock inference implementation for Second Occurrence Masking environment.

This module provides mock inference models for development and testing without
requiring external dependencies or GPU resources. Supports multiple modes:

1. Identity mode: Perfect accuracy for testing reward functions
2. Simple completion mode: Basic text completion for general testing  
3. Masking-aware mode: Heuristic-based mask filling for realistic testing

Usage:
    python mock_inference.py --mode identity
    python mock_inference.py --mode masking_aware --accuracy 0.7
"""

import re
import random
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class MockResponse:
    """Mock response matching OpenAI API format."""
    content: str
    finish_reason: str = "stop"


class MockInferenceModel:
    """Base class for mock inference models."""
    
    def __init__(self, mode: str = "identity", accuracy: float = 1.0, seed: int = 42):
        self.mode = mode
        self.accuracy = accuracy
        random.seed(seed)
        
    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> MockResponse:
        """Generate a completion for the given messages."""
        raise NotImplementedError


class IdentityModel(MockInferenceModel):
    """Identity model that returns perfect answers for testing."""
    
    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> MockResponse:
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # For mask filling tasks, extract the expected answer
        if "Fill in the [MASK]" in user_message and ":" in user_message:
            # Extract the text after the colon
            text_part = user_message.split(":", 1)[1].strip()
            
            # For testing, return a plausible answer based on context
            if "cat chased the [MASK]" in text_part:
                return MockResponse("cat")
            elif "opened the door and walked through the [MASK]" in text_part:
                return MockResponse("door")
            elif "[MASK] student" in text_part:
                return MockResponse("The")
            else:
                # Try to find repeated words in the context
                words = re.findall(r'\b\w+\b', text_part.replace("[MASK]", ""))
                word_counts = {}
                for word in words:
                    word_lower = word.lower()
                    word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
                
                # Return the most common word (likely to be correct)
                if word_counts:
                    most_common = max(word_counts.items(), key=lambda x: x[1])
                    if most_common[1] > 1:  # Only if it appears more than once
                        return MockResponse(most_common[0])
        
        # Default: return the input for identity behavior
        return MockResponse(user_message)


class SimpleCompletionModel(MockInferenceModel):
    """Simple completion model for basic testing."""
    
    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> MockResponse:
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Simple responses for different types of queries
        if "hello" in user_message.lower():
            return MockResponse("Hello back!")
        elif "Fill in the [MASK]" in user_message:
            return MockResponse("word")  # Generic response
        elif "?" in user_message:
            return MockResponse("Yes, that's correct.")
        else:
            return MockResponse("I understand your request.")


class MaskingAwareModel(MockInferenceModel):
    """Mask-aware model with heuristic-based responses."""
    
    def __init__(self, mode: str = "masking_aware", accuracy: float = 0.8, seed: int = 42):
        super().__init__(mode, accuracy, seed)
        
        # Common word patterns for different contexts
        self.common_patterns = {
            "the": ["cat", "dog", "house", "door", "book", "car"],
            "a": ["cat", "dog", "house", "door", "book", "car"],
            "and": ["or", "but", "then"],
            "cat": ["dog", "bird", "mouse"],
            "dog": ["cat", "bird", "mouse"],
            "student": ["teacher", "student", "person"],
            "book": ["paper", "notebook", "document"],
            "door": ["window", "gate", "entrance"]
        }
    
    def complete(self, messages: List[Dict[str, Any]], **kwargs) -> MockResponse:
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        if "Fill in the [MASK]" in user_message:
            return self._fill_masks(user_message)
        else:
            return MockResponse("I can help with mask filling tasks.")
    
    def _fill_masks(self, message: str) -> MockResponse:
        """Fill masks using heuristic approaches."""
        try:
            # Extract the text with masks
            if ":" in message:
                text_with_masks = message.split(":", 1)[1].strip()
            else:
                text_with_masks = message
            
            # Count masks
            mask_count = text_with_masks.count("[MASK]")
            if mask_count == 0:
                return MockResponse("No masks found.")
            
            # Find words that appear in the context (likely candidates)
            words_in_context = re.findall(r'\b\w+\b', text_with_masks.replace("[MASK]", ""))
            word_counts = {}
            
            for word in words_in_context:
                word_lower = word.lower()
                word_counts[word_lower] = word_counts.get(word_lower, 0) + 1
            
            # Find words that appear multiple times (good candidates for masking)
            repeated_words = [word for word, count in word_counts.items() if count > 0]
            
            predictions = []
            
            for i in range(mask_count):
                if random.random() < self.accuracy and repeated_words:
                    # Choose a word that likely appeared before
                    if i < len(repeated_words):
                        prediction = repeated_words[i]
                    else:
                        prediction = random.choice(repeated_words)
                    predictions.append(prediction)
                else:
                    # Random/wrong answer
                    wrong_words = ["word", "thing", "item", "stuff", "something"]
                    predictions.append(random.choice(wrong_words))
            
            return MockResponse(" ".join(predictions))
            
        except Exception as e:
            return MockResponse(f"Error processing masks: {e}")


def create_mock_model(mode: str = "identity", accuracy: float = 1.0, seed: int = 42) -> MockInferenceModel:
    """Factory function to create mock models."""
    if mode == "identity":
        return IdentityModel(mode, accuracy, seed)
    elif mode == "simple_completion":
        return SimpleCompletionModel(mode, accuracy, seed)
    elif mode == "masking_aware":
        return MaskingAwareModel(mode, accuracy, seed)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def demo_mock_models():
    """Demonstrate different mock model capabilities."""
    print("🎭 Mock Inference Models Demo\n")
    
    test_messages = [
        [{"role": "user", "content": "Hello! Please respond with 'Hello back!'"}],
        [{"role": "user", "content": "Fill in the [MASK] tokens with the original words: The cat chased the [MASK]."}],
        [{"role": "user", "content": "Fill in the [MASK] tokens with the original words: She opened the door and walked through the [MASK]."}],
        [{"role": "user", "content": "Fill in the [MASK] tokens with the original words: The student studied hard. [MASK] student passed."}],
    ]
    
    modes = ["identity", "simple_completion", "masking_aware"]
    
    for mode in modes:
        print(f"--- {mode.upper()} MODE ---")
        model = create_mock_model(mode, accuracy=0.8)
        
        for i, messages in enumerate(test_messages, 1):
            print(f"Test {i}: {messages[0]['content']}")
            response = model.complete(messages)
            print(f"Response: {response.content}")
            print()
        
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Mock inference for second occurrence masking")
    parser.add_argument("--mode", choices=["identity", "simple_completion", "masking_aware"], 
                       default="identity", help="Mock model mode")
    parser.add_argument("--accuracy", type=float, default=1.0, 
                       help="Simulated accuracy (0.0-1.0)")
    parser.add_argument("--seed", type=int, default=42, 
                       help="Random seed for reproducibility")
    parser.add_argument("--demo", action="store_true", 
                       help="Run demo of all modes")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_mock_models()
    else:
        print(f"🎭 Mock model created: {args.mode} (accuracy={args.accuracy})")
        
        # Interactive test
        model = create_mock_model(args.mode, args.accuracy, args.seed)
        
        while True:
            try:
                user_input = input("\nEnter a message (or 'quit' to exit): ")
                if user_input.lower() in ['quit', 'exit', 'q']:
                    break
                
                messages = [{"role": "user", "content": user_input}]
                response = model.complete(messages)
                print(f"Response: {response.content}")
                
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break