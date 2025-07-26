"""
Second Occurrence Masking Environment Loader

This module implements the core logic for the Second-Occurrence Masking task, where:
1. Text is processed to identify words that appear multiple times
2. All occurrences after the first are replaced with [MASK]
3. The agent must predict the original words that were masked

Example:
    Original: "The cat chased the cat and the dog."
    Masked: "The cat chased the [MASK] and [MASK] dog."
    Agent task: Fill in the masks with "cat" and "the"
"""

import re
import random
from typing import List, Tuple, Dict, Set
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class MaskingResult:
    """Result of the masking process"""
    original_text: str
    masked_text: str
    mask_positions: List[Tuple[int, str]]  # (position, original_word)
    target_words: List[str]
    metadata: Dict


class SecondOccurrenceMaskingLoader:
    """
    Handles loading and processing text for second occurrence masking tasks.
    """
    
    def __init__(
        self,
        dataset_name: str = "roneneldan/TinyStories",
        min_length: int = 50,
        max_length: int = 300,
        min_masks: int = 1,
        max_masks: int = 10,
        content_words_only: bool = True,
        exclude_patterns: List[str] = None,
        seed: int = 42
    ):
        """
        Initialize the SecondOccurrenceMaskingLoader.
        
        Args:
            dataset_name: HuggingFace dataset to use for text sources
            min_length: Minimum text length in characters
            max_length: Maximum text length in characters  
            min_masks: Minimum number of masks to create
            max_masks: Maximum number of masks to create
            content_words_only: If True, only mask content words (avoid articles, prepositions)
            exclude_patterns: Regex patterns for words to never mask
            seed: Random seed for reproducibility
        """
        self.dataset_name = dataset_name
        self.min_length = min_length
        self.max_length = max_length
        self.min_masks = min_masks
        self.max_masks = max_masks
        self.content_words_only = content_words_only
        self.exclude_patterns = exclude_patterns or []
        self.seed = seed
        
        # Common function words to avoid masking if content_words_only=True
        self.function_words = {
            'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these',
            'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him',
            'her', 'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their'
        }
        
        random.seed(seed)
        self._dataset = None
        
    def _load_dataset(self):
        """Load the dataset if not already loaded."""
        if self._dataset is None:
            try:
                # Try to load the specified dataset
                self._dataset = load_dataset(self.dataset_name, split="train", streaming=True)
            except Exception:
                # Fallback to TinyStories if the specified dataset fails
                print(f"Failed to load {self.dataset_name}, falling back to TinyStories")
                self._dataset = load_dataset("roneneldan/TinyStories", split="train", streaming=True)
    
    def _is_content_word(self, word: str) -> bool:
        """
        Determine if a word is a content word (noun, verb, adjective, adverb).
        Returns False for function words if content_words_only is True.
        """
        if not self.content_words_only:
            return True
            
        word_lower = word.lower()
        
        # Exclude function words
        if word_lower in self.function_words:
            return False
            
        # Exclude very short words (often function words)
        if len(word_lower) <= 2:
            return False
            
        # Check against exclude patterns
        for pattern in self.exclude_patterns:
            if re.match(pattern, word_lower):
                return False
                
        return True
    
    def _tokenize_text(self, text: str) -> List[Tuple[str, int, int]]:
        """
        Tokenize text into words with their start and end positions.
        Returns list of (word, start_pos, end_pos) tuples.
        """
        # Use regex to find word boundaries, preserving punctuation context
        word_pattern = r'\b\w+\b'
        tokens = []
        
        for match in re.finditer(word_pattern, text):
            word = match.group()
            start_pos = match.start()
            end_pos = match.end()
            tokens.append((word, start_pos, end_pos))
            
        return tokens
    
    def _find_repeated_words(self, tokens: List[Tuple[str, int, int]]) -> Dict[str, List[Tuple[int, int, int]]]:
        """
        Find words that appear multiple times in the text.
        Returns dict mapping word -> list of (occurrence_index, start_pos, end_pos).
        """
        word_occurrences = {}
        
        for i, (word, start_pos, end_pos) in enumerate(tokens):
            word_lower = word.lower()
            
            # Only consider content words if specified
            if not self._is_content_word(word):
                continue
                
            if word_lower not in word_occurrences:
                word_occurrences[word_lower] = []
            word_occurrences[word_lower].append((i, start_pos, end_pos, word))
        
        # Filter to only words that appear multiple times
        repeated_words = {
            word: occurrences 
            for word, occurrences in word_occurrences.items() 
            if len(occurrences) > 1
        }
        
        return repeated_words
    
    def _select_words_to_mask(self, repeated_words: Dict[str, List[Tuple[int, int, int, str]]]) -> Dict[str, List[Tuple[int, int, int, str]]]:
        """
        Select which repeated words to mask, respecting min/max mask constraints.
        """
        if not repeated_words:
            return {}
        
        # Calculate total possible masks (all non-first occurrences)
        total_possible_masks = sum(len(occurrences) - 1 for occurrences in repeated_words.values())
        
        if total_possible_masks < self.min_masks:
            # Not enough repeated words, return all
            return repeated_words
        
        # If we have more potential masks than max_masks, randomly select words
        words_list = list(repeated_words.keys())
        random.shuffle(words_list)
        
        selected_words = {}
        total_masks = 0
        
        for word in words_list:
            occurrences = repeated_words[word]
            masks_from_this_word = len(occurrences) - 1  # All but first occurrence
            
            if total_masks + masks_from_this_word <= self.max_masks:
                selected_words[word] = occurrences
                total_masks += masks_from_this_word
            elif total_masks < self.max_masks:
                # Partially include this word to reach max_masks
                remaining_masks = self.max_masks - total_masks
                # Keep first occurrence + enough others to reach the limit
                selected_occurrences = occurrences[:1 + remaining_masks]
                selected_words[word] = selected_occurrences
                total_masks = self.max_masks
                break
        
        return selected_words
    
    def mask_text(self, text: str) -> MaskingResult:
        """
        Apply second occurrence masking to the given text.
        
        Args:
            text: Input text to mask
            
        Returns:
            MaskingResult containing masked text and metadata
        """
        # Tokenize the text
        tokens = self._tokenize_text(text)
        
        # Find repeated words
        repeated_words = self._find_repeated_words(tokens)
        
        # Select words to mask
        words_to_mask = self._select_words_to_mask(repeated_words)
        
        if not words_to_mask:
            # No repeated words found, return original text
            return MaskingResult(
                original_text=text,
                masked_text=text,
                mask_positions=[],
                target_words=[],
                metadata={
                    "num_tokens": len(tokens),
                    "num_repeated_words": 0,
                    "num_masks": 0
                }
            )
        
        # Create mask positions (skip first occurrence of each word)
        mask_positions = []
        target_words = []
        
        for word, occurrences in words_to_mask.items():
            # Skip the first occurrence, mask the rest
            for i, (token_idx, start_pos, end_pos, original_word) in enumerate(occurrences[1:], 1):
                mask_positions.append((start_pos, end_pos, original_word))
                target_words.append(original_word)
        
        # Sort mask positions by start position (descending) to replace from end to beginning
        mask_positions.sort(key=lambda x: x[0], reverse=True)
        
        # Apply masks to text
        masked_text = text
        final_mask_positions = []
        
        for start_pos, end_pos, original_word in mask_positions:
            masked_text = masked_text[:start_pos] + "[MASK]" + masked_text[end_pos:]
            final_mask_positions.append((start_pos, original_word))
        
        # Reverse the list to get original order
        final_mask_positions.reverse()
        target_words.reverse()
        
        return MaskingResult(
            original_text=text,
            masked_text=masked_text,
            mask_positions=final_mask_positions,
            target_words=target_words,
            metadata={
                "num_tokens": len(tokens),
                "num_repeated_words": len(repeated_words),
                "num_masks": len(target_words),
                "words_masked": list(words_to_mask.keys())
            }
        )
    
    def get_sample_text(self) -> str:
        """
        Get a sample text from the dataset that meets length requirements.
        """
        self._load_dataset()
        
        for example in self._dataset:
            # Extract text from the example (adjust field name based on dataset)
            if "text" in example:
                text = example["text"]
            elif "story" in example:
                text = example["story"]
            else:
                # Try the first string field
                text = next((v for v in example.values() if isinstance(v, str)), "")
            
            if not text:
                continue
                
            # Check length requirements
            if self.min_length <= len(text) <= self.max_length:
                return text.strip()
        
        # Fallback: create a simple example
        return "The cat chased the mouse. The mouse ran away from the cat. The dog watched the cat and the mouse play."
    
    def calculate_reward(self, original_text: str, response: str, mask_positions: List[Tuple[int, str]], target_words: List[str]) -> float:
        """
        Calculate reward based on how many masks were correctly filled.
        
        Args:
            original_text: Original unmasked text
            response: Agent's response 
            mask_positions: List of (position, original_word) tuples
            target_words: List of target words that should fill the masks
            
        Returns:
            Reward score between 0.0 and 1.0
        """
        if not target_words:
            return 1.0  # Perfect score if no masks to fill
        
        # Extract words from response - look for any words that match targets
        response_words = re.findall(r'\b\w+\b', response.lower())
        target_words_lower = [word.lower() for word in target_words]
        
        # Count correct matches
        correct_matches = 0
        used_response_words = set()
        
        for target_word in target_words_lower:
            # Find the first unused occurrence of this target word in response
            for i, response_word in enumerate(response_words):
                if response_word == target_word and i not in used_response_words:
                    correct_matches += 1
                    used_response_words.add(i)
                    break
        
        # Calculate reward as fraction of correct masks
        reward = correct_matches / len(target_words)
        return min(1.0, max(0.0, reward))


def create_sample_examples():
    """Create some sample examples for testing."""
    loader = SecondOccurrenceMaskingLoader(min_length=30, max_length=150)
    
    examples = [
        "The cat chased the cat and the dog watched.",
        "She opened the door and walked through the door into the garden.",
        "The student studied hard for the test. The test was difficult but the student passed.",
        "In the morning, the birds sing. At night, the birds sleep in their nests.",
        "The book was interesting. I read the book twice because the book had many surprising twists."
    ]
    
    results = []
    for text in examples:
        result = loader.mask_text(text)
        results.append(result)
        
    return results


if __name__ == "__main__":
    # Demo the masking functionality
    loader = SecondOccurrenceMaskingLoader()
    
    print("=== Second Occurrence Masking Demo ===\n")
    
    # Test with sample examples
    examples = create_sample_examples()
    
    for i, result in enumerate(examples, 1):
        print(f"Example {i}:")
        print(f"Original: {result.original_text}")
        print(f"Masked:   {result.masked_text}")
        print(f"Targets:  {result.target_words}")
        print(f"Metadata: {result.metadata}")
        print()
    
    # Test reward calculation
    print("=== Reward Calculation Demo ===")
    result = examples[0]  # Use first example
    test_responses = [
        "cat dog",  # Correct
        "cat",      # Partially correct
        "dog cat",  # Correct but wrong order (still gets full credit)
        "mouse",    # Wrong
        ""          # Empty
    ]
    
    for response in test_responses:
        reward = loader.calculate_reward(
            result.original_text, 
            response, 
            result.mask_positions, 
            result.target_words
        )
        print(f"Response: '{response}' -> Reward: {reward:.2f}")