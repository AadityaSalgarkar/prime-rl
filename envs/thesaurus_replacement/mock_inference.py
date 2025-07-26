"""
Mock inference implementation for testing purposes.
Provides an identity model that maps input to output (input == output).
"""

import json
import re
from typing import Dict, List, Optional
from pathlib import Path


class MockInferenceEngine:
    """
    Mock inference engine that returns input as output.
    Useful for testing training pipelines without actual model inference.
    """
    
    def __init__(self, mode: str = "identity"):
        """
        Initialize mock inference engine.
        
        Args:
            mode: Type of mock behavior
                - "identity": Return input as output
                - "simple_completion": Add simple completions
                - "thesaurus_aware": Attempt basic thesaurus restoration
        """
        self.mode = mode
        self._load_thesaurus_data()
    
    def _load_thesaurus_data(self):
        """Load thesaurus data for thesaurus-aware mode."""
        try:
            thesaurus_path = Path(__file__).parent / "en_thesaurus.jsonl"
            self.synonym_to_original = {}
            
            if thesaurus_path.exists():
                with open(thesaurus_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            entry = json.loads(line.strip())
                            word = entry.get("word", "").lower()
                            synonyms = entry.get("synonyms", [])
                            
                            # Create reverse mapping: synonym -> original word
                            for synonym in synonyms:
                                if synonym.lower() != word:
                                    self.synonym_to_original[synonym.lower()] = word
                        except json.JSONDecodeError:
                            continue
        except Exception:
            self.synonym_to_original = {}
    
    def complete(self, prompt: str) -> str:
        """
        Generate completion based on mock mode.
        
        Args:
            prompt: Input prompt
            
        Returns:
            Mock completion
        """
        if self.mode == "identity":
            return self._identity_completion(prompt)
        elif self.mode == "simple_completion":
            return self._simple_completion(prompt)
        elif self.mode == "thesaurus_aware":
            return self._thesaurus_aware_completion(prompt)
        else:
            return prompt
    
    def _identity_completion(self, prompt: str) -> str:
        """Return input as output."""
        return prompt
    
    def _simple_completion(self, prompt: str) -> str:
        """Add simple completions to prompts."""
        if "Complete this sentence:" in prompt:
            # Extract the partial sentence and add a simple ending
            sentence_start = prompt.split("Complete this sentence:")[-1].strip()
            return sentence_start + " and lived happily ever after."
        elif "Restore the original text:" in prompt:
            # Extract the text after "Restore the original text:"
            text_to_restore = prompt.split("Restore the original text:")[-1].strip()
            return text_to_restore  # Return as-is for simple mode
        else:
            return prompt + " [completed]"
    
    def _thesaurus_aware_completion(self, prompt: str) -> str:
        """Attempt basic thesaurus restoration."""
        if "Restore the original text:" in prompt:
            text_to_restore = prompt.split("Restore the original text:")[-1].strip()
            return self._attempt_restoration(text_to_restore)
        else:
            return prompt
    
    def _attempt_restoration(self, text: str) -> str:
        """
        Attempt to restore original text by replacing known synonyms.
        This is a simple heuristic-based approach for testing.
        """
        words = re.findall(r'\b\w+\b', text)
        restored_words = []
        
        for word in words:
            word_lower = word.lower()
            
            # Check if this word is a known synonym
            if word_lower in self.synonym_to_original:
                original = self.synonym_to_original[word_lower]
                
                # Preserve case pattern
                if word.isupper():
                    restored_words.append(original.upper())
                elif word.istitle():
                    restored_words.append(original.capitalize())
                else:
                    restored_words.append(original)
            else:
                restored_words.append(word)
        
        # Reconstruct the text maintaining original structure
        result = text
        word_idx = 0
        for match in re.finditer(r'\b\w+\b', text):
            if word_idx < len(restored_words):
                result = result[:match.start()] + restored_words[word_idx] + result[match.end():]
                word_idx += 1
        
        return result


class MockOpenAICompatibleAPI:
    """
    Mock OpenAI-compatible API for testing.
    Mimics the OpenAI client interface but uses mock inference.
    """
    
    def __init__(self, mode: str = "identity"):
        self.engine = MockInferenceEngine(mode)
        self.chat = self
        self.completions = self
    
    def create(self, model: str, messages: List[Dict], **kwargs) -> 'MockResponse':
        """
        Create a chat completion using mock inference.
        
        Args:
            model: Model name (ignored in mock)
            messages: List of message dicts
            **kwargs: Additional parameters (ignored)
            
        Returns:
            Mock response object
        """
        # Extract the user message content
        user_content = ""
        for message in messages:
            if message.get("role") == "user":
                user_content = message.get("content", "")
                break
        
        # Generate mock completion
        completion = self.engine.complete(user_content)
        
        return MockResponse(completion)


class MockResponse:
    """Mock response object mimicking OpenAI API response structure."""
    
    def __init__(self, content: str):
        self.choices = [MockChoice(content)]


class MockChoice:
    """Mock choice object."""
    
    def __init__(self, content: str):
        self.message = MockMessage(content)


class MockMessage:
    """Mock message object."""
    
    def __init__(self, content: str):
        self.content = content


def create_mock_client(mode: str = "identity") -> MockOpenAICompatibleAPI:
    """
    Factory function to create a mock OpenAI-compatible client.
    
    Args:
        mode: Mock behavior mode
        
    Returns:
        Mock client instance
    """
    return MockOpenAICompatibleAPI(mode)


if __name__ == "__main__":
    # Test the mock inference engine
    print("Testing Mock Inference Engine")
    print("=" * 40)
    
    modes = ["identity", "simple_completion", "thesaurus_aware"]
    test_prompts = [
        "Hello world",
        "Complete this sentence: The cat sat on the",
        "Restore the original text: She unfastened the antique door.",
        "Restore the original text: The unspoiled dog ran fast."
    ]
    
    for mode in modes:
        print(f"\n📝 Testing mode: {mode}")
        engine = MockInferenceEngine(mode)
        
        for prompt in test_prompts:
            completion = engine.complete(prompt)
            print(f"  Input:  {prompt}")
            print(f"  Output: {completion}")
            print()
    
    # Test OpenAI-compatible API
    print("\n🔌 Testing Mock OpenAI API")
    client = create_mock_client("thesaurus_aware")
    
    messages = [{"role": "user", "content": "Restore the original text: She unfastened the antique door."}]
    response = client.create(model="mock", messages=messages)
    
    print(f"API Input: {messages[0]['content']}")
    print(f"API Output: {response.choices[0].message.content}")