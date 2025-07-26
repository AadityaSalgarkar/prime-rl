"""
Mock inference implementation for swap tracking environment.
Provides different mock models for testing without requiring external services.
"""

import random
import re
from typing import Dict, List, Optional


class MockSwapTrackingModel:
    """Mock model that can simulate different behaviors for swap tracking tasks."""
    
    def __init__(self, mode: str = "identity", n_boxes: int = 10):
        """
        Initialize mock model.
        
        Args:
            mode: Mock behavior mode
                  - "identity": Return original order (for testing)
                  - "simple_completion": Basic text completion
                  - "swap_aware": Attempt simple swap tracking
                  - "random": Random box arrangement
            n_boxes: Number of boxes to work with
        """
        self.mode = mode
        self.n_boxes = n_boxes
    
    def parse_swaps_from_instruction(self, instruction: str) -> List[tuple]:
        """Parse swap operations from instruction text."""
        # Extract swap instructions using regex
        swap_pattern = r"box at location (\w+) is swapped with the box at location (\w+)"
        matches = re.findall(swap_pattern, instruction)
        
        # Convert word numbers to integers
        word_to_num = {
            "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        
        swaps = []
        for word1, word2 in matches:
            if word1 in word_to_num and word2 in word_to_num:
                swaps.append((word_to_num[word1], word_to_num[word2]))
        
        return swaps
    
    def simulate_swaps(self, swaps: List[tuple]) -> List[int]:
        """Simulate swaps to get final arrangement."""
        # Start with boxes in original order
        boxes = list(range(1, self.n_boxes + 1))
        
        # Apply each swap
        for pos1, pos2 in swaps:
            if 1 <= pos1 <= self.n_boxes and 1 <= pos2 <= self.n_boxes:
                boxes[pos1 - 1], boxes[pos2 - 1] = boxes[pos2 - 1], boxes[pos1 - 1]
        
        return boxes
    
    def extract_n_boxes_from_message(self, message: str) -> int:
        """Extract the number of boxes from the message if specified."""
        import re
        # Look for patterns like "n=5" or "5 boxes"
        patterns = [
            r"n=(\d+)",
            r"(\d+)\s+boxes",
            r"all\s+(\d+)\s+boxes"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, message)
            if match:
                return int(match.group(1))
        
        # Default to instance setting
        return self.n_boxes
    
    def generate_response(self, messages: List[Dict]) -> str:
        """Generate response based on mode."""
        if not messages:
            return "[]"
        
        # Get the user message (instruction)
        user_message = ""
        for msg in messages:
            if msg.get("role") == "user":
                user_message = msg.get("content", "")
                break
        
        # Extract number of boxes from the message
        n_boxes = self.extract_n_boxes_from_message(user_message)
        
        if self.mode == "identity":
            # Return original order (useful for testing)
            return str(list(range(1, n_boxes + 1)))
        
        elif self.mode == "simple_completion":
            # Simple text completion - just return a generic response
            return f"The final arrangement is {list(range(1, n_boxes + 1))}."
        
        elif self.mode == "swap_aware":
            # Try to parse and simulate swaps (might make mistakes)
            try:
                swaps = self.parse_swaps_from_instruction(user_message)
                
                # Create a SwapTrackingModel with the correct number of boxes
                temp_model = MockSwapTrackingModel(self.mode, n_boxes)
                
                # Add some noise to simulate model errors
                if random.random() < 0.3:  # 30% chance of error
                    # Random error: skip a swap or apply wrong swap
                    if swaps and random.random() < 0.5:
                        swaps = swaps[:-1]  # Skip last swap
                    else:
                        # Add random swap
                        pos1, pos2 = random.sample(range(1, n_boxes + 1), 2)
                        swaps.append((pos1, pos2))
                
                final_arrangement = temp_model.simulate_swaps(swaps)
                return str(final_arrangement)
                
            except Exception:
                # Fallback to random if parsing fails
                boxes = list(range(1, n_boxes + 1))
                random.shuffle(boxes)
                return str(boxes)
        
        elif self.mode == "random":
            # Return random arrangement
            boxes = list(range(1, n_boxes + 1))
            random.shuffle(boxes)
            return str(boxes)
        
        else:
            # Default to original order
            return str(list(range(1, n_boxes + 1)))


class MockInferenceClient:
    """Mock OpenAI-compatible client for testing."""
    
    def __init__(self, model_name: str = "mock-identity", n_boxes: int = 10):
        """
        Initialize mock client.
        
        Args:
            model_name: Name of mock model (determines behavior)
            n_boxes: Number of boxes for swap tracking
        """
        self.model_name = model_name
        
        # Parse mode from model name
        if "identity" in model_name:
            mode = "identity"
        elif "simple" in model_name:
            mode = "simple_completion"
        elif "swap" in model_name or "aware" in model_name:
            mode = "swap_aware"
        elif "random" in model_name:
            mode = "random"
        else:
            mode = "identity"  # Default
        
        self.model = MockSwapTrackingModel(mode=mode, n_boxes=n_boxes)
    
    def create_completion(self, messages: List[Dict], **kwargs) -> Dict:
        """Create mock completion response."""
        response_text = self.model.generate_response(messages)
        
        # Return OpenAI-compatible response format
        return {
            "choices": [{
                "message": {
                    "content": response_text,
                    "role": "assistant"
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "model": self.model_name,
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 20,
                "total_tokens": 70
            }
        }


def test_mock_models():
    """Test different mock model modes."""
    print("Testing mock inference models...")
    
    # Test instruction
    test_instruction = (
        "Boxes are arranged from 1 to n=5. "
        "Then the box at location one is swapped with the box at location five. "
        "Then the box at location two is swapped with the box at location four. "
        "What are the final contents of all 5 boxes? "
        "Provide your answer as a list of 5 numbers."
    )
    
    messages = [{"role": "user", "content": test_instruction}]
    
    # Expected result: [5, 4, 3, 2, 1] (after swaps: 1↔5, 2↔4)
    
    modes = ["identity", "simple_completion", "swap_aware", "random"]
    
    for mode in modes:
        print(f"\nTesting mode: {mode}")
        model_name = f"mock-{mode}"
        client = MockInferenceClient(model_name, n_boxes=5)
        
        response = client.create_completion(messages)
        result = response["choices"][0]["message"]["content"]
        print(f"Response: {result}")
        
        # Test parsing
        try:
            numbers = re.findall(r'\d+', result)
            if len(numbers) == 5:
                parsed = [int(n) for n in numbers]
                print(f"Parsed: {parsed}")
            else:
                print(f"Warning: Expected 5 numbers, got {len(numbers)}")
        except Exception as e:
            print(f"Parsing error: {e}")


if __name__ == "__main__":
    test_mock_models()