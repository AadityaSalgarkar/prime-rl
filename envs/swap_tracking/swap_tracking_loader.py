"""
Swap tracking data loader for the swap tracking box prediction environment.
Generates tasks where an agent must track box positions through a sequence of swaps.
"""

import random
from typing import Dict, List, Tuple, Optional


class SwapTrackingLoader:
    """Generates swap tracking tasks for RLVR training."""
    
    # Mapping numbers to words for 1–10
    NUM_TO_WORD = {
        1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
        6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
    }
    
    def __init__(self, n_boxes: int = 10, n_swaps: int = 20):
        """
        Initialize the swap tracking loader.
        
        Args:
            n_boxes: Number of boxes (default: 10)
            n_swaps: Number of swaps per task (default: 20)
        """
        self.n_boxes = n_boxes
        self.n_swaps = n_swaps
    
    def generate_swap_task(self, seed: Optional[int] = None) -> Tuple[str, List[Tuple[int, int]], List[int]]:
        """
        Generates a swap-based box prediction task.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of:
            - instruction_text: Description of swaps in words
            - swaps: List of (x, y) index pairs that were swapped
            - final_state: Final arrangement of boxes
        """
        if seed is not None:
            random.seed(seed)
        
        # Initialize boxes 1..n
        boxes = list(range(1, self.n_boxes + 1))
        swaps = []
        instructions = [f"Boxes are arranged from 1 to n={self.n_boxes}."]
        
        # Perform m random swaps
        for _ in range(self.n_swaps):
            x, y = random.sample(range(1, self.n_boxes + 1), 2)
            boxes[x - 1], boxes[y - 1] = boxes[y - 1], boxes[x - 1]
            swaps.append((x, y))
            instructions.append(
                f"Then the box at location {self.NUM_TO_WORD[x]} is swapped with the box at location {self.NUM_TO_WORD[y]}."
            )
        
        instruction_text = " ".join(instructions)
        return instruction_text, swaps, boxes
    
    def format_question(self, instruction_text: str) -> str:
        """Format the instruction text as a question for the model."""
        return f"{instruction_text} What are the final contents of all {self.n_boxes} boxes? Provide your answer as a list of {self.n_boxes} numbers, such as [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]."
    
    def format_answer(self, final_state: List[int]) -> str:
        """Format the final state as the expected answer format."""
        return str(final_state)
    
    def calculate_reward(self, prediction: str, final_state: List[int]) -> float:
        """
        Calculate reward based on position-wise accuracy.
        
        Args:
            prediction: Model's prediction as string
            final_state: True final arrangement
            
        Returns:
            Reward from 0.0 to 1.0 (fraction of correct positions)
        """
        try:
            # Try to parse the prediction as a list
            import re
            
            # Extract numbers from the prediction
            numbers = re.findall(r'\d+', prediction)
            if len(numbers) != self.n_boxes:
                return 0.0
            
            predicted_state = [int(num) for num in numbers]
            
            # Count exact position matches
            matches = sum(1 for i in range(self.n_boxes) 
                         if predicted_state[i] == final_state[i])
            
            return matches / self.n_boxes
            
        except (ValueError, IndexError):
            return 0.0
    
    def generate_training_examples(self, num_examples: int, seed: Optional[int] = None) -> List[Dict]:
        """
        Generate multiple training examples.
        
        Args:
            num_examples: Number of examples to generate
            seed: Random seed for reproducibility
            
        Returns:
            List of training examples with question, answer, and metadata
        """
        if seed is not None:
            random.seed(seed)
        
        examples = []
        for i in range(num_examples):
            # Use different seed for each example to ensure variety
            task_seed = seed + i if seed is not None else None
            
            instruction_text, swaps, final_state = self.generate_swap_task(task_seed)
            question = self.format_question(instruction_text)
            answer = self.format_answer(final_state)
            
            examples.append({
                "question": question,
                "answer": answer,
                "info": {
                    "instruction_text": instruction_text,
                    "swaps": swaps,
                    "final_state": final_state,
                    "n_boxes": self.n_boxes,
                    "n_swaps": self.n_swaps
                },
                "task": "swap-tracking",
            })
        
        return examples


def generate_swap_task(n=10, m=20, seed=None):
    """
    Standalone function that matches the format from TODOS.md.
    Generates a swap-based box prediction task.
    
    Returns:
      instruction_text: str, description of swaps in words
      swaps: list of tuple, index pairs (x, y) swapped
      final_state: list of int, the final arrangement of boxes
    """
    loader = SwapTrackingLoader(n_boxes=n, n_swaps=m)
    return loader.generate_swap_task(seed=seed)


if __name__ == "__main__":
    # Test the swap tracking loader
    loader = SwapTrackingLoader()
    
    # Generate a test task
    instruction, swap_pairs, final_boxes = loader.generate_swap_task(seed=42)
    print("Instruction:", instruction)
    print("Swap pairs:", swap_pairs)
    print("Final boxes:", final_boxes)
    
    # Test the question formatting
    question = loader.format_question(instruction)
    print("\nFormatted question:", question)
    
    # Test the answer formatting
    answer = loader.format_answer(final_boxes)
    print("Expected answer:", answer)
    
    # Test reward calculation
    test_prediction = str(final_boxes)  # Perfect prediction
    reward = loader.calculate_reward(test_prediction, final_boxes)
    print(f"Reward for perfect prediction: {reward}")
    
    # Test reward calculation with imperfect prediction
    wrong_prediction = "[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]"  # Original order
    reward = loader.calculate_reward(wrong_prediction, final_boxes)
    print(f"Reward for original order: {reward}")
    
    # Generate multiple examples
    examples = loader.generate_training_examples(3, seed=42)
    print(f"\nGenerated {len(examples)} training examples")
    for i, example in enumerate(examples):
        print(f"Example {i+1}: {len(example['info']['swaps'])} swaps")