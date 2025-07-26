Each environment presents the agent with an augmented (noisy) version of a text segment, and the agent’s action is always “output the original, uncorrupted text.” 

- Thesaurus‐Replacement
Augmentation: Randomly swap X% of words with near-synonyms. Use this for generating synonyms : https://github.com/zaibacu/thesaurus/tree/master
State: Sentence with synonyms substituted.
Action: Reconstruct the exact original words.
Reward: +1 for each word matching the original.
Example:
	•	Original: “She opened the ancient door.”
	•	Augmented: “She unfastened the antique door.”
	•	Agent must output: “She opened the ancient door.”

- Second‐Occurrence Masking

Augmentation: Leave only the first instance of a target word; mask all subsequent occurrences. Use a dataset with long paragraphs like long descriptions in AndyReas/frontpage-news, or roneneldan/TinyStories
State: Text where later occurrences are replaced with [MASK].
Action: Fill each [MASK] with the correct original word.
Reward: +1 per mask correctly filled.
Example:
	•	Original: “The cat chased the cat.”
	•	Augmented: “The cat chased the [MASK].”
	•	Agent must output: “cat.”


- Swap‐Tracking Box Prediction
Augmentation/Env: Start with boxes numbered 1…10 in order. Then perform m = 20 random swaps, recording each swapped pair of positions. Present the full sequence of swaps (with positions written as words) to the agent.
State: A single instruction string, for example:

“Boxes are arranged from 1 to n=10. Then the box at location three is swapped with the box at location seven. Then the box at location one is swapped with the box at location ten. …”
Action: Predict the final contents of all 10 boxes, i.e. output the box numbers in positions 1 through 10.
Reward: The number of positions where the agent’s prediction matches the true final arrangement (0–10).
Example:

	•	Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	•	Swaps (sample): (3 ↔ 7), (1 ↔ 10), … 20 total
	•	Final state: [10, 2, 7, 4, 5, 6, 3, 8, 9, 1]
	•	Agent’s guess: [10, 2, 7, 4, 5, 6, 3, 8, 9, 1] → Reward = 10

Below is Python code to generate such tasks (swap log + final arrangement).
```
import random

# Mapping numbers to words for 1–10
num2word = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"
}

def generate_swap_task(n=10, m=20, seed=None):
    """
    Generates a swap-based box prediction task.
    
    Returns:
      instruction_text: str, description of swaps in words
      swaps: list of tuple, index pairs (x, y) swapped
      final_state: list of int, the final arrangement of boxes
    """
    if seed is not None:
        random.seed(seed)
    
    # Initialize boxes 1..n
    boxes = list(range(1, n + 1))
    swaps = []
    instructions = [f"Boxes are arranged from 1 to n={n}."]
    
    # Perform m random swaps
    for _ in range(m):
        x, y = random.sample(range(1, n + 1), 2)
        boxes[x - 1], boxes[y - 1] = boxes[y - 1], boxes[x - 1]
        swaps.append((x, y))
        instructions.append(
            f"Then the box at location {num2word[x]} is swapped with the box at location {num2word[y]}."
        )
    
    instruction_text = " ".join(instructions)
    return instruction_text, swaps, boxes

# Example usage
if __name__ == "__main__":
    instr, swap_pairs, final_boxes = generate_swap_task(seed=42)
    print(instr)
    print("Swap pairs:", swap_pairs)
    print("Final boxes:", final_boxes)
```

