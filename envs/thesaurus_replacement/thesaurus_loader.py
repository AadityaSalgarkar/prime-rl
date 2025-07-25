"""
Thesaurus data loader for the thesaurus replacement environment.
Loads and processes the en_thesaurus.jsonl data.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Set


class ThesaurusLoader:
    """Loads and manages thesaurus data for synonym replacement."""
    
    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = Path(__file__).parent / "en_thesaurus.jsonl"
        
        self.data_path = Path(data_path)
        self.word_to_synonyms: Dict[str, List[str]] = {}
        self._load_data()
    
    def _load_data(self):
        """Load thesaurus data from JSONL file."""
        if not self.data_path.exists():
            raise FileNotFoundError(f"Thesaurus data not found at {self.data_path}")
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    entry = json.loads(line.strip())
                    word = entry.get("word", "").lower()
                    synonyms = entry.get("synonyms", [])
                    
                    # Only include words with at least 2 synonyms for meaningful replacement
                    if word and len(synonyms) >= 2:
                        # Clean synonyms (remove duplicates, convert to lowercase)
                        clean_synonyms = []
                        seen = set()
                        for syn in synonyms:
                            syn_clean = syn.lower().strip()
                            if syn_clean and syn_clean != word and syn_clean not in seen:
                                clean_synonyms.append(syn_clean)
                                seen.add(syn_clean)
                        
                        if len(clean_synonyms) >= 2:
                            self.word_to_synonyms[word] = clean_synonyms
                            
                except (json.JSONDecodeError, KeyError):
                    continue
    
    def get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word."""
        return self.word_to_synonyms.get(word.lower(), [])
    
    def has_synonyms(self, word: str) -> bool:
        """Check if a word has available synonyms."""
        return word.lower() in self.word_to_synonyms
    
    def get_random_synonym(self, word: str) -> Optional[str]:
        """Get a random synonym for a word."""
        synonyms = self.get_synonyms(word)
        if not synonyms:
            return None
        return random.choice(synonyms)
    
    def get_replaceable_words(self, text: str) -> List[str]:
        """Get list of words in text that have available synonyms."""
        import re
        
        # Simple word tokenization (can be improved)
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [word for word in words if self.has_synonyms(word)]
    
    def replace_with_synonyms(self, text: str, replacement_rate: float = 0.3) -> tuple[str, Dict[int, str]]:
        """
        Replace words in text with synonyms.
        
        Args:
            text: Original text
            replacement_rate: Fraction of replaceable words to replace
            
        Returns:
            Tuple of (modified_text, word_position_to_original_word_mapping)
        """
        import re
        
        # Tokenize while preserving positions and case
        word_pattern = re.compile(r'\b[a-zA-Z]+\b')
        words_with_positions = [(match.group(), match.start(), match.end()) 
                               for match in word_pattern.finditer(text)]
        
        # Find replaceable words
        replaceable_indices = []
        for i, (word, start, end) in enumerate(words_with_positions):
            if self.has_synonyms(word.lower()):
                replaceable_indices.append(i)
        
        # Select subset to replace
        num_to_replace = max(1, int(len(replaceable_indices) * replacement_rate))
        if replaceable_indices:
            indices_to_replace = random.sample(replaceable_indices, 
                                             min(num_to_replace, len(replaceable_indices)))
        else:
            indices_to_replace = []
        
        # Build modified text and mapping
        modified_text = text
        replacements = {}  # position -> original_word
        offset = 0
        
        for i in sorted(indices_to_replace):
            original_word, start, end = words_with_positions[i]
            synonym = self.get_random_synonym(original_word.lower())
            
            if synonym:
                # Preserve original case pattern
                if original_word.isupper():
                    replacement = synonym.upper()
                elif original_word.istitle():
                    replacement = synonym.capitalize()
                else:
                    replacement = synonym
                
                # Apply replacement
                actual_start = start + offset
                actual_end = end + offset
                modified_text = (modified_text[:actual_start] + 
                               replacement + 
                               modified_text[actual_end:])
                
                # Track replacement for reward calculation
                replacements[i] = original_word
                
                # Update offset for subsequent replacements
                offset += len(replacement) - len(original_word)
        
        return modified_text, replacements


def load_thesaurus(data_path: Optional[str] = None) -> ThesaurusLoader:
    """Factory function to load thesaurus data."""
    return ThesaurusLoader(data_path)


if __name__ == "__main__":
    # Test the thesaurus loader
    loader = load_thesaurus()
    
    test_sentence = "She opened the ancient door."
    print(f"Original: {test_sentence}")
    
    modified, replacements = loader.replace_with_synonyms(test_sentence, replacement_rate=0.5)
    print(f"Modified: {modified}")
    print(f"Replacements: {replacements}")
    
    # Test specific words
    test_words = ["opened", "ancient", "door", "she"]
    for word in test_words:
        synonyms = loader.get_synonyms(word)
        print(f"{word}: {synonyms[:5]}")  # Show first 5 synonyms