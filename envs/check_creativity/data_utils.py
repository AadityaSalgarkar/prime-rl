#!/usr/bin/env python3
"""
Data utilities for creativity environment

This module provides advanced data generation, loading, and augmentation 
capabilities for the creativity RLVR environment.
"""

import json
import random
import re
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path


class CreativeTextGenerator:
    """
    Advanced text generator for creativity training data.
    
    Provides multiple strategies for generating diverse, creative text samples
    that can be used for training and evaluation.
    """
    
    def __init__(self, seed: int = 42):
        """Initialize the generator with creative templates and patterns."""
        random.seed(seed)
        np.random.seed(seed)
        
        self.word_pools = self._init_word_pools()
        self.sentence_starters = self._init_sentence_starters()
        self.creative_techniques = self._init_creative_techniques()
    
    def _init_word_pools(self) -> Dict[str, List[str]]:
        """Initialize diverse word pools for creative text generation."""
        return {
            "unusual_adjectives": [
                "iridescent", "ephemeral", "labyrinthine", "gossamer", "incandescent",
                "mercurial", "pristine", "ethereal", "luminous", "whimsical",
                "enigmatic", "kaleidoscopic", "serendipitous", "mellifluous", "crystalline",
                "opalescent", "gossamer", "diaphanous", "scintillating", "resplendent"
            ],
            "abstract_nouns": [
                "reverie", "solitude", "serendipity", "wanderlust", "nostalgia",
                "melancholy", "euphoria", "tranquility", "epiphany", "resonance",
                "convergence", "metamorphosis", "synchronicity", "luminescence", "cadence",
                "tessellation", "juxtaposition", "cacophony", "harmony", "dissonance"
            ],
            "sensory_verbs": [
                "whisper", "shimmer", "cascade", "undulate", "permeate", "resonate",
                "emanate", "glisten", "pulsate", "oscillate", "radiate", "diffuse",
                "illuminate", "envelop", "caress", "embrace", "punctuate", "infuse"
            ],
            "uncommon_words": [
                "susurrus", "petrichor", "sonder", "lacuna", "hiraeth", "saudade",
                "schadenfreude", "fernweh", "forelsket", "ubuntu", "mamihlapinatapai",
                "tsundoku", "komorebi", "wabi-sabi", "hygge", "gezelligheid"
            ]
        }
    
    def _init_sentence_starters(self) -> List[str]:
        """Initialize creative sentence starters to promote variety."""
        return [
            "In the liminal space between",
            "Where shadows dance with",
            "Through the gossamer veil of",
            "Beneath the surface of",
            "In the symphony of",
            "Where time itself seems to",
            "Through corridors of",
            "In the gentle embrace of",
            "Where whispers become",
            "Across the landscape of",
            "In the delicate balance of",
            "Through the prism of",
            "Where silence speaks",
            "In the convergence of",
            "Through the labyrinth of"
        ]
    
    def _init_creative_techniques(self) -> Dict[str, callable]:
        """Initialize different creative writing techniques."""
        return {
            "synesthesia": self._apply_synesthesia,
            "metaphorical": self._apply_metaphors,
            "stream_of_consciousness": self._apply_stream_consciousness,
            "minimalist": self._apply_minimalist_style,
            "baroque": self._apply_baroque_style,
            "experimental_punctuation": self._apply_experimental_punctuation,
            "repetition_variation": self._apply_repetition_variation,
        }
    
    def generate_creative_sample(
        self, 
        prompt: str, 
        length: str = "medium",
        technique: Optional[str] = None
    ) -> str:
        """
        Generate a creative text sample based on a prompt.
        
        Args:
            prompt: Base prompt to inspire the text
            length: Desired length (short, medium, long)
            technique: Specific creative technique to apply
            
        Returns:
            Generated creative text
        """
        # Determine target word count based on length
        length_targets = {
            "short": (30, 80),
            "medium": (80, 200), 
            "long": (200, 400)
        }
        min_words, max_words = length_targets.get(length, (80, 200))
        target_words = random.randint(min_words, max_words)
        
        # Select creative technique
        if technique is None:
            technique = random.choice(list(self.creative_techniques.keys()))
        
        # Generate base text
        base_text = self._generate_base_creative_text(prompt, target_words)
        
        # Apply creative technique
        if technique in self.creative_techniques:
            creative_text = self.creative_techniques[technique](base_text)
        else:
            creative_text = base_text
        
        return creative_text
    
    def _generate_base_creative_text(self, prompt: str, target_words: int) -> str:
        """Generate base creative text responding to prompt."""
        words_generated = 0
        sentences = []
        
        # Extract key concepts from prompt
        key_concepts = self._extract_concepts(prompt)
        
        while words_generated < target_words:
            # Create sentence with varied structure
            sentence = self._create_creative_sentence(key_concepts)
            sentences.append(sentence)
            words_generated += len(sentence.split())
            
            # Add variety with different sentence types
            if random.random() < 0.2 and words_generated < target_words * 0.8:
                question = self._create_creative_question(key_concepts)
                sentences.append(question)
                words_generated += len(question.split())
        
        return " ".join(sentences)
    
    def _extract_concepts(self, prompt: str) -> List[str]:
        """Extract key concepts from prompt for creative development."""
        # Simple concept extraction - in practice could use NLP
        words = re.findall(r'\b\w+\b', prompt.lower())
        concepts = [w for w in words if len(w) > 4 and w not in 
                   {'about', 'where', 'write', 'create', 'describe', 'story', 'poem'}]
        return concepts[:5]  # Limit to avoid repetition
    
    def _create_creative_sentence(self, concepts: List[str]) -> str:
        """Create a creative sentence incorporating concepts."""
        # Select sentence structure
        structures = [
            self._create_descriptive_sentence,
            self._create_metaphorical_sentence,
            self._create_sensory_sentence,
            self._create_abstract_sentence,
        ]
        
        structure_func = random.choice(structures)
        return structure_func(concepts)
    
    def _create_descriptive_sentence(self, concepts: List[str]) -> str:
        """Create descriptive sentence with unusual word choices."""
        starter = random.choice(self.sentence_starters)
        adjective = random.choice(self.word_pools["unusual_adjectives"])
        noun = random.choice(self.word_pools["abstract_nouns"])
        verb = random.choice(self.word_pools["sensory_verbs"])
        
        concept = random.choice(concepts) if concepts else "existence"
        
        templates = [
            f"{starter} {adjective} {concept}, there {verb}s a {noun} of understanding.",
            f"The {adjective} nature of {concept} {verb}s through layers of {noun}.",
            f"Within each {adjective} moment of {concept}, {noun} begins to {verb}."
        ]
        
        return random.choice(templates)
    
    def _create_metaphorical_sentence(self, concepts: List[str]) -> str:
        """Create sentence with rich metaphors."""
        concept = random.choice(concepts) if concepts else "time"
        adjective = random.choice(self.word_pools["unusual_adjectives"])
        noun = random.choice(self.word_pools["abstract_nouns"])
        
        templates = [
            f"{concept.title()} is a {adjective} river of {noun}, flowing through dimensions unseen.",
            f"Like {adjective} threads of {noun}, {concept} weaves itself into the fabric of being.",
            f"Each {concept} becomes a {adjective} mirror, reflecting infinite {noun}."
        ]
        
        return random.choice(templates)
    
    def _create_sensory_sentence(self, concepts: List[str]) -> str:
        """Create sentence emphasizing sensory experience."""
        verb = random.choice(self.word_pools["sensory_verbs"])
        adjective = random.choice(self.word_pools["unusual_adjectives"])
        concept = random.choice(concepts) if concepts else "sound"
        
        templates = [
            f"The {concept} {verb}s with {adjective} intensity, touching every sense.",
            f"You can feel how {concept} {verb}s across your consciousness, {adjective} and alive.",
            f"There's something {adjective} in the way {concept} {verb}s through space."
        ]
        
        return random.choice(templates)
    
    def _create_abstract_sentence(self, concepts: List[str]) -> str:
        """Create abstract, philosophical sentence."""
        abstract_noun = random.choice(self.word_pools["abstract_nouns"])
        unusual_adj = random.choice(self.word_pools["unusual_adjectives"])
        concept = random.choice(concepts) if concepts else "meaning"
        
        templates = [
            f"In the {unusual_adj} intersection of {concept} and {abstract_noun}, new possibilities emerge.",
            f"The essence of {concept} dissolves into {unusual_adj} {abstract_noun}, transcending ordinary understanding.",
            f"Between the known and unknown {concept}, there exists a {unusual_adj} {abstract_noun}."
        ]
        
        return random.choice(templates)
    
    def _create_creative_question(self, concepts: List[str]) -> str:
        """Create creative rhetorical questions."""
        concept = random.choice(concepts) if concepts else "existence"
        
        templates = [
            f"But what if {concept} is more than we imagine?",
            f"How does {concept} shape the very nature of perception?",
            f"What happens when {concept} meets its own reflection?",
            f"Could {concept} be the key to understanding something greater?"
        ]
        
        return random.choice(templates)
    
    # Creative technique applications
    def _apply_synesthesia(self, text: str) -> str:
        """Apply synesthetic descriptions (mixing senses)."""
        # Add synesthetic descriptions
        synesthetic_phrases = [
            "the sound tastes like copper and dreams",
            "colors that hum with ancient wisdom",
            "textures that whisper forgotten names",
            "the warm sound of yellow",
            "the bitter taste of silence",
            "the rough texture of laughter"
        ]
        
        # Insert synesthetic elements
        sentences = text.split('.')
        enhanced_sentences = []
        
        for sentence in sentences:
            enhanced_sentences.append(sentence)
            if random.random() < 0.3:
                synesthetic = random.choice(synesthetic_phrases)
                enhanced_sentences.append(f" Here, {synesthetic}")
        
        return '.'.join(enhanced_sentences)
    
    def _apply_metaphors(self, text: str) -> str:
        """Apply rich metaphorical language."""
        # Enhanced with metaphorical substitutions
        metaphor_pairs = [
            ("time", "a river flowing backwards"),
            ("memory", "crystalline fragments"),
            ("words", "luminous bridges"),
            ("silence", "velvet darkness"),
            ("thoughts", "gossamer threads")
        ]
        
        enhanced_text = text
        for original, metaphor in metaphor_pairs:
            if original in enhanced_text and random.random() < 0.4:
                enhanced_text = enhanced_text.replace(original, metaphor, 1)
        
        return enhanced_text
    
    def _apply_stream_consciousness(self, text: str) -> str:
        """Apply stream-of-consciousness style."""
        # Add flowing, interconnected thoughts
        connectors = [
            " and suddenly ",
            " yet somehow ",
            " because in that moment ",
            " which reminds me ",
            " or perhaps ",
            " then again "
        ]
        
        words = text.split()
        enhanced_words = []
        
        for i, word in enumerate(words):
            enhanced_words.append(word)
            if i > 10 and random.random() < 0.1:
                connector = random.choice(connectors)
                enhanced_words.append(connector)
        
        return ' '.join(enhanced_words)
    
    def _apply_minimalist_style(self, text: str) -> str:
        """Apply minimalist, precise style."""
        # Shorten sentences, remove excess
        sentences = text.split('.')
        minimalist_sentences = []
        
        for sentence in sentences:
            words = sentence.split()
            if len(words) > 8:
                # Keep essential words
                essential_words = [w for w in words if len(w) > 3]
                minimalist_sentence = ' '.join(essential_words[:6])
                minimalist_sentences.append(minimalist_sentence)
            else:
                minimalist_sentences.append(sentence)
        
        return '. '.join(minimalist_sentences)
    
    def _apply_baroque_style(self, text: str) -> str:
        """Apply elaborate, ornate style."""
        # Add elaborate descriptions
        baroque_modifiers = [
            "exquisitely intricate",
            "sublimely complex", 
            "magnificently elaborate",
            "breathtakingly ornate",
            "wondrously detailed"
        ]
        
        sentences = text.split('.')
        baroque_sentences = []
        
        for sentence in sentences:
            if random.random() < 0.4:
                modifier = random.choice(baroque_modifiers)
                # Insert modifier into sentence
                words = sentence.split()
                if len(words) > 3:
                    insertion_point = random.randint(1, len(words)-1)
                    words.insert(insertion_point, modifier)
                    sentence = ' '.join(words)
            
            baroque_sentences.append(sentence)
        
        return '.'.join(baroque_sentences)
    
    def _apply_experimental_punctuation(self, text: str) -> str:
        """Apply experimental punctuation patterns."""
        experimental_marks = ['...', '!', '?', ';', '—', ':', '!!', '??']
        
        sentences = text.split('.')
        experimental_sentences = []
        
        for sentence in sentences:
            if random.random() < 0.3:
                new_mark = random.choice(experimental_marks)
                sentence = sentence + new_mark
            experimental_sentences.append(sentence)
        
        return ' '.join(experimental_sentences)
    
    def _apply_repetition_variation(self, text: str) -> str:
        """Apply creative repetition with variations."""
        words = text.split()
        if len(words) < 10:
            return text
        
        # Find key words to repeat with variation
        key_words = [w for w in words if len(w) > 4][:3]
        
        enhanced_words = words.copy()
        
        for key_word in key_words:
            if random.random() < 0.3:
                variations = [
                    key_word,
                    key_word.upper(),
                    f"*{key_word}*",
                    f"{key_word}, {key_word}",
                ]
                
                # Replace one instance with variation
                for i, word in enumerate(enhanced_words):
                    if word == key_word:
                        enhanced_words[i] = random.choice(variations)
                        break
        
        return ' '.join(enhanced_words)


class DataAugmentation:
    """
    Data augmentation utilities for creativity training.
    
    Provides methods to augment existing text data to create
    more diverse training samples.
    """
    
    @staticmethod
    def augment_prompt_variations(base_prompts: List[str]) -> List[str]:
        """Create variations of existing prompts to increase diversity."""
        augmented_prompts = []
        
        style_variations = [
            "In a stream-of-consciousness style, ",
            "Using vivid metaphors and imagery, ",
            "With experimental sentence structures, ",
            "Through a minimalist approach, ",
            "In an ornate, baroque style, ",
            "Using synesthetic descriptions, ",
            "With unconventional punctuation, ",
        ]
        
        perspective_variations = [
            "From the perspective of an observer, ",
            "As if speaking to a close friend, ",
            "In the voice of a poet, ",
            "Through the eyes of a child, ",
            "With the wisdom of age, ",
        ]
        
        for prompt in base_prompts:
            # Original prompt
            augmented_prompts.append(prompt)
            
            # Style variations
            for style in style_variations[:3]:  # Use subset to avoid explosion
                if random.random() < 0.3:
                    augmented_prompts.append(style + prompt.lower())
            
            # Perspective variations
            for perspective in perspective_variations[:2]:
                if random.random() < 0.2:
                    augmented_prompts.append(perspective + prompt.lower())
        
        return augmented_prompts
    
    @staticmethod
    def create_prompt_chains(prompts: List[str]) -> List[str]:
        """Create chained prompts that build on each other."""
        chained_prompts = []
        
        for i in range(0, len(prompts) - 1, 2):
            if i + 1 < len(prompts):
                base_prompt = prompts[i]
                second_prompt = prompts[i + 1]
                
                chains = [
                    f"{base_prompt} Then, building on that idea, {second_prompt.lower()}",
                    f"First, {base_prompt.lower()} Next, explore how this connects to: {second_prompt.lower()}",
                    f"{base_prompt} Let this inspire you to also {second_prompt.lower()}"
                ]
                
                chained_prompts.extend(chains)
        
        return chained_prompts


def load_external_creative_datasets(data_dir: Optional[Path] = None) -> Dict[str, List[str]]:
    """
    Load external creative writing datasets if available.
    
    Args:
        data_dir: Directory containing additional creative text data
        
    Returns:
        Dictionary of loaded creative texts by category
    """
    creative_texts = {
        "examples": [],
        "prompts": [],
        "samples": []
    }
    
    if data_dir and data_dir.exists():
        # Load JSON files if they exist
        for file_path in data_dir.glob("*.json"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                if isinstance(data, dict):
                    for key, values in data.items():
                        if key in creative_texts and isinstance(values, list):
                            creative_texts[key].extend(values)
                elif isinstance(data, list):
                    creative_texts["samples"].extend(data)
                    
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        # Load text files
        for file_path in data_dir.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:
                        creative_texts["samples"].append(content)
            except IOError as e:
                print(f"Warning: Could not load {file_path}: {e}")
    
    return creative_texts