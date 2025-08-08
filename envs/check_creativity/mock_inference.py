#!/usr/bin/env python3
"""
Mock Inference System for Creativity Environment

Provides mock model responses for testing and development without requiring
actual model inference. Supports different creativity levels and response patterns.
"""

import json
import random
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
import numpy as np
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class MockResponse:
    """Mock response from inference system."""
    content: str
    metadata: Dict[str, Any]
    simulated_metrics: Dict[str, float]
    generation_info: Dict[str, Any]


class MockCreativityInference:
    """
    Mock inference system that simulates creative text generation.
    
    Provides realistic creative responses based on prompt analysis and
    configured creativity levels.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize mock inference system.
        
        Args:
            config_path: Path to mock configuration file
        """
        self.config = self._load_config(config_path)
        self.response_templates = self._init_response_templates()
        self.creativity_patterns = self._init_creativity_patterns()
        
        # Initialize response statistics
        self.response_count = 0
        self.category_usage = {
            "storytelling": 0,
            "poetry": 0,
            "descriptive": 0,
            "philosophical": 0,
            "experimental": 0
        }
        
        logger.info(f"MockCreativityInference initialized with mode: {self.config.get('mock_mode', 'creative')}")
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load mock configuration."""
        default_config = {
            "mock_mode": "creative",
            "response_quality": "high",
            "simulated_latency": 0.5,
            "mock_creativity_level": "high",
            "include_variety": True,
            "use_creative_vocabulary": True
        }
        
        if config_path and config_path.exists():
            try:
                import toml
                file_config = toml.load(config_path)
                
                # Extract mock-specific settings
                config = default_config.copy()
                
                if 'model' in file_config:
                    config.update(file_config['model'])
                if 'generation' in file_config:
                    config.update(file_config['generation'])
                if 'mock_responses' in file_config:
                    config['response_templates'] = file_config['mock_responses']
                
                return config
                
            except Exception as e:
                logger.warning(f"Could not load config from {config_path}: {e}")
        
        return default_config
    
    def _init_response_templates(self) -> Dict[str, List[str]]:
        """Initialize response templates by category."""
        # Use templates from config if available
        if 'response_templates' in self.config:
            return self.config['response_templates']
        
        # Default templates
        return {
            "storytelling": [
                "In the twilight realm where shadows dance with forgotten memories, there existed a peculiar library whose books wrote themselves each midnight. The weathered librarian, whose eyes held the luminescence of captured starlight, would whisper secrets to the empty shelves, watching as gossamer threads of narrative materialized into tangible stories.",
                "The clockmaker's daughter discovered that each timepiece in her father's shop contained a different rhythm of existence. Some ticked with the heartbeat of hummingbirds, others pulsed with the slow, ancient breath of mountains. When she wound the celestial chronometer at the shop's center, time itself became malleable, stretching and contracting like warm honey in her fingers.",
                "Beyond the veil of ordinary perception lay a marketplace where emotions were currency and memories could be traded like precious gems. The merchant of dreams sat beneath a canopy of crystallized laughter, her wares displayed in bottles that hummed with the essence of forgotten summer afternoons and the taste of first snow.",
            ],
            "poetry": [
                "Silence tastes of silver rain / on copper-colored autumn leaves, / while whispered words bloom crystalline / in gardens where the moon grieves. / Each breath becomes a painted note / upon the canvas of the night, / where dreams and waking thoughts devote / themselves to dance in fading light.",
                "Between the spaces of heartbeats / lie infinite symphonies unheard, / where color sings and shadow meets / the echo of each unspoken word. / Time moves like honey through my veins, / sweet amber capturing moments bright, / while consciousness itself explains / the texture of perceived delight.",
                "Words cascade like liquid starlight / through neural pathways unexplored, / each syllable a universe / where meaning waits to be adored. / In the architecture of language / we build cathedrals made of sound, / where whispered prayers and shouted joy / make sacred what was merely ground.",
            ],
            "descriptive": [
                "The marketplace existed in the liminal space between sleeping and waking, where vendors sold bottled laughter and crystallized tears. Gossamer curtains separated stalls offering memories in mason jars, their contents swirling with opalescent mist that tasted of childhood summers and forgotten names.",
                "The garden breathed with photosynthetic sighs, each leaf a miniature lung exhaling stories into the perfumed air. Flowers bloomed backwards through time, their petals unfurling from wilted brown to vibrant youth, while the ancient oak at its center hummed lullabies in languages that predated human speech.",
                "In the observatory of lost thoughts, telescopes pointed inward toward the galaxies of the mind. Dust motes danced in shafts of memory-light, each particle a forgotten moment spinning through the vast emptiness of what might have been, creating constellations of regret and wonder.",
            ],
            "philosophical": [
                "What if consciousness itself is merely the universe's attempt to understand its own existence through countless fragmentary perspectives? Each mind becomes a unique lens through which infinity observes itself, creating meaning from the intersection of awareness and experience, like light refracting through a prism of individual understanding.",
                "The weight of unspoken words accumulates in the spaces between conversations, creating invisible archives of potential meaning that hover in the air around us. Perhaps silence is not the absence of communication, but rather its most profound form—a language of pauses that speaks volumes about the ineffable nature of human connection.",
                "Time exists not as a river flowing forward, but as an ocean where past, present, and future exist simultaneously in different depths. We swim through moments, sometimes diving deep into memory, sometimes floating on the surface of now, occasionally catching glimpses of the mysterious currents that carry us toward tomorrow.",
            ],
            "experimental": [
                "words | cascade | through | neural | pathways || SUDDEN || like || digital || rain ||| each || thought || fragments || into || kaleidoscopic || meaning || ??? || where || punctuation || becomes || rhythm || and || syntax || dissolves || into || pure || expression ||||||",
                "The question-mark-shaped-thoughts-dance-between-semicolons; while exclamation-points! burst! like! fireworks! in! the! mind's! eye! and periods. become. stepping. stones. across. the. river. of. consciousness. flowing. ever. flowing. into. the. ocean. of. collective. understanding???",
                "In the beginning was the Word... but which word? THE? word? WORD? word. word! word? word--- Each punctuation mark creates a universe of meaning, a multiverse of interpretation where every comma is a pause for breath and every ellipsis a doorway to... infinity...",
            ]
        }
    
    def _init_creativity_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize creativity patterns for different response types."""
        return {
            "high": {
                "unusual_word_ratio": 0.25,
                "sentence_variety": 0.8,
                "punctuation_diversity": 0.7,
                "metaphor_density": 0.6,
                "word_length_variance": 2.5
            },
            "medium": {
                "unusual_word_ratio": 0.15,
                "sentence_variety": 0.6,
                "punctuation_diversity": 0.5,
                "metaphor_density": 0.4,
                "word_length_variance": 2.0
            },
            "low": {
                "unusual_word_ratio": 0.08,
                "sentence_variety": 0.4,
                "punctuation_diversity": 0.3,
                "metaphor_density": 0.2,
                "word_length_variance": 1.5
            }
        }
    
    def _classify_prompt(self, prompt: str) -> str:
        """Classify prompt to select appropriate response category."""
        prompt_lower = prompt.lower()
        
        # Keywords for different categories
        keywords = {
            "storytelling": ["story", "narrative", "character", "plot", "tale", "adventure"],
            "poetry": ["poem", "verse", "rhyme", "stanza", "poetry", "lyric"],
            "descriptive": ["describe", "paint", "picture", "scene", "landscape", "appearance"],
            "philosophical": ["meaning", "existence", "nature", "philosophy", "contemplate", "reflect"],
            "experimental": ["experimental", "creative", "unusual", "innovative", "unique"]
        }
        
        # Score each category
        scores = {}
        for category, words in keywords.items():
            scores[category] = sum(1 for word in words if word in prompt_lower)
        
        # Return category with highest score, or random if tie
        if max(scores.values()) == 0:
            return random.choice(list(keywords.keys()))
        
        return max(scores, key=scores.get)
    
    def _generate_creative_response(self, prompt: str, category: str) -> str:
        """Generate creative response based on prompt and category."""
        # Select base template
        if category in self.response_templates:
            base_response = random.choice(self.response_templates[category])
        else:
            # Fallback to any category
            all_responses = []
            for responses in self.response_templates.values():
                all_responses.extend(responses)
            base_response = random.choice(all_responses)
        
        # Apply creativity modifications based on mode
        creativity_level = self.config.get("mock_creativity_level", "high")
        
        if creativity_level == "high":
            # Enhance with additional creative elements
            enhanced_response = self._enhance_creativity(base_response)
        elif creativity_level == "medium":
            # Use base response with minor enhancements
            enhanced_response = base_response
        else:
            # Simplify the response
            enhanced_response = self._simplify_response(base_response)
        
        return enhanced_response
    
    def _enhance_creativity(self, text: str) -> str:
        """Enhance text with additional creative elements."""
        # Add creative flourishes
        creative_additions = [
            " Like whispered secrets between stars, ",
            " In the gossamer threads of imagination, ",
            " Where dreams crystallize into reality, ",
            " Through corridors of liquid light, ",
        ]
        
        # Randomly insert creative elements
        sentences = text.split('. ')
        enhanced_sentences = []
        
        for i, sentence in enumerate(sentences):
            enhanced_sentences.append(sentence)
            
            # 30% chance to add creative flourish
            if random.random() < 0.3 and i < len(sentences) - 1:
                flourish = random.choice(creative_additions)
                enhanced_sentences[-1] += flourish
        
        return '. '.join(enhanced_sentences)
    
    def _simplify_response(self, text: str) -> str:
        """Simplify text for lower creativity mode."""
        # Remove some of the more unusual words
        unusual_words = [
            "gossamer", "ethereal", "crystalline", "opalescent", 
            "iridescent", "luminous", "kaleidoscopic", "gossamer"
        ]
        
        simple_replacements = {
            "gossamer": "delicate",
            "ethereal": "light",
            "crystalline": "clear",
            "opalescent": "shimmering",
            "iridescent": "colorful",
            "luminous": "bright",
            "kaleidoscopic": "colorful"
        }
        
        simplified = text
        for unusual, simple in simple_replacements.items():
            if unusual in simplified and random.random() < 0.5:
                simplified = simplified.replace(unusual, simple)
        
        return simplified
    
    def _simulate_metrics(self, text: str, category: str) -> Dict[str, float]:
        """Simulate creativity metrics for the generated text."""
        creativity_level = self.config.get("mock_creativity_level", "high")
        patterns = self.creativity_patterns[creativity_level]
        
        # Base scores with some randomness
        simulated_metrics = {}
        
        # Entropy (3.0-7.5 range)
        entropy_base = 5.5 if creativity_level == "high" else 4.5 if creativity_level == "medium" else 3.5
        simulated_metrics["entropy"] = max(3.0, min(7.5, entropy_base + random.uniform(-0.5, 0.5)))
        
        # Distinct ratio (0.4-0.9 range)
        distinct_base = 0.75 if creativity_level == "high" else 0.65 if creativity_level == "medium" else 0.55
        simulated_metrics["distinct_ratio"] = max(0.4, min(0.9, distinct_base + random.uniform(-0.1, 0.1)))
        
        # Uncommon words (0.1-0.4 range)
        uncommon_base = patterns["unusual_word_ratio"]
        simulated_metrics["uncommon_words"] = max(0.1, min(0.4, uncommon_base + random.uniform(-0.05, 0.05)))
        
        # Bigram diversity (0.6-0.95 range)
        bigram_base = 0.85 if creativity_level == "high" else 0.75 if creativity_level == "medium" else 0.65
        simulated_metrics["bigram_diversity"] = max(0.6, min(0.95, bigram_base + random.uniform(-0.05, 0.05)))
        
        # Sentence variance (1.0-4.0 range)
        sentence_var_base = patterns["sentence_variety"] * 3.5
        simulated_metrics["sentence_variance"] = max(1.0, min(4.0, sentence_var_base + random.uniform(-0.3, 0.3)))
        
        # Word variance (1.5-3.0 range)
        word_var_base = patterns["word_length_variance"]
        simulated_metrics["word_variance"] = max(1.5, min(3.0, word_var_base + random.uniform(-0.2, 0.2)))
        
        # Ending variety (0.5-2.0 range)
        ending_var_base = patterns["punctuation_diversity"] * 1.8
        simulated_metrics["ending_variety"] = max(0.5, min(2.0, ending_var_base + random.uniform(-0.1, 0.1)))
        
        return simulated_metrics
    
    def _calculate_total_score(self, metrics: Dict[str, float]) -> float:
        """Calculate total creativity score from individual metrics."""
        weights = {
            'w_entropy': 1.0,
            'w_distinct': 1.0,
            'w_uncommon': 0.8,
            'w_bigrams': 1.2,
            'w_sentence_len_var': 0.6,
            'w_word_len_var': 0.4,
            'w_sentence_end_var': 0.5,
        }
        
        total_score = (
            weights['w_entropy'] * metrics.get('entropy', 0) +
            weights['w_distinct'] * metrics.get('distinct_ratio', 0) +
            weights['w_uncommon'] * metrics.get('uncommon_words', 0) +
            weights['w_bigrams'] * metrics.get('bigram_diversity', 0) +
            weights['w_sentence_len_var'] * metrics.get('sentence_variance', 0) +
            weights['w_word_len_var'] * metrics.get('word_variance', 0) +
            weights['w_sentence_end_var'] * metrics.get('ending_variety', 0)
        )
        
        return total_score
    
    def generate_response(
        self, 
        prompt: str, 
        max_tokens: int = 300,
        temperature: float = 0.8,
        **kwargs
    ) -> MockResponse:
        """
        Generate mock creative response to a prompt.
        
        Args:
            prompt: Input prompt for creative generation
            max_tokens: Maximum tokens (used for response length estimation)
            temperature: Generation temperature (affects creativity simulation)
            **kwargs: Additional generation parameters
            
        Returns:
            MockResponse with generated content and metadata
        """
        # Simulate processing latency
        latency = self.config.get("simulated_latency", 0.5)
        if latency > 0:
            time.sleep(latency + random.uniform(-0.1, 0.1))
        
        # Classify prompt
        category = self._classify_prompt(prompt)
        self.category_usage[category] += 1
        
        # Generate response
        content = self._generate_creative_response(prompt, category)
        
        # Simulate response length based on max_tokens
        words = content.split()
        target_words = min(len(words), int(max_tokens * 0.75))  # Rough tokens to words conversion
        if len(words) > target_words:
            content = ' '.join(words[:target_words]) + "..."
        
        # Simulate creativity metrics
        simulated_metrics = self._simulate_metrics(content, category)
        total_score = self._calculate_total_score(simulated_metrics)
        
        # Create response metadata
        metadata = {
            "model": "mock-creativity-model",
            "category": category,
            "creativity_level": self.config.get("mock_creativity_level", "high"),
            "total_creativity_score": total_score,
            "response_id": self.response_count,
            "processing_time": latency
        }
        
        generation_info = {
            "prompt_classification": category,
            "template_used": True,
            "enhancements_applied": self.config.get("mock_creativity_level", "high"),
            "simulated_processing": True
        }
        
        self.response_count += 1
        
        response = MockResponse(
            content=content,
            metadata=metadata,
            simulated_metrics=simulated_metrics,
            generation_info=generation_info
        )
        
        logger.info(f"Generated mock response #{self.response_count} (category: {category}, score: {total_score:.3f})")
        
        return response
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get usage statistics for the mock system."""
        return {
            "total_responses": self.response_count,
            "category_usage": self.category_usage.copy(),
            "configuration": self.config.copy(),
            "creativity_patterns": self.creativity_patterns.copy()
        }
    
    def batch_generate(
        self, 
        prompts: List[str], 
        **kwargs
    ) -> List[MockResponse]:
        """
        Generate responses for multiple prompts.
        
        Args:
            prompts: List of prompts to process
            **kwargs: Generation parameters
            
        Returns:
            List of MockResponse objects
        """
        responses = []
        
        for prompt in prompts:
            response = self.generate_response(prompt, **kwargs)
            responses.append(response)
        
        logger.info(f"Completed batch generation of {len(responses)} responses")
        
        return responses


def main():
    """Demo of mock inference system."""
    # Initialize mock system
    mock_system = MockCreativityInference()
    
    # Test prompts
    test_prompts = [
        "Write a short story about a character who can taste colors.",
        "Create a poem about the sound of silence.",
        "Describe a marketplace that exists only in dreams.",
        "What is the nature of time from the perspective of a mayfly?",
        "Write using only questions, but tell a complete story."
    ]
    
    print("Mock Creativity Inference System Demo")
    print("=" * 50)
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\nTest {i}: {prompt[:50]}...")
        
        response = mock_system.generate_response(
            prompt=prompt,
            max_tokens=200,
            temperature=0.8
        )
        
        print(f"Category: {response.metadata['category']}")
        print(f"Creativity Score: {response.metadata['total_creativity_score']:.3f}")
        print(f"Content: {response.content[:150]}...")
        
        # Show some metrics
        metrics = response.simulated_metrics
        print(f"Key Metrics: entropy={metrics['entropy']:.2f}, "
              f"distinct_ratio={metrics['distinct_ratio']:.2f}, "
              f"uncommon_words={metrics['uncommon_words']:.2f}")
    
    # Show statistics
    print("\n" + "=" * 50)
    print("System Statistics:")
    stats = mock_system.get_statistics()
    print(f"Total responses: {stats['total_responses']}")
    print(f"Category usage: {stats['category_usage']}")


if __name__ == "__main__":
    main()