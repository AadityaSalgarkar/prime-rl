#!/usr/bin/env python3
"""
Evaluation Rubrics and Analysis for Creativity Environment

This module provides comprehensive evaluation rubrics, metrics analysis,
and performance tracking for the creativity RLVR environment.
"""

import json
import statistics
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from pathlib import Path
from dataclasses import dataclass, asdict
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

from .reward import reward_function


@dataclass
class CreativityAnalysis:
    """Comprehensive analysis of creativity metrics."""
    sample_id: str
    text: str
    total_score: float
    component_scores: Dict[str, float]
    text_stats: Dict[str, Any]
    timestamp: str
    category: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class CreativityRubric:
    """
    Comprehensive rubric for evaluating creativity across multiple dimensions.
    
    Provides structured evaluation criteria and scoring methods
    for different aspects of creative text generation.
    """
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize creativity rubric.
        
        Args:
            weights: Custom weights for creativity components
        """
        self.weights = weights or {
            'w_entropy': 1.0,
            'w_distinct': 1.0,
            'w_uncommon': 0.8,
            'w_bigrams': 1.2,
            'w_sentence_len_var': 0.6,
            'w_word_len_var': 0.4,
            'w_sentence_end_var': 0.5,
        }
        
        # Define creativity evaluation criteria
        self.criteria = {
            'linguistic_diversity': {
                'description': 'Variety in vocabulary, sentence structure, and expression',
                'components': ['entropy', 'distinct_ratio', 'bigram_diversity'],
                'weight': 0.3
            },
            'originality': {
                'description': 'Use of uncommon words and novel combinations',
                'components': ['uncommon_words', 'word_creativity'],
                'weight': 0.25
            },
            'structural_variety': {
                'description': 'Variation in sentence and word lengths',
                'components': ['sentence_variance', 'word_variance'],
                'weight': 0.2
            },
            'expressive_range': {
                'description': 'Variety in punctuation and sentence endings',
                'components': ['ending_variety', 'punctuation_diversity'],
                'weight': 0.15
            },
            'coherence': {
                'description': 'Maintains readability while being creative',
                'components': ['readability', 'flow'],
                'weight': 0.1
            }
        }
    
    def evaluate_text(self, text: str, prompt: str = "", category: str = "") -> CreativityAnalysis:
        """
        Comprehensive evaluation of text creativity.
        
        Args:
            text: Text to evaluate
            prompt: Original prompt (for context)
            category: Text category
            
        Returns:
            Comprehensive creativity analysis
        """
        # Calculate core creativity score
        total_score = reward_function(text, **self.weights)
        
        # Calculate component scores
        component_scores = self._calculate_component_scores(text)
        
        # Calculate text statistics
        text_stats = self._calculate_text_stats(text)
        
        # Create analysis object
        analysis = CreativityAnalysis(
            sample_id=f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            text=text,
            total_score=total_score,
            component_scores=component_scores,
            text_stats=text_stats,
            timestamp=datetime.now().isoformat(),
            category=category,
            metadata={
                'prompt': prompt,
                'evaluation_weights': self.weights
            }
        )
        
        return analysis
    
    def _calculate_component_scores(self, text: str) -> Dict[str, float]:
        """Calculate individual component scores."""
        try:
            import nltk
            from collections import Counter
            from nltk.util import bigrams
            from nltk.corpus import words as nltk_words
            
            # Ensure NLTK data
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt', quiet=True)
                nltk.download('words', quiet=True)
            
            COMMON_WORDS = set(nltk_words.words())
            
            # Process text
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            words = [w.lower() for w in words if w.isalpha()]
            
            if len(words) == 0:
                return {key: 0.0 for key in ['entropy', 'distinct_ratio', 'uncommon_words', 
                                           'bigram_diversity', 'sentence_variance', 
                                           'word_variance', 'ending_variety']}
            
            # Calculate components
            word_freqs = Counter(words)
            probs = [count / len(words) for count in word_freqs.values()]
            entropy = -sum(p * np.log2(p) for p in probs if p > 0)
            
            distinct_ratio = len(set(words)) / len(words)
            
            uncommon_words = [w for w in words if w not in COMMON_WORDS]
            uncommon_ratio = len(uncommon_words) / len(words)
            
            word_bigrams = list(bigrams(words))
            bigram_diversity = len(set(word_bigrams)) / (len(word_bigrams) + 1e-8)
            
            sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
            sentence_variance = np.std(sentence_lengths) if len(sentences) > 1 else 0
            
            word_lengths = [len(w) for w in words]
            word_variance = np.std(word_lengths)
            
            sentence_endings = [s.strip()[-1] if s.strip() else '.' for s in sentences]
            end_counts = Counter(sentence_endings)
            end_probs = [count / len(sentences) for count in end_counts.values()]
            ending_variety = -sum(p * np.log2(p) for p in end_probs if p > 0)
            
            return {
                'entropy': entropy,
                'distinct_ratio': distinct_ratio,
                'uncommon_words': uncommon_ratio,
                'bigram_diversity': bigram_diversity,
                'sentence_variance': sentence_variance,
                'word_variance': word_variance,
                'ending_variety': ending_variety
            }
            
        except Exception as e:
            print(f"Error calculating component scores: {e}")
            return {key: 0.0 for key in ['entropy', 'distinct_ratio', 'uncommon_words',
                                       'bigram_diversity', 'sentence_variance',
                                       'word_variance', 'ending_variety']}
    
    def _calculate_text_stats(self, text: str) -> Dict[str, Any]:
        """Calculate comprehensive text statistics."""
        try:
            import nltk
            
            sentences = nltk.sent_tokenize(text)
            words = nltk.word_tokenize(text)
            alphabetic_words = [w for w in words if w.isalpha()]
            
            stats = {
                'char_count': len(text),
                'word_count': len(alphabetic_words),
                'sentence_count': len(sentences),
                'avg_sentence_length': np.mean([len(nltk.word_tokenize(s)) for s in sentences]) if sentences else 0,
                'avg_word_length': np.mean([len(w) for w in alphabetic_words]) if alphabetic_words else 0,
                'unique_words': len(set(alphabetic_words)),
                'lexical_diversity': len(set(alphabetic_words)) / len(alphabetic_words) if alphabetic_words else 0,
                'punctuation_marks': len([c for c in text if c in '.,!?;:-()[]"\''])
            }
            
            # Advanced statistics
            if alphabetic_words:
                word_lengths = [len(w) for w in alphabetic_words]
                stats.update({
                    'word_length_std': np.std(word_lengths),
                    'long_words_ratio': len([w for w in alphabetic_words if len(w) > 6]) / len(alphabetic_words),
                    'short_words_ratio': len([w for w in alphabetic_words if len(w) <= 3]) / len(alphabetic_words)
                })
            
            return stats
            
        except Exception as e:
            print(f"Error calculating text stats: {e}")
            return {'char_count': len(text), 'word_count': 0, 'sentence_count': 0}
    
    def grade_creativity_level(self, score: float) -> Tuple[str, str]:
        """
        Grade creativity level based on score.
        
        Args:
            score: Total creativity score
            
        Returns:
            Tuple of (grade, description)
        """
        if score >= 9.0:
            return "Exceptional", "Highly creative with outstanding diversity and originality"
        elif score >= 7.0:
            return "Creative", "Good creativity with strong diversity metrics"
        elif score >= 5.0:
            return "Moderate", "Some creative elements present"
        elif score >= 3.0:
            return "Limited", "Basic creativity with room for improvement"
        else:
            return "Minimal", "Very limited creativity, mostly conventional language"


class CreativityTracker:
    """
    Tracks creativity metrics over time and provides analysis.
    
    Monitors training progress, identifies trends, and provides
    insights for improving creativity training.
    """
    
    def __init__(self, save_dir: Optional[Path] = None):
        """
        Initialize creativity tracker.
        
        Args:
            save_dir: Directory to save tracking data
        """
        self.save_dir = Path(save_dir) if save_dir else Path("creativity_tracking")
        self.save_dir.mkdir(exist_ok=True)
        
        # Tracking data
        self.analyses: List[CreativityAnalysis] = []
        self.training_metrics: Dict[str, List[float]] = defaultdict(list)
        self.iteration_scores: List[float] = []
        
        # Analysis cache
        self._analysis_cache = {}
    
    def add_analysis(self, analysis: CreativityAnalysis, iteration: Optional[int] = None):
        """Add a new creativity analysis."""
        self.analyses.append(analysis)
        
        # Track metrics over time
        for metric, value in analysis.component_scores.items():
            self.training_metrics[metric].append(value)
        
        self.training_metrics['total_score'].append(analysis.total_score)
        
        if iteration is not None:
            self.iteration_scores.append(analysis.total_score)
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get training progress summary."""
        if not self.analyses:
            return {"status": "No data available"}
        
        recent_analyses = self.analyses[-10:]  # Last 10 analyses
        
        summary = {
            "total_samples": len(self.analyses),
            "latest_score": self.analyses[-1].total_score,
            "recent_average": np.mean([a.total_score for a in recent_analyses]),
            "overall_average": np.mean([a.total_score for a in self.analyses]),
            "improvement_trend": self._calculate_trend(),
            "best_score": max(a.total_score for a in self.analyses),
            "component_progress": self._analyze_component_progress(),
            "category_performance": self._analyze_category_performance()
        }
        
        return summary
    
    def _calculate_trend(self) -> str:
        """Calculate overall improvement trend."""
        if len(self.analyses) < 5:
            return "insufficient_data"
        
        recent_scores = [a.total_score for a in self.analyses[-10:]]
        early_scores = [a.total_score for a in self.analyses[:10]]
        
        recent_avg = np.mean(recent_scores)
        early_avg = np.mean(early_scores)
        
        improvement = (recent_avg - early_avg) / early_avg if early_avg > 0 else 0
        
        if improvement > 0.1:
            return "strong_improvement"
        elif improvement > 0.05:
            return "moderate_improvement"
        elif improvement > -0.05:
            return "stable"
        else:
            return "declining"
    
    def _analyze_component_progress(self) -> Dict[str, Dict[str, float]]:
        """Analyze progress of individual creativity components."""
        component_progress = {}
        
        for component in ['entropy', 'distinct_ratio', 'uncommon_words', 
                         'bigram_diversity', 'sentence_variance', 'word_variance', 'ending_variety']:
            if component in self.training_metrics and len(self.training_metrics[component]) >= 5:
                values = self.training_metrics[component]
                recent_avg = np.mean(values[-10:])
                early_avg = np.mean(values[:10])
                
                component_progress[component] = {
                    'current': recent_avg,
                    'initial': early_avg,
                    'improvement': (recent_avg - early_avg) / early_avg if early_avg > 0 else 0,
                    'std': np.std(values),
                    'trend': 'improving' if recent_avg > early_avg else 'stable' if recent_avg == early_avg else 'declining'
                }
        
        return component_progress
    
    def _analyze_category_performance(self) -> Dict[str, Dict[str, float]]:
        """Analyze performance by text category."""
        category_performance = defaultdict(list)
        
        for analysis in self.analyses:
            if analysis.category:
                category_performance[analysis.category].append(analysis.total_score)
        
        category_stats = {}
        for category, scores in category_performance.items():
            category_stats[category] = {
                'count': len(scores),
                'average': np.mean(scores),
                'std': np.std(scores),
                'best': max(scores),
                'worst': min(scores)
            }
        
        return category_stats
    
    def generate_training_insights(self) -> Dict[str, Any]:
        """Generate actionable training insights."""
        if len(self.analyses) < 10:
            return {"message": "Insufficient data for insights"}
        
        progress = self.get_progress_summary()
        
        insights = {
            "overall_performance": self._assess_overall_performance(progress),
            "component_recommendations": self._get_component_recommendations(progress),
            "training_adjustments": self._suggest_training_adjustments(progress),
            "strengths": self._identify_strengths(progress),
            "areas_for_improvement": self._identify_weaknesses(progress)
        }
        
        return insights
    
    def _assess_overall_performance(self, progress: Dict[str, Any]) -> str:
        """Assess overall training performance."""
        avg_score = progress['overall_average']
        trend = progress['improvement_trend']
        
        if avg_score >= 7.0 and trend in ['strong_improvement', 'stable']:
            return "excellent"
        elif avg_score >= 5.0 and trend != 'declining':
            return "good"
        elif avg_score >= 3.0:
            return "moderate"
        else:
            return "needs_improvement"
    
    def _get_component_recommendations(self, progress: Dict[str, Any]) -> Dict[str, str]:
        """Get recommendations for improving specific components."""
        recommendations = {}
        
        component_progress = progress.get('component_progress', {})
        
        for component, stats in component_progress.items():
            if stats['trend'] == 'declining':
                if component == 'entropy':
                    recommendations[component] = "Encourage more diverse vocabulary usage"
                elif component == 'distinct_ratio':
                    recommendations[component] = "Reduce word repetition, increase lexical variety"
                elif component == 'uncommon_words':
                    recommendations[component] = "Include prompts that encourage unusual word usage"
                elif component == 'bigram_diversity':
                    recommendations[component] = "Promote varied phrase combinations and word pairs"
                elif component in ['sentence_variance', 'word_variance']:
                    recommendations[component] = "Encourage varied sentence and word lengths"
                elif component == 'ending_variety':
                    recommendations[component] = "Use prompts that require different punctuation styles"
        
        return recommendations
    
    def _suggest_training_adjustments(self, progress: Dict[str, Any]) -> Dict[str, Any]:
        """Suggest training parameter adjustments."""
        adjustments = {}
        
        avg_score = progress['overall_average']
        trend = progress['improvement_trend']
        
        if trend == 'declining':
            adjustments['learning_rate'] = "Consider reducing learning rate"
            adjustments['temperature'] = "Increase temperature for more diverse outputs"
        elif avg_score < 4.0:
            adjustments['reward_weights'] = "Consider adjusting reward component weights"
            adjustments['prompts'] = "Use more diverse and challenging prompts"
        elif trend == 'stable' and avg_score > 6.0:
            adjustments['difficulty'] = "Consider increasing prompt difficulty"
            adjustments['evaluation'] = "Add more sophisticated creativity metrics"
        
        return adjustments
    
    def _identify_strengths(self, progress: Dict[str, Any]) -> List[str]:
        """Identify training strengths."""
        strengths = []
        
        component_progress = progress.get('component_progress', {})
        for component, stats in component_progress.items():
            if stats['current'] > 0.8 or stats['improvement'] > 0.2:
                strengths.append(f"Strong {component.replace('_', ' ')}")
        
        if progress['improvement_trend'] in ['strong_improvement', 'moderate_improvement']:
            strengths.append("Consistent improvement trend")
        
        return strengths
    
    def _identify_weaknesses(self, progress: Dict[str, Any]) -> List[str]:
        """Identify areas needing improvement."""
        weaknesses = []
        
        component_progress = progress.get('component_progress', {})
        for component, stats in component_progress.items():
            if stats['current'] < 0.3 or stats['improvement'] < -0.1:
                weaknesses.append(f"Low {component.replace('_', ' ')}")
        
        if progress['overall_average'] < 4.0:
            weaknesses.append("Below-average overall creativity")
        
        return weaknesses
    
    def save_analysis(self, filepath: Optional[Path] = None):
        """Save analysis data to file."""
        if filepath is None:
            filepath = self.save_dir / f"creativity_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'analyses': [asdict(analysis) for analysis in self.analyses],
            'training_metrics': dict(self.training_metrics),
            'progress_summary': self.get_progress_summary(),
            'training_insights': self.generate_training_insights(),
            'metadata': {
                'total_samples': len(self.analyses),
                'generated_at': datetime.now().isoformat()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_analysis(self, filepath: Path):
        """Load analysis data from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore analyses
        self.analyses = []
        for analysis_data in data.get('analyses', []):
            analysis = CreativityAnalysis(**analysis_data)
            self.analyses.append(analysis)
        
        # Restore metrics
        self.training_metrics = defaultdict(list, data.get('training_metrics', {}))
    
    def create_visualizations(self, save_dir: Optional[Path] = None):
        """Create visualization plots of creativity metrics."""
        if save_dir is None:
            save_dir = self.save_dir / "visualizations"
        save_dir.mkdir(exist_ok=True)
        
        # Only create visualizations if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Overall creativity score over time
            plt.figure(figsize=(12, 6))
            scores = [a.total_score for a in self.analyses]
            plt.plot(scores, linewidth=2)
            plt.title('Creativity Score Over Time')
            plt.xlabel('Sample')
            plt.ylabel('Creativity Score')
            plt.grid(True, alpha=0.3)
            plt.savefig(save_dir / 'creativity_over_time.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Component scores distribution
            if len(self.analyses) > 5:
                fig, axes = plt.subplots(2, 4, figsize=(16, 10))
                axes = axes.flatten()
                
                components = ['entropy', 'distinct_ratio', 'uncommon_words', 'bigram_diversity',
                             'sentence_variance', 'word_variance', 'ending_variety']
                
                for i, component in enumerate(components):
                    if i < len(axes):
                        values = [a.component_scores.get(component, 0) for a in self.analyses]
                        axes[i].hist(values, bins=20, alpha=0.7)
                        axes[i].set_title(f'{component.replace("_", " ").title()}')
                        axes[i].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(save_dir / 'component_distributions.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Category performance comparison
            category_scores = defaultdict(list)
            for analysis in self.analyses:
                if analysis.category:
                    category_scores[analysis.category].append(analysis.total_score)
            
            if len(category_scores) > 1:
                plt.figure(figsize=(10, 6))
                categories = list(category_scores.keys())
                scores_by_category = [category_scores[cat] for cat in categories]
                
                plt.boxplot(scores_by_category, labels=categories)
                plt.title('Creativity Score by Category')
                plt.ylabel('Creativity Score')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.savefig(save_dir / 'category_performance.png', dpi=300, bbox_inches='tight')
                plt.close()
            
        except ImportError:
            print("Matplotlib not available, skipping visualizations")


class CreativityBenchmark:
    """
    Benchmarking system for creativity evaluation.
    
    Provides standardized evaluation against reference texts
    and established creativity benchmarks.
    """
    
    def __init__(self):
        """Initialize benchmark system."""
        self.reference_texts = self._load_reference_texts()
        self.rubric = CreativityRubric()
    
    def _load_reference_texts(self) -> Dict[str, List[str]]:
        """Load reference texts for benchmarking."""
        # These could be loaded from files or predefined
        return {
            'high_creativity': [
                "The kaleidoscope of memories danced through her consciousness like gossamer threads weaving impossible tapestries. Each thought tasted of copper pennies and forgotten lullabies, while the silence between heartbeats echoed with the whispers of stars yet unborn.",
                "Time moved like honey through his fingers, each second a crystalline fragment containing universes of possibility. The room breathed with him, walls pulsing in rhythm with thoughts too large for words, too bright for ordinary sight."
            ],
            'medium_creativity': [
                "She walked through the garden, observing how each flower seemed to tell its own story. The roses whispered secrets of summer days, while the daisies hummed cheerful melodies that reminded her of childhood afternoons.",
                "The old bookstore held memories in every corner. Dust particles danced in golden sunbeams, and the smell of aged paper mixed with the faint aroma of forgotten adventures waiting to be discovered."
            ],
            'low_creativity': [
                "The cat sat on the mat. It was a nice day outside. The sun was shining and the birds were singing. Everything looked normal and peaceful.",
                "John went to the store to buy some milk. He walked down the street, entered the shop, picked up the milk, paid for it, and walked home. It was a regular day."
            ]
        }
    
    def benchmark_text(self, text: str) -> Dict[str, Any]:
        """
        Benchmark text against reference samples.
        
        Args:
            text: Text to benchmark
            
        Returns:
            Benchmark results with comparison scores
        """
        analysis = self.rubric.evaluate_text(text)
        
        # Compare against reference texts
        comparisons = {}
        for level, ref_texts in self.reference_texts.items():
            ref_scores = [self.rubric.evaluate_text(ref_text).total_score for ref_text in ref_texts]
            avg_ref_score = np.mean(ref_scores)
            
            comparisons[level] = {
                'reference_average': avg_ref_score,
                'difference': analysis.total_score - avg_ref_score,
                'percentile': self._calculate_percentile(analysis.total_score, ref_scores)
            }
        
        # Determine benchmark level
        benchmark_level = self._determine_benchmark_level(analysis.total_score, comparisons)
        
        return {
            'analysis': analysis,
            'benchmark_level': benchmark_level,
            'comparisons': comparisons,
            'overall_assessment': self._generate_assessment(analysis, benchmark_level)
        }
    
    def _calculate_percentile(self, score: float, reference_scores: List[float]) -> float:
        """Calculate percentile ranking against reference scores."""
        if not reference_scores:
            return 50.0
        
        below_count = sum(1 for ref_score in reference_scores if ref_score < score)
        percentile = (below_count / len(reference_scores)) * 100
        return percentile
    
    def _determine_benchmark_level(self, score: float, comparisons: Dict[str, Any]) -> str:
        """Determine which benchmark level the text best matches."""
        closest_level = "medium_creativity"
        min_difference = float('inf')
        
        for level, comparison in comparisons.items():
            abs_diff = abs(comparison['difference'])
            if abs_diff < min_difference:
                min_difference = abs_diff
                closest_level = level
        
        return closest_level
    
    def _generate_assessment(self, analysis: CreativityAnalysis, benchmark_level: str) -> str:
        """Generate textual assessment of the creativity level."""
        score = analysis.total_score
        
        assessments = {
            'high_creativity': f"Exceptional creativity (score: {score:.2f}). Text demonstrates sophisticated language use, diverse vocabulary, and original expression comparable to highly creative reference texts.",
            'medium_creativity': f"Good creativity (score: {score:.2f}). Text shows creative elements with room for enhancement in vocabulary diversity and originality.",
            'low_creativity': f"Basic creativity (score: {score:.2f}). Text uses conventional language with limited diversity. Consider incorporating more varied vocabulary and creative expression."
        }
        
        return assessments.get(benchmark_level, f"Creativity score: {score:.2f}")