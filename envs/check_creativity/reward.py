import numpy as np
import nltk
from nltk.util import bigrams
from collections import Counter
import string
import math

nltk.download("punkt", quiet=True)
nltk.download("words", quiet=True)

from nltk.corpus import words as nltk_words

COMMON_WORDS = set(nltk_words.words())  # Can be replaced with a smaller 'common words' list if needed

def compute_entropy(probs):
    """Shannon entropy"""
    return -sum(p * math.log2(p) for p in probs if p > 0)

def reward_function(
    text: str,
    w_entropy=1.0,
    w_distinct=1.0,
    w_uncommon=1.0,
    w_bigrams=1.0,
    w_sentence_len_var=1.0,
    w_word_len_var=1.0,
    w_sentence_end_var=1.0,
):
    """
    Compute creativity reward for English text based on various diversity metrics.
    
    Args:
        text: Input text to evaluate
        w_entropy: Weight for word distribution entropy
        w_distinct: Weight for distinct word ratio
        w_uncommon: Weight for uncommon word usage
        w_bigrams: Weight for bigram diversity
        w_sentence_len_var: Weight for sentence length variance
        w_word_len_var: Weight for word length variance  
        w_sentence_end_var: Weight for sentence ending diversity
        
    Returns:
        Weighted reward score
    """
    # Preprocess
    sentences = nltk.sent_tokenize(text)
    words = nltk.word_tokenize(text)
    words = [w.lower() for w in words if w.isalpha()]  # filter out punctuation
    total_words = len(words)
    
    if total_words == 0 or len(sentences) == 0:
        return 0.0  # Avoid division by zero

    # Entropy of word distribution
    word_freqs = Counter(words)
    probs = [count / total_words for count in word_freqs.values()]
    entropy_score = compute_entropy(probs)

    # Distinct words / total words
    distinct_score = len(set(words)) / total_words

    # Uncommon word ratio
    uncommon_words = [w for w in words if w not in COMMON_WORDS]
    uncommon_score = len(uncommon_words) / total_words

    # Bigram diversity (unique bigrams / total bigrams)
    word_bigrams = list(bigrams(words))
    bigram_score = len(set(word_bigrams)) / (len(word_bigrams) + 1e-8)

    # Sentence length variance (number of words per sentence)
    sentence_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
    sentence_len_var_score = np.std(sentence_lengths)

    # Word length variance (letters per word)
    word_lengths = [len(w) for w in words]
    word_len_var_score = np.std(word_lengths)

    # Sentence endings diversity (punctuation usage)
    sentence_endings = [s.strip()[-1] if s.strip() else '.' for s in sentences]
    end_counts = Counter(sentence_endings)
    end_probs = [count / len(sentences) for count in end_counts.values()]
    sentence_end_var_score = compute_entropy(end_probs)

    # Weighted reward
    reward = (
        w_entropy * entropy_score +
        w_distinct * distinct_score +
        w_uncommon * uncommon_score +
        w_bigrams * bigram_score +
        w_sentence_len_var * sentence_len_var_score +
        w_word_len_var * word_len_var_score +
        w_sentence_end_var * sentence_end_var_score
    )

    return reward