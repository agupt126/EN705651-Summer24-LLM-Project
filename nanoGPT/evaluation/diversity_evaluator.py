import torch
from sacrebleu import corpus_bleu
from concurrent.futures import ThreadPoolExecutor
from nltk.util import ngrams
import numpy as np
from functools import lru_cache

class TextQualityEvaluator:
    """
    This class evaluates the quality of generated text using various metrics.
    """
    def __init__(self, gram: int = 4):
        """
        Args:
            gram: The n-gram size for Distinct-n calculation (default: 4).
        """
        self.gram = gram

    @lru_cache(None)  # Cache BLEU scores to avoid redundant calculations
    def _cached_bleu_score(self, hypothesis, references):
        bleu_score = corpus_bleu([hypothesis], [[ref] for ref in references], use_effective_order=True)
        return bleu_score.score

    def calculate_self_bleu(self, generated_texts):
        num_texts = len(generated_texts)
        
        def compute_bleu(i):
            hypothesis = generated_texts[i]
            references = [generated_texts[j] for j in range(num_texts) if j != i]
            return self._cached_bleu_score(hypothesis, tuple(references))
        
        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(compute_bleu, range(num_texts)))
        
        average_score = sum(scores) / len(scores) if scores else 0
        return average_score

    def calculate_distinct_n(self, generated_texts):
        """
        Calculates the Distinct-n score for a list of generated texts.
        
        Args:
            generated_texts: A list of strings representing the generated text samples.
        
        Returns:
            The proportion of unique n-grams.
        """
        unique_ngrams = set()
        total_tokens = 0
        
        for text in generated_texts:
            tokens = text.split()
            total_tokens += len(tokens)
            unique_ngrams.update(ngrams(tokens, self.gram))
        
        return len(unique_ngrams) / total_tokens if total_tokens > 0 else 0  # Avoid division by zero

    def evaluate(self, generated_texts):
        """
        Calculates various quality metrics for a list of generated texts.

        Args:
            generated_texts: A list of strings representing the generated text samples.

        Returns:
            A dictionary of scores for different metrics.
        """
        scores = {}

        # Calculate Self-BLEU score
        scores["self_bleu"] = self.calculate_self_bleu(generated_texts)

        # Calculate Distinct-n score
        scores["distinct_n"] = self.calculate_distinct_n(generated_texts)

        return scores

# Example usage:
# generated_texts = [
#     "the quick brown fox jumps over the lazy dog",
#     "the cat sat on the mat"
# ]
# evaluator = TextQualityEvaluator()
# scores = evaluator.evaluate(generated_texts)
# print(scores)
