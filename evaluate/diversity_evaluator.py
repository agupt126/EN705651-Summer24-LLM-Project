import torch
from sacrebleu import corpus_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

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
    
    def calculate_self_bleu(self,generated_texts):
        scores = []
        num_texts = len(generated_texts)
        
        for i in range(num_texts):
             # Hypothesis is the current text
            hypothesis = generated_texts[i]
            
            # References are all other texts
            references = [generated_texts[j] for j in range(num_texts) if j != i]
            
            # sacrebleu expects references to be a list of lists of strings
            bleu_score = corpus_bleu([hypothesis], [[ref] for ref in references])
            scores.append(bleu_score.score)
        
        # Average the scores
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
        total_tokens = 0
        unique_ngrams = set()
        for text in generated_texts:
            tokens = text.split()
            total_tokens += len(tokens)
            for i in range(len(tokens) - self.gram + 1):
                ngram = " ".join(tokens[i:i+self.gram])
                unique_ngrams.add(ngram)
        
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
        score =  self.calculate_self_bleu(generated_texts)
        scores["self_bleu"] = score

        # Calculate Distinct-n score
        distinct_n_score = self.calculate_distinct_n(generated_texts)
        scores["distinct_n"] = distinct_n_score
    
      

        return scores




# generated_texts = [
#     "the quick brown fox jumps over the lazy dog",
#     "the cat sat on the mat"
# ]

# reference_texts = [
#     "the quick brown fox jumped over the lazy dog",
#     "the cat is sitting on the mat"
# ]

# evaluator = TextQualityEvaluator()

# # Calculate all metrics
# scores = evaluator.evaluate(generated_texts)
# print(scores)

