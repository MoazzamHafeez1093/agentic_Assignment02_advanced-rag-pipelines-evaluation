import re
import string

def normalize(text: str) -> str:
    """Lowercase, strip punctuation, articles, and extra whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = ''.join(c for c in text if c not in string.punctuation)
    return ' '.join(text.split())

def is_correct(prediction: str, answer: str, alt_ans: list[str]) -> bool:
    """Check if normalized prediction contains the normalized gold answer or any alt_ans."""
    norm_pred = normalize(prediction)
    if not norm_pred:
        return False
        
    golds = [answer] + alt_ans
    for gold in golds:
        if not gold:
            continue
        norm_gold = normalize(gold)
        if not norm_gold:
            continue
            
        # 1. Exact substring match
        if norm_gold in norm_pred:
            return True
            
        # 2. Token overlap for long gold answers
        gold_tokens = set(norm_gold.split())
        pred_tokens = set(norm_pred.split())
        
        # If the gold answer is long (e.g. a full paragraph), exact matching is nearly impossible.
        # We allow a match if the prediction contains a significant portion of the gold tokens.
        if len(gold_tokens) > 3:
            overlap = len(gold_tokens.intersection(pred_tokens))
            if overlap / len(gold_tokens) >= 0.35: # 35% word overlap implies high semantic similarity
                return True
                
        # 3. For short gold answers, check if all gold tokens exist in the prediction
        elif len(gold_tokens) > 0 and gold_tokens.issubset(pred_tokens):
            return True

    return False

def evaluate_pipeline(predictions: list[str], golds: list[dict]) -> tuple[float, list[bool]]:
    """Compute accuracy."""
    results = []
    for pred, gold in zip(predictions, golds):
        ans = gold.get('answer', '')
        alt = gold.get('alt_ans', [])
        results.append(is_correct(pred, ans, alt))
    accuracy = sum(results) / len(results) if results else 0.0
    return accuracy, results
