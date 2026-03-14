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
    golds = [answer] + alt_ans
    for gold in golds:
        if not gold:
            continue
        norm_gold = normalize(gold)
        if norm_gold and norm_gold in norm_pred:
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
