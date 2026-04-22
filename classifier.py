"""
Naive Bayes Spam Classifier — built from scratch, no sklearn.

Math:
  P(spam | words) ∝ P(spam) * ∏ P(word | spam)
  Using log-probabilities to avoid underflow.
  Laplace smoothing to handle unseen words.
"""

import re
import math
import csv
import os
from collections import defaultdict
from typing import Tuple


# ── Preprocessing ──────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """Lowercase, strip punctuation, split into words."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [w for w in text.split() if len(w) > 1]


STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "is", "it", "i", "you", "we", "my", "your", "this", "that",
    "are", "was", "be", "have", "has", "do", "did", "will", "not", "no",
    "so", "up", "out", "if", "as", "me", "he", "she", "they", "from"
}

def clean_tokens(tokens: list[str]) -> list[str]:
    return [t for t in tokens if t not in STOPWORDS]


# ── Naive Bayes Classifier ─────────────────────────────────────────────────────

class NaiveBayesClassifier:
    def __init__(self, alpha: float = 1.0):
        """
        alpha: Laplace smoothing parameter.
               alpha=1 is standard (add-one smoothing).
        """
        self.alpha = alpha
        self.log_prior: dict[str, float] = {}
        self.log_likelihood: dict[str, dict[str, float]] = {}
        self.vocab: set[str] = set()
        self.classes: list[str] = []

    def fit(self, X: list[list[str]], y: list[str]) -> "NaiveBayesClassifier":
        """Train on tokenized documents."""
        n = len(y)
        self.classes = list(set(y))

        # Count word occurrences per class
        word_counts: dict[str, defaultdict] = {c: defaultdict(int) for c in self.classes}
        class_counts: dict[str, int] = defaultdict(int)

        for tokens, label in zip(X, y):
            class_counts[label] += 1
            for word in tokens:
                word_counts[label][word] += 1
                self.vocab.add(word)

        vocab_size = len(self.vocab)

        # Log prior: log P(class)
        for c in self.classes:
            self.log_prior[c] = math.log(class_counts[c] / n)

        # Log likelihood: log P(word | class) with Laplace smoothing
        self.log_likelihood = {}
        for c in self.classes:
            total_words = sum(word_counts[c].values())
            self.log_likelihood[c] = {}
            for word in self.vocab:
                count = word_counts[c].get(word, 0)
                self.log_likelihood[c][word] = math.log(
                    (count + self.alpha) / (total_words + self.alpha * vocab_size)
                )
            # Score for unknown words (unseen in vocab)
            self.log_likelihood[c]["__UNK__"] = math.log(
                self.alpha / (total_words + self.alpha * vocab_size)
            )

        return self

    def predict_proba(self, tokens: list[str]) -> dict[str, float]:
        """Return log-probability scores per class."""
        scores = {}
        for c in self.classes:
            score = self.log_prior[c]
            for word in tokens:
                score += self.log_likelihood[c].get(
                    word, self.log_likelihood[c]["__UNK__"]
                )
            scores[c] = score
        return scores

    def predict(self, tokens: list[str]) -> str:
        scores = self.predict_proba(tokens)
        return max(scores, key=scores.get)

    def predict_with_confidence(self, tokens: list[str]) -> Tuple[str, float]:
        """Returns (label, confidence%) using softmax over log-scores."""
        scores = self.predict_proba(tokens)
        # Softmax to get probabilities
        max_score = max(scores.values())
        exp_scores = {c: math.exp(s - max_score) for c, s in scores.items()}
        total = sum(exp_scores.values())
        probs = {c: v / total for c, v in exp_scores.items()}
        label = max(probs, key=probs.get)
        return label, probs[label] * 100


# ── Dataset Loader ─────────────────────────────────────────────────────────────

def load_sms_spam(filepath: str) -> Tuple[list[str], list[str]]:
    """
    Load the SMS Spam Collection dataset.
    Format: tab-separated, col0=label (ham/spam), col1=message
    Download from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
    """
    texts, labels = [], []
    with open(filepath, encoding="latin-1") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) >= 2:
                labels.append(row[0].strip())
                texts.append(row[1].strip())
    return texts, labels


def train_test_split(X, y, test_ratio=0.2, seed=42):
    """Simple deterministic split."""
    import random
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_ratio))
    train_idx, test_idx = indices[:split], indices[split:]
    return (
        [X[i] for i in train_idx], [X[i] for i in test_idx],
        [y[i] for i in train_idx], [y[i] for i in test_idx]
    )


# ── Evaluation ─────────────────────────────────────────────────────────────────

def evaluate(clf, X_test, y_test):
    """Compute accuracy, precision, recall, F1 for spam class."""
    tp = fp = tn = fn = 0
    for tokens, true_label in zip(X_test, y_test):
        pred = clf.predict(tokens)
        if pred == "spam" and true_label == "spam": tp += 1
        elif pred == "spam" and true_label == "ham":  fp += 1
        elif pred == "ham"  and true_label == "ham":  tn += 1
        elif pred == "ham"  and true_label == "spam": fn += 1

    accuracy  = (tp + tn) / (tp + tn + fp + fn) if (tp+tn+fp+fn) else 0
    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {
        "accuracy":  round(accuracy  * 100, 2),
        "precision": round(precision * 100, 2),
        "recall":    round(recall    * 100, 2),
        "f1":        round(f1        * 100, 2),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn}
    }


def top_spam_words(clf, n=20) -> list[Tuple[str, float]]:
    """Words most associated with spam (log-likelihood ratio)."""
    if "spam" not in clf.log_likelihood or "ham" not in clf.log_likelihood:
        return []
    scores = []
    for word in clf.vocab:
        ratio = clf.log_likelihood["spam"][word] - clf.log_likelihood["ham"][word]
        scores.append((word, ratio))
    return sorted(scores, key=lambda x: x[1], reverse=True)[:n]


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    DATA_PATH = "data/SMSSpamCollection"

    print("=" * 60)
    print("  Naive Bayes Spam Classifier — built from scratch")
    print("=" * 60)

    if not os.path.exists(DATA_PATH):
        print(f"\n[!] Dataset not found at '{DATA_PATH}'")
        print("    Download from: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection")
        print("    Place the 'SMSSpamCollection' file in the data/ folder.\n")
        print("    Running demo mode with synthetic data...\n")
        _demo_mode()
        return

    # Load & preprocess
    print("\n[1] Loading dataset...")
    texts, labels = load_sms_spam(DATA_PATH)
    print(f"    {len(texts)} messages loaded ({labels.count('spam')} spam, {labels.count('ham')} ham)")

    print("[2] Tokenizing...")
    X = [clean_tokens(tokenize(t)) for t in texts]

    print("[3] Splitting (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_ratio=0.2)
    print(f"    Train: {len(X_train)}, Test: {len(X_test)}")

    print("[4] Training Naive Bayes classifier...")
    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(X_train, y_train)
    print(f"    Vocabulary size: {len(clf.vocab):,} words")

    print("[5] Evaluating on test set...")
    metrics = evaluate(clf, X_test, y_test)
    print(f"\n  Accuracy:  {metrics['accuracy']}%")
    print(f"  Precision: {metrics['precision']}%  (of predicted spam, how many were spam)")
    print(f"  Recall:    {metrics['recall']}%   (of actual spam, how many caught)")
    print(f"  F1 Score:  {metrics['f1']}%")
    print(f"\n  Confusion Matrix:")
    c = metrics['confusion']
    print(f"  ┌─────────────────────────────┐")
    print(f"  │           Predicted          │")
    print(f"  │        Spam    │    Ham      │")
    print(f"  │ Spam   {c['tp']:5}   │  {c['fn']:5}      │ ← Actual")
    print(f"  │ Ham    {c['fp']:5}   │  {c['tn']:5}      │")
    print(f"  └─────────────────────────────┘")

    print("\n[6] Top 15 spam indicator words:")
    for word, score in top_spam_words(clf, 15):
        print(f"  {word:<20} log-ratio: {score:.3f}")

    # Interactive
    print("\n" + "=" * 60)
    print("  Interactive classifier — type a message to classify")
    print("  (Ctrl+C to exit)")
    print("=" * 60)
    while True:
        try:
            msg = input("\n> ").strip()
            if not msg:
                continue
            tokens = clean_tokens(tokenize(msg))
            label, confidence = clf.predict_with_confidence(tokens)
            icon = "🚫 SPAM" if label == "spam" else "✅ HAM"
            print(f"  {icon}  ({confidence:.1f}% confidence)")
        except KeyboardInterrupt:
            print("\n\nFarewell.\n")
            break


def _demo_mode():
    """Run with tiny synthetic data when real dataset not found."""
    samples = [
        ("ham",  "Hey, are you free for lunch tomorrow?"),
        ("ham",  "I will be late tonight, don't wait up"),
        ("ham",  "Can you pick up some milk on your way home?"),
        ("spam", "WINNER! You have been selected for a FREE prize. Call now!"),
        ("spam", "Congratulations! Claim your free iPhone. Click here to win cash"),
        ("spam", "FREE entry to win £1000. Text WIN to 80085 now"),
        ("ham",  "Meeting moved to 3pm, see you there"),
        ("spam", "Urgent: your account has been compromised. Verify now at link"),
        ("ham",  "Thanks for calling me back"),
        ("spam", "You are a lucky winner of our grand prize lottery"),
    ]
    labels = [s[0] for s in samples]
    texts  = [s[1] for s in samples]
    X = [clean_tokens(tokenize(t)) for t in texts]

    clf = NaiveBayesClassifier(alpha=1.0)
    clf.fit(X, labels)

    test_msgs = [
        "Click here to claim your free gift now",
        "Are we still on for dinner tonight?",
        "You won a prize, text back immediately",
    ]
    print("Demo predictions:")
    for msg in test_msgs:
        tokens = clean_tokens(tokenize(msg))
        label, conf = clf.predict_with_confidence(tokens)
        icon = "🚫 SPAM" if label == "spam" else "✅ HAM"
        print(f"  '{msg[:50]}...' → {icon} ({conf:.1f}%)")


if __name__ == "__main__":
    main()
