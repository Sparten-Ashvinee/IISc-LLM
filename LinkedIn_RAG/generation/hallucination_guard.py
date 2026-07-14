"""NLI-based hallucination guard — checks if generated claims are supported by context."""

import re
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class HallucinationGuard:
    def __init__(self, model_name: str = "microsoft/deberta-v3-base",
                 entailment_threshold: float = 0.7):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        self.threshold = entailment_threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        # DeBERTa NLI label mapping: 0=contradiction, 1=neutral, 2=entailment
        self.labels = ["contradiction", "neutral", "entailment"]

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if len(s.strip()) > 10]

    def check(self, answer: str, context_chunks: list[dict]) -> dict:
        """Check each sentence in the answer against retrieved context.

        Returns:
            dict with 'is_faithful', 'score', 'flagged_sentences'
        """
        # Combine context
        context = "\n".join(c["text"][:500] for c in context_chunks)
        sentences = self._split_sentences(answer)

        if not sentences:
            return {"is_faithful": True, "score": 1.0, "flagged_sentences": []}

        flagged = []
        entailment_scores = []

        for sentence in sentences:
            inputs = self.tokenizer(
                context, sentence,
                return_tensors="pt", truncation=True, max_length=512,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1)[0]

            entailment_prob = probs[2].item()  # entailment class
            contradiction_prob = probs[0].item()
            entailment_scores.append(entailment_prob)

            if entailment_prob < self.threshold:
                flagged.append({
                    "sentence": sentence,
                    "entailment_prob": round(entailment_prob, 3),
                    "contradiction_prob": round(contradiction_prob, 3),
                    "verdict": self.labels[probs.argmax().item()],
                })

        avg_score = sum(entailment_scores) / len(entailment_scores) if entailment_scores else 0.0

        return {
            "is_faithful": len(flagged) == 0,
            "score": round(avg_score, 3),
            "flagged_sentences": flagged,
        }
