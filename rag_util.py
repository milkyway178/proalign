"""Lightweight anonymized RAG retrieval utility for dialogue KT."""
import pandas as pd


class RAGRetriever:
    """Retrieve few-shot examples by student-profile similarity.

    The public version keeps the interface. The exact similarity features and
selection heuristics used in experiments are simplified to avoid copy-ready reuse.
    """
    def __init__(self, train_path, encoding="utf-8"):
        self.df = pd.read_csv(train_path, encoding=encoding)

    def retrieve_examples(self, profile, k=3, max_chars=4000):
        if "student_profile" not in self.df.columns:
            return ""
        candidates = self.df[self.df["student_profile"].astype(str) == str(profile)].head(k)
        blocks = []
        for i, (_, row) in enumerate(candidates.iterrows(), 1):
            text = str(row.get("conversation", ""))[:max_chars]
            blocks.append(f"--- Example {i} ---\nDialogue History: {text}\n")
        return "\n".join(blocks)
