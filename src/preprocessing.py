"""
Text preprocessing utilities for medical report classification.
"""

import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def ensure_nltk_data() -> None:
    """Download required NLTK resources if not present."""
    resources = ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]
    for resource in resources:
        try:
            nltk.data.find(
                f"corpora/{resource}" if resource in {"stopwords", "wordnet", "omw-1.4"} else f"tokenizers/{resource}"
            )
        except LookupError:
            nltk.download(resource, quiet=True)


class TextPreprocessor:
    """Preprocess medical report text for classification."""

    def __init__(self) -> None:
        ensure_nltk_data()
        self._stop_words = set(stopwords.words("english"))
        self._lemmatizer = WordNetLemmatizer()

    def clean_text(self, text: str) -> str:
        """Apply full preprocessing pipeline to a single report."""
        if not isinstance(text, str):
            return ""

        text = text.lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\d+", " ", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        tokens = nltk.word_tokenize(text)
        tokens = [
            self._lemmatizer.lemmatize(token)
            for token in tokens
            if token not in self._stop_words and len(token) > 2
        ]
        return " ".join(tokens)

    def transform(self, texts: list[str]) -> list[str]:
        """Preprocess a batch of reports."""
        return [self.clean_text(text) for text in texts]
