"""Shared text preprocessing for the language-detection model.

Keeping the cleaning logic in one place guarantees that the text seen at
training time is byte-for-byte identical to the text seen at inference time.
A mismatch here is the classic cause of a model that scores great offline
but behaves strangely in production.
"""

from __future__ import annotations

import re

#: Punctuation, digits and bracket characters stripped from every input.
#: (Brackets are escaped so the character class actually matches them.)
NOISE_PATTERN = re.compile(r'[!@#$(),\n"%^*?\:;~`0-9\[\]]')


def clean_text(text: str) -> str:
    """Normalise ``text`` the same way for training and inference.

    Removes punctuation, digits and bracket characters, then lowercases the
    result.  Whitespace is collapsed by the vectorizer itself.

    Args:
        text: Raw input text in any language.

    Returns:
        The cleaned, lowercased text.

    Example:
        >>> clean_text("Ciao, come stai? [test] 123")
        'ciao  come stai   test   '
    """
    return NOISE_PATTERN.sub(" ", text).lower()
