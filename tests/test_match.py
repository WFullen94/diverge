"""Tests for answer matching strategies."""

import pytest
from diverge.match import exact_match, normalized_match, letter_match, semantic_match


# ---------------------------------------------------------------------------
# exact_match
# ---------------------------------------------------------------------------

def test_exact_match_same():
    assert exact_match("Paris", "Paris")

def test_exact_match_case_insensitive():
    assert exact_match("paris", "PARIS")

def test_exact_match_whitespace_stripped():
    assert exact_match("  Paris  ", "Paris")

def test_exact_match_different():
    assert not exact_match("Paris", "London")


# ---------------------------------------------------------------------------
# normalized_match
# ---------------------------------------------------------------------------

def test_normalized_strips_punctuation():
    assert normalized_match("Paris.", "Paris")

def test_normalized_strips_multiple_punct():
    assert normalized_match("Yes, I agree!", "yes i agree")

def test_normalized_collapses_whitespace():
    assert normalized_match("the  quick   brown", "the quick brown")

def test_normalized_different_words():
    assert not normalized_match("Paris", "London")

def test_normalized_same_after_norm():
    assert normalized_match("It's Paris!", "its paris")


# ---------------------------------------------------------------------------
# letter_match
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a,b,expected", [
    ("A", "A", True),
    ("A", "B", False),
    ("A)", "A.", True),
    ("The answer is A", "A", True),
    ("Answer: B", "b", True),
    ("(C) Paris", "C", True),
    ("D", "d", True),
    # Falls back to normalized_match when no letter found
    ("Paris", "paris", True),
    ("Paris", "London", False),
])
def test_letter_match_variants(a, b, expected):
    assert letter_match(a, b) == expected

def test_letter_match_mixed_case():
    assert letter_match("a) Paris", "A")

def test_letter_match_answer_phrase():
    assert letter_match("The correct answer is C.", "C")
    assert not letter_match("The correct answer is C.", "D")


# ---------------------------------------------------------------------------
# semantic_match (integration — requires sentence-transformers)
# ---------------------------------------------------------------------------

def test_semantic_match_identical():
    try:
        fn = semantic_match(threshold=0.9)
        assert fn("Paris is the capital of France.", "Paris is the capital of France.")
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_semantic_match_paraphrase():
    try:
        fn = semantic_match(threshold=0.7)
        assert fn("The capital of France is Paris.", "Paris is the capital of France.")
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_semantic_match_different_topics():
    try:
        fn = semantic_match(threshold=0.8)
        assert not fn("Paris is in France.", "Dogs are mammals.")
    except ImportError:
        pytest.skip("sentence-transformers not installed")


def test_semantic_match_threshold_respected():
    try:
        strict = semantic_match(threshold=0.999)
        loose = semantic_match(threshold=0.1)
        a = "The capital of France is Paris."
        b = "Paris is France's capital city."
        # loose should agree, strict may not
        assert loose(a, b)
    except ImportError:
        pytest.skip("sentence-transformers not installed")
