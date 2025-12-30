"""
Curriculum learning utilities for base pretraining.

This module provides Flesch Reading Ease score computation for
implementing curriculum learning (easy text first, harder text later).
"""

import re


def count_syllables(word: str) -> int:
    """
    Approximate syllable count for a word using vowel group counting.

    This is a simple heuristic that works reasonably well for English text.
    It counts vowel groups and applies common corrections.
    """
    word = word.lower().strip()
    if not word:
        return 0

    # Count vowel groups (consecutive vowels count as one syllable)
    vowels = "aeiouy"
    count = 0
    prev_was_vowel = False

    for char in word:
        is_vowel = char in vowels
        if is_vowel and not prev_was_vowel:
            count += 1
        prev_was_vowel = is_vowel

    # Apply common corrections
    # Silent 'e' at end of word
    if word.endswith('e') and count > 1:
        count -= 1

    # Words ending in 'le' preceded by consonant add a syllable
    if len(word) >= 2 and word.endswith('le') and len(word) > 2 and word[-3] not in vowels:
        count += 1

    # Ensure at least one syllable for non-empty words
    return max(1, count)


def count_syllables_text(text: str) -> int:
    """Count total syllables in a text."""
    words = re.findall(r'[a-zA-Z]+', text)
    return sum(count_syllables(word) for word in words)


def flesch_reading_ease(text: str) -> float:
    """
    Calculate Flesch Reading Ease score for a text.

    The Flesch Reading Ease formula:
    206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)

    Score interpretation:
    - 90-100: Very easy (5th grade)
    - 80-90: Easy (6th grade)
    - 70-80: Fairly easy (7th grade)
    - 60-70: Standard (8th-9th grade)
    - 50-60: Fairly difficult (10th-12th grade)
    - 30-50: Difficult (college)
    - 0-30: Very difficult (college graduate)

    Higher scores = easier to read.

    Args:
        text: The text to analyze

    Returns:
        Flesch Reading Ease score, clamped to [0, 100]
    """
    if not text or not text.strip():
        return 100.0  # Empty text treated as easiest

    # Count sentences (end with . ! or ?)
    sentences = len(re.findall(r'[.!?]+', text))
    if sentences == 0:
        sentences = 1  # Treat as one sentence if no punctuation

    # Count words
    words = re.findall(r'[a-zA-Z]+', text)
    num_words = len(words)

    if num_words == 0:
        return 100.0  # No words = easiest

    # Count syllables
    syllables = sum(count_syllables(word) for word in words)

    # Calculate Flesch Reading Ease
    avg_sentence_length = num_words / sentences
    avg_syllables_per_word = syllables / num_words

    score = 206.835 - 1.015 * avg_sentence_length - 84.6 * avg_syllables_per_word

    # Clamp to valid range
    return max(0.0, min(100.0, score))


def flesch_reading_ease_batch(texts: list[str]) -> list[float]:
    """
    Calculate Flesch Reading Ease scores for a batch of texts.

    Args:
        texts: List of texts to analyze

    Returns:
        List of Flesch Reading Ease scores
    """
    return [flesch_reading_ease(text) for text in texts]


def assign_tier(score: float, num_tiers: int = 10) -> int:
    """
    Assign a tier based on Flesch score.

    Tier 0 = easiest (highest scores), tier (num_tiers-1) = hardest (lowest scores).

    Args:
        score: Flesch Reading Ease score (0-100)
        num_tiers: Number of tiers to divide into

    Returns:
        Tier index (0 = easiest, num_tiers-1 = hardest)
    """
    # Divide 0-100 range into num_tiers buckets
    # Higher score = easier = lower tier number
    tier_size = 100.0 / num_tiers
    tier = int((100.0 - score) / tier_size)
    return min(tier, num_tiers - 1)  # Clamp to valid range
