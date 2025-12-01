"""Tests for dataset processing and spam distribution."""

import pytest
import numpy as np

from src.task import distribute_spam


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_spam_messages():
    """Sample spam messages for testing."""
    return [
        {"text": "You won $1000! Click here", "label": 1},
        {"text": "Free iPhone! Claim now", "label": 1},
        {"text": "Congratulations! You've been selected", "label": 1},
        {"text": "Urgent: Your account is compromised", "label": 1},
        {"text": "Win a free vacation", "label": 1},
        {"text": "Limited time offer!", "label": 1},
        {"text": "Click here for free money", "label": 1},
        {"text": "You're a winner!", "label": 1},
        {"text": "Act now! Don't miss out", "label": 1},
        {"text": "Special deal just for you", "label": 1},
    ]


@pytest.fixture
def sample_persona_ids():
    """Sample persona UUIDs for testing."""
    return [
        "persona-001",
        "persona-002",
        "persona-003",
        "persona-004",
        "persona-005",
    ]


# =============================================================================
# Tests for distribute_spam - IID strategy
# =============================================================================

class TestDistributeSpamIID:
    """Tests for IID spam distribution strategy."""

    def test_iid_distributes_to_all_personas(self, sample_spam_messages, sample_persona_ids):
        """All personas should receive spam messages."""
        result = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        assert set(result.keys()) == set(sample_persona_ids)

    def test_iid_distributes_all_messages(self, sample_spam_messages, sample_persona_ids):
        """All spam messages should be distributed (no loss)."""
        result = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        total_distributed = sum(len(msgs) for msgs in result.values())
        assert total_distributed == len(sample_spam_messages)

    def test_iid_roughly_equal_distribution(self, sample_spam_messages, sample_persona_ids):
        """IID should distribute messages roughly equally."""
        result = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        counts = [len(msgs) for msgs in result.values()]
        # With 10 messages and 5 personas, each should get 2
        expected_per_persona = len(sample_spam_messages) // len(sample_persona_ids)
        
        for count in counts:
            # Allow for rounding (±1)
            assert abs(count - expected_per_persona) <= 1

    def test_iid_deterministic_with_seed(self, sample_spam_messages, sample_persona_ids):
        """Same seed should produce same distribution."""
        result1 = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        result2 = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        for pid in sample_persona_ids:
            assert len(result1[pid]) == len(result2[pid])
            for m1, m2 in zip(result1[pid], result2[pid]):
                assert m1["text"] == m2["text"]

    def test_iid_different_seeds_different_results(self, sample_spam_messages, sample_persona_ids):
        """Different seeds should produce different distributions."""
        result1 = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        result2 = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=123,
        )
        
        # At least one persona should have different messages
        different = False
        for pid in sample_persona_ids:
            if len(result1[pid]) != len(result2[pid]):
                different = True
                break
            texts1 = [m["text"] for m in result1[pid]]
            texts2 = [m["text"] for m in result2[pid]]
            if texts1 != texts2:
                different = True
                break
        
        assert different, "Different seeds should produce different distributions"


# =============================================================================
# Tests for distribute_spam - Dirichlet strategy
# =============================================================================

class TestDistributeSpamDirichlet:
    """Tests for Dirichlet (non-IID) spam distribution strategy."""

    def test_dirichlet_distributes_to_all_personas(self, sample_spam_messages, sample_persona_ids):
        """All personas should be in the result (though some may have 0 messages)."""
        result = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="dirichlet",
            alpha=0.5,
            seed=42,
        )
        
        assert set(result.keys()) == set(sample_persona_ids)

    def test_dirichlet_distributes_all_messages(self, sample_spam_messages, sample_persona_ids):
        """All spam messages should be distributed (no loss)."""
        result = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="dirichlet",
            alpha=0.5,
            seed=42,
        )
        
        total_distributed = sum(len(msgs) for msgs in result.values())
        assert total_distributed == len(sample_spam_messages)

    def test_dirichlet_low_alpha_creates_imbalance(self, sample_persona_ids):
        """Low alpha should create more imbalanced distribution."""
        # Use more messages to see the effect
        many_spam = [{"text": f"Spam {i}", "label": 1} for i in range(100)]
        
        result = distribute_spam(
            many_spam,
            sample_persona_ids,
            strategy="dirichlet",
            alpha=0.1,  # Very low = very non-IID
            seed=42,
        )
        
        counts = [len(msgs) for msgs in result.values()]
        # With low alpha, variance should be high
        variance = np.var(counts)
        
        # Compare to IID distribution
        iid_result = distribute_spam(
            many_spam,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        iid_counts = [len(msgs) for msgs in iid_result.values()]
        iid_variance = np.var(iid_counts)
        
        # Dirichlet with low alpha should have higher variance
        assert variance > iid_variance

    def test_dirichlet_high_alpha_approaches_iid(self, sample_persona_ids):
        """High alpha should approximate IID distribution."""
        many_spam = [{"text": f"Spam {i}", "label": 1} for i in range(100)]
        
        result = distribute_spam(
            many_spam,
            sample_persona_ids,
            strategy="dirichlet",
            alpha=100.0,  # Very high = approaches uniform
            seed=42,
        )
        
        counts = [len(msgs) for msgs in result.values()]
        expected = len(many_spam) / len(sample_persona_ids)
        
        # With high alpha, all counts should be close to expected
        for count in counts:
            assert abs(count - expected) < expected * 0.3  # Within 30%

    def test_dirichlet_deterministic_with_seed(self, sample_spam_messages, sample_persona_ids):
        """Same seed should produce same distribution."""
        result1 = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="dirichlet",
            alpha=0.5,
            seed=42,
        )
        result2 = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="dirichlet",
            alpha=0.5,
            seed=42,
        )
        
        for pid in sample_persona_ids:
            assert len(result1[pid]) == len(result2[pid])


# =============================================================================
# Tests for edge cases
# =============================================================================

class TestDistributeSpamEdgeCases:
    """Tests for edge cases in spam distribution."""

    def test_empty_spam_list(self, sample_persona_ids):
        """Empty spam list should return empty lists for all personas."""
        result = distribute_spam(
            [],
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        for pid in sample_persona_ids:
            assert result[pid] == []

    def test_single_spam_message(self, sample_persona_ids):
        """Single message should go to exactly one persona."""
        single_spam = [{"text": "Only spam", "label": 1}]
        
        result = distribute_spam(
            single_spam,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        total = sum(len(msgs) for msgs in result.values())
        assert total == 1
        
        # Exactly one persona should have the message
        non_empty = [pid for pid, msgs in result.items() if len(msgs) > 0]
        assert len(non_empty) == 1

    def test_single_persona(self, sample_spam_messages):
        """Single persona should get all spam."""
        single_persona = ["persona-only"]
        
        result = distribute_spam(
            sample_spam_messages,
            single_persona,
            strategy="iid",
            seed=42,
        )
        
        assert len(result["persona-only"]) == len(sample_spam_messages)

    def test_more_personas_than_spam(self):
        """More personas than spam messages - some get nothing."""
        few_spam = [{"text": "Spam 1", "label": 1}, {"text": "Spam 2", "label": 1}]
        many_personas = [f"persona-{i}" for i in range(10)]
        
        result = distribute_spam(
            few_spam,
            many_personas,
            strategy="iid",
            seed=42,
        )
        
        total = sum(len(msgs) for msgs in result.values())
        assert total == len(few_spam)

    def test_unknown_strategy_raises_error(self, sample_spam_messages, sample_persona_ids):
        """Unknown strategy should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown strategy"):
            distribute_spam(
                sample_spam_messages,
                sample_persona_ids,
                strategy="unknown",  # type: ignore
                seed=42,
            )


# =============================================================================
# Tests for message integrity
# =============================================================================

class TestMessageIntegrity:
    """Tests to ensure messages are not modified during distribution."""

    def test_messages_not_modified(self, sample_spam_messages, sample_persona_ids):
        """Original messages should not be modified."""
        original_texts = [m["text"] for m in sample_spam_messages]
        
        distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        # Check original list is unchanged
        for i, msg in enumerate(sample_spam_messages):
            assert msg["text"] == original_texts[i]

    def test_distributed_messages_reference_originals(self, sample_spam_messages, sample_persona_ids):
        """Distributed messages should be the same objects as originals."""
        result = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        # All distributed messages should be in the original list
        original_ids = {id(m) for m in sample_spam_messages}
        for msgs in result.values():
            for msg in msgs:
                assert id(msg) in original_ids

    def test_no_duplicate_distribution(self, sample_spam_messages, sample_persona_ids):
        """Each message should appear exactly once across all personas."""
        result = distribute_spam(
            sample_spam_messages,
            sample_persona_ids,
            strategy="iid",
            seed=42,
        )
        
        all_texts = []
        for msgs in result.values():
            all_texts.extend(m["text"] for m in msgs)
        
        # No duplicates
        assert len(all_texts) == len(set(all_texts))
        
        # All original messages present
        original_texts = {m["text"] for m in sample_spam_messages}
        assert set(all_texts) == original_texts
