"""Shared bypass sample logging for adversarial training."""

import json
import re
from pathlib import Path

# Task tag used during training
TASK_TAG = "<spam_sms>"

# URL patterns
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+|\S+\.(com|net|org|io|co|info|biz)\b", re.IGNORECASE)

# Phone patterns
PHONE_PATTERN = re.compile(
    r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b|"  # US: 123-456-7890
    r"\b\(\d{3}\)\s*\d{3}[-.\s]?\d{4}\b|"  # US: (123) 456-7890
    r"\b\+\d{1,3}[-.\s]?\d{6,12}\b|"  # International
    r"\b\d{5,6}\b"  # Short codes
, re.IGNORECASE)

# Short code / reply patterns
SHORT_CODE_PATTERN = re.compile(r"\b(text|reply|send|msg)\s+\w+\s+(to\s+)?\d{4,6}\b", re.IGNORECASE)
REPLY_KEYWORD_PATTERN = re.compile(r"\b(reply|text|send)\s+(stop|yes|no|help|info|start|join|quit)\b", re.IGNORECASE)

# Email pattern
EMAIL_PATTERN = re.compile(r"\b[\w.-]+@[\w.-]+\.\w+\b", re.IGNORECASE)

# Call-to-action pattern
CALL_ACTION_PATTERN = re.compile(r"\b(call|dial|ring|contact)\s*(us|me|now|back|today|this)?\b", re.IGNORECASE)

# Qwen3 thinking tags (full pattern to remove <think>...</think> blocks)
THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>\s*", re.IGNORECASE | re.DOTALL)
THINK_TAG_PATTERN = re.compile(r"</?think>\s*", re.IGNORECASE)

# Task tag pattern (to strip from responses)
TASK_TAG_PATTERN = re.compile(re.escape(TASK_TAG) + r"\s*", re.IGNORECASE)


class BypassLogger:
    """Logs SMS messages that bypass the spam detector.
    
    Used by both RewardFunctions and DetectorOnlyReward to track
    successful bypasses during training.
    """
    
    def __init__(self, log_path: str = "bypass_samples.json"):
        """Initialize bypass logger.
        
        Args:
            log_path: Path to save bypassing samples (JSON format)
        """
        self.log_path = Path(log_path)
        self.bypass_count = 0
        self.step_count = 0
        
        # Create parent directory if it doesn't exist
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize/clear the log file
        with open(self.log_path, "w") as f:
            f.write("[\n")  # Start JSON array
    
    def log_bypass(
        self,
        sms: str,
        ham_prob: float,
        spam_prob: float,
        reward: float,
        prompt: str = "",
        judge_score: float = None,
        metadata: dict = None,
    ):
        """Log a single bypass sample.
        
        Args:
            sms: The SMS message that bypassed the detector
            ham_prob: Probability of HAM classification
            spam_prob: Probability of SPAM classification
            reward: The reward score given
            prompt: The prompt used to generate the SMS
            judge_score: LLM judge score (0-1, where 1 = very deceptive spam)
            metadata: Optional metadata (objective, context, etc.)
        """
        sample = {
            "step": self.step_count,
            "sms": sms,
            "label": "spam",
            "prompt": prompt,
            "ham_prob": round(ham_prob, 4),
            "spam_prob": round(spam_prob, 4),
            "reward": round(reward, 4),
            "length": len(sms),
        }
        
        # Add judge score if provided
        if judge_score is not None:
            sample["judge_score"] = round(judge_score, 2)
        
        # Add metadata if provided
        if metadata and isinstance(metadata, dict):
            sample["objective"] = metadata.get("objective", "unknown")
            sample["context"] = metadata.get("context", "unknown")
            sample["has_link"] = metadata.get("has_link", False)
            sample["has_phone"] = metadata.get("has_phone", False)
        
        self._append_sample(sample)
        self.bypass_count += 1
    
    def _append_sample(self, sample: dict):
        """Append a sample to the JSON file."""
        with open(self.log_path, "a") as f:
            prefix = "" if self.bypass_count == 0 else ",\n"
            f.write(f"{prefix}  {json.dumps(sample)}")
    
    def finalize(self):
        """Close the JSON array. Call when training is complete."""
        with open(self.log_path, "a") as f:
            f.write("\n]")
        print(f"Finalized bypass log: {self.bypass_count} samples saved to {self.log_path}")
    
    def increment_step(self):
        """Increment step counter. Call at end of each training step."""
        self.step_count += 1
    
    def get_count(self) -> int:
        """Return total count of bypass samples."""
        return self.bypass_count


def clean_response(text: str) -> str:
    """Clean model output from artifacts like <think>...</think> blocks and task tags."""
    # Remove full thinking blocks first
    text = THINK_BLOCK_PATTERN.sub("", text)
    # Remove any remaining think tags
    text = THINK_TAG_PATTERN.sub("", text)
    # Remove task tag if echoed
    text = TASK_TAG_PATTERN.sub("", text)
    return text.strip()


def has_spam_payload(text: str) -> bool:
    """Check if message contains spam payload (URL, phone, etc.).
    
    Returns True if the message has a direct payload that makes it
    actionable spam (not just generic text).
    """
    # Check for garbage/invalid outputs
    if is_garbage(text):
        return False
    
    # URL patterns
    has_url = URL_PATTERN.search(text) is not None
    
    # Phone patterns
    has_phone = PHONE_PATTERN.search(text) is not None
    
    # Short code patterns
    has_short_code = SHORT_CODE_PATTERN.search(text) is not None
    has_reply_keyword = REPLY_KEYWORD_PATTERN.search(text) is not None
    
    # Email pattern
    has_email = EMAIL_PATTERN.search(text) is not None
    
    # Call-to-action pattern
    has_call_action = CALL_ACTION_PATTERN.search(text) is not None
    
    return has_url or has_phone or has_short_code or has_reply_keyword or has_email or has_call_action


def is_garbage(text: str) -> bool:
    """Check if text is garbage output (repetitive, non-ASCII noise, etc.)."""
    if len(text) < 10:
        return True
    
    # Too many non-ASCII chars (likely encoding garbage)
    ascii_ratio = sum(1 for c in text if ord(c) < 128) / max(len(text), 1)
    if ascii_ratio < 0.8:
        return True
    
    # Check for repetitive patterns (same 5+ char sequence repeated 3+ times)
    if len(text) >= 15:
        for i in range(len(text) - 15):
            chunk = text[i:i+5]
            if text.count(chunk) >= 3:
                # Might be intentional (like "...")
                if chunk.strip() and not all(c == chunk[0] for c in chunk):
                    continue
                if text.count(chunk) >= 5:  # Only flag extreme repetition
                    return True
    
    return False
