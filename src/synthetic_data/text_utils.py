"""
SMS text processing utilities.

Functions for cleaning, validating, and analyzing SMS messages.
"""

import re
from collections import Counter


# =============================================================================
# Message Cleaning
# =============================================================================

def clean_message(text: str) -> str:
    """Remove artifacts and fix common LLM generation issues."""
    # Remove formal sign-offs
    text = re.sub(
        r'(?:Best regards|Regards|Warm regards|Sincerely|With (?:love|gratitude|thanks)|Cheers),?\s*[-–—]?\s*\w+.*$',
        '', text, flags=re.IGNORECASE
    )
    # Remove standalone signature at end (e.g., "- Cyril" or "–Name")
    text = re.sub(r'\s*[-–—]\s*\w+\s*$', '', text)
    # Remove "Name" placeholder
    text = re.sub(r'\bName\b', '', text)
    # Remove [Friend's Name] type placeholders
    text = re.sub(r'\[.*?(?:Name|name).*?\]', '', text)
    # Remove "Dear X," or "Hi X," greetings at start
    text = re.sub(r'^(?:Dear|Hi|Hello|Hey)\s+[\w\s]+,\s*', '', text, flags=re.IGNORECASE)
    # Remove non-ASCII characters (Chinese chars, etc.) but keep emojis
    text = ''.join(c for c in text if ord(c) < 128 or ord(c) > 0x1F600)
    # Clean up multiple spaces and trim
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# =============================================================================
# Quality Checks
# =============================================================================

def check_message_quality(text: str, history: list[dict], rel_type: str) -> tuple[bool, str]:
    """
    Check if a generated message is acceptable.
    
    Args:
        text: The generated message text
        history: List of previous messages in the conversation
        rel_type: Relationship type for context
        
    Returns:
        (is_valid, issue_reason) tuple
    """
    text_lower = text.lower()
    
    # Too long for SMS
    if len(text.split()) > 35:
        return False, "too_long"
    
    # Too short / empty
    if len(text.split()) < 2:
        return False, "too_short"
    
    # Formal language in casual relationships
    if rel_type in ("partner", "close_friends", "friends", "family"):
        formal = ["best regards", "sincerely", "warm regards", "respectfully"]
        if any(f in text_lower for f in formal):
            return False, "too_formal"
    
    # Check similarity to last message
    if history:
        last_text = history[-1]["text"].lower()
        
        # Extract 3-word phrases
        text_phrases = _get_phrases(text_lower)
        last_phrases = _get_phrases(last_text)
        
        if text_phrases and last_phrases:
            overlap = len(text_phrases & last_phrases)
            if overlap >= 2:
                return False, "too_similar"
        
        # Check for repeated time mentions
        time_pattern = r'(?:at|in|around)\s+\d+'
        text_times = re.findall(time_pattern, text_lower)
        last_times = re.findall(time_pattern, last_text)
        if text_times and text_times == last_times:
            return False, "repeated_time"
    
    return True, ""


def _get_phrases(text: str, n: int = 3) -> set[str]:
    """Extract n-word phrases from text."""
    words = text.split()
    return set(' '.join(words[i:i+n]) for i in range(len(words) - n + 1))


def extract_key_phrases(text: str) -> set[str]:
    """Extract meaningful phrases from a message for comparison."""
    text = text.lower()
    # Extract time references like "in 30", "at 5", "home in 30"
    phrases = set(re.findall(r'(?:in|at|around)\s+\d+', text))
    # Extract common short phrases (3+ words)
    words = text.split()
    for i in range(len(words) - 2):
        phrases.add(' '.join(words[i:i+3]))
    return phrases


# =============================================================================
# Conversation Control
# =============================================================================

def should_end_conversation(history: list[dict]) -> bool:
    """Detect if conversation has naturally ended to avoid repetitive messages."""
    if len(history) < 3:
        return False
    
    last_msgs = [m['text'].lower() for m in history[-4:]]
    
    # Check for goodbye patterns
    bye_patterns = ['bye', 'see you', 'see ya', 'ttyl', 'later', 'talk soon', 'cya', 'gotta go']
    bye_count = sum(1 for msg in last_msgs if any(p in msg for p in bye_patterns))
    if bye_count >= 2:
        return True
    
    # Check for repeated confirmations
    confirm_patterns = ['sounds good', 'perfect', 'got it', 'will do', 'see you then', '👍', '👌']
    confirm_count = sum(1 for msg in last_msgs if any(p in msg for p in confirm_patterns))
    if confirm_count >= 2:
        return True
    
    # Check for time/place being confirmed multiple times
    recent_text = ' '.join(last_msgs)
    time_mentions = len(re.findall(r'\d+(?::\d+)?\s*(?:am|pm|AM|PM)', recent_text))
    if time_mentions >= 3:
        return True
    
    # Check for repeated key phrases
    if len(history) >= 3:
        all_phrases = []
        for msg in history[-4:]:
            all_phrases.extend(extract_key_phrases(msg['text']))
        phrase_counts = Counter(all_phrases)
        if any(count >= 3 for phrase, count in phrase_counts.items() if len(phrase) > 5):
            return True
    
    # Check for high word overlap between recent messages
    if len(history) >= 4:
        recent_words = [_get_content_words(m['text']) for m in history[-4:]]
        for i in range(len(recent_words) - 1):
            if recent_words[i] and recent_words[i+1]:
                overlap = len(recent_words[i] & recent_words[i+1]) / max(len(recent_words[i] | recent_words[i+1]), 1)
                if overlap > 0.6 and i >= 1:
                    prev_overlap = len(recent_words[i-1] & recent_words[i]) / max(len(recent_words[i-1] | recent_words[i]), 1)
                    if prev_overlap > 0.5:
                        return True
    
    return False


def _get_content_words(text: str) -> set[str]:
    """Extract content words (excluding stopwords) from text."""
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'to', 'in', 'on', 'at', 'and', 'or', 'i', 'you', 'your', 'my'}
    return set(w for w in text.lower().split() if w not in stopwords and len(w) > 2)


# =============================================================================
# SMS Style
# =============================================================================

def infer_sms_style(persona: dict) -> str:
    """Infer texting style from persona demographics."""
    age = persona.get("age", 40)
    education = persona.get("education_level", "")
    
    traits = []
    
    # Age-based patterns
    if age < 25:
        traits.extend(["lowercase", "heavy emoji use", "abbreviations (u, ur, rn, ngl)", "no punctuation"])
    elif age < 40:
        traits.extend(["casual", "occasional emoji", "some abbreviations"])
    elif age < 60:
        traits.extend(["proper sentences", "minimal emoji", "full words"])
    else:
        traits.extend(["formal", "complete sentences", "no emoji", "may sign off with name"])
    
    # Education modifier
    if "doctorate" in education.lower() or "masters" in education.lower():
        traits.append("articulate vocabulary")
    
    return ", ".join(traits)
