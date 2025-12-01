"""
SMS Conversation Quality Filter using distilabel Pipeline.
Run this AFTER generating conversations to filter and improve quality.
"""

import subprocess
import sys
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
                       "pydantic", "distilabel[vllm]"])

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Generator

from distilabel.models import vLLM
from distilabel.pipeline import Pipeline
from distilabel.steps import Step, step
from distilabel.steps.tasks import TextGeneration

# Paths
DATA_DIR = Path("data")
INPUT_FILE = DATA_DIR / "conversations.json"
OUTPUT_FILE = DATA_DIR / "conversations_filtered.json"

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"


# ------------------------------------------------------------------
# Quality Scoring (Rule-based - fast)
# ------------------------------------------------------------------

def compute_quality_score(conv: dict) -> tuple[float, list[str]]:
    """
    Compute quality score for a conversation using heuristics.
    Returns (score 0-1, list of issues).
    """
    issues = []
    score = 1.0
    
    messages = conv.get("messages", [])
    if len(messages) < 2:
        return 0.0, ["too_short"]
    
    rel_type = conv.get("relationship_type", "other")
    texts = [m["text"] for m in messages]
    
    # 1. Check message lengths
    avg_words = sum(len(t.split()) for t in texts) / len(texts)
    if avg_words > 30:
        score -= 0.25
        issues.append("messages_too_long")
    
    # 2. Check for formal language in casual relationships
    casual_rels = {"partner", "close_friends", "friends", "family"}
    if rel_type in casual_rels:
        formal_phrases = ["best regards", "sincerely", "warm regards", "respectfully", "dear"]
        for text in texts:
            if any(p in text.lower() for p in formal_phrases):
                score -= 0.2
                issues.append("too_formal_for_relationship")
                break
    
    # 3. Check for repetition
    all_phrases = []
    for text in texts:
        # Get 3-grams
        words = text.lower().split()
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            if len(phrase) > 8:
                all_phrases.append(phrase)
    
    phrase_counts = Counter(all_phrases)
    repeated = [p for p, c in phrase_counts.items() if c >= 3]
    if repeated:
        score -= 0.3
        issues.append(f"repetitive_phrases")
    
    # 4. Check for same-side repetition (same sender repeating themselves)
    for i in range(2, len(messages)):
        sender_i = messages[i].get("sender_uuid")
        sender_prev = messages[i-2].get("sender_uuid")
        if sender_i == sender_prev:
            text_i = set(texts[i].lower().split())
            text_prev = set(texts[i-2].lower().split())
            if text_i and text_prev:
                overlap = len(text_i & text_prev) / len(text_i | text_prev)
                if overlap > 0.6:
                    score -= 0.2
                    issues.append("sender_repeats_self")
                    break
    
    # 5. Check for natural conversation flow
    last_text = texts[-1].lower()
    good_endings = ["bye", "see you", "later", "ttyl", "sounds good", "ok", "perfect", "👍", "❤️", "lol", "haha"]
    has_natural_end = any(e in last_text for e in good_endings)
    
    # Also check if it's a reasonable last message even without explicit ending
    if not has_natural_end and len(messages) > 5:
        score -= 0.1
        issues.append("abrupt_ending")
    
    # 6. Check for placeholder artifacts
    for text in texts:
        if "Name" in text or "[" in text or "]" in text:
            score -= 0.15
            issues.append("placeholder_artifacts")
            break
    
    # 7. Check for non-ASCII garbage (but allow emojis)
    for text in texts:
        non_ascii = [c for c in text if ord(c) > 127 and not (0x1F300 <= ord(c) <= 0x1F9FF)]
        if len(non_ascii) > 3:
            score -= 0.1
            issues.append("encoding_issues")
            break
    
    return max(0.0, min(1.0, score)), issues


# ------------------------------------------------------------------
# LLM-based Quality Check (optional, more thorough)
# ------------------------------------------------------------------

class SMSQualityJudge(Step):
    """Use LLM to judge SMS conversation quality."""
    
    llm: Any = None
    
    @property
    def inputs(self) -> list[str]:
        return ["conversation"]
    
    @property
    def outputs(self) -> list[str]:
        return ["conversation", "llm_quality_score", "llm_feedback"]
    
    def process(self, inputs: list[dict]) -> Generator[list[dict], None, None]:
        batch_prompts = []
        
        for item in inputs:
            conv = item["conversation"]
            prompt = self._build_judge_prompt(conv)
            batch_prompts.append([[
                {"role": "system", "content": prompt["system"]},
                {"role": "user", "content": prompt["user"]}
            ]])
        
        # Batch generate
        flat_prompts = [p[0] for p in batch_prompts]
        results = self.llm.generate(flat_prompts)
        
        outputs = []
        for item, result in zip(inputs, results):
            text = result["generations"][0]
            score, feedback = self._parse_judge_response(text)
            outputs.append({
                **item,
                "llm_quality_score": score,
                "llm_feedback": feedback,
            })
        
        yield outputs
    
    def _build_judge_prompt(self, conv: dict) -> dict:
        rel_type = conv.get("relationship_type", "unknown")
        messages = conv.get("messages", [])
        
        conv_text = "\n".join(
            f"[{m.get('sender_uuid', '')[:8]}]: {m['text']}"
            for m in messages
        )
        
        system = """You are a quality judge for synthetic SMS conversations.
Rate the conversation on these criteria:
1. NATURALNESS: Does it sound like real text messages?
2. BREVITY: Are messages appropriately short for SMS?
3. NO REPETITION: Do people avoid repeating the same info?
4. APPROPRIATE TONE: Does the tone match the relationship?
5. COHERENT FLOW: Does the conversation make sense?

Respond with:
SCORE: [1-5]
ISSUES: [brief list or "none"]"""

        user = f"""Relationship type: {rel_type}

Conversation:
{conv_text}

Rate this conversation."""

        return {"system": system, "user": user}
    
    def _parse_judge_response(self, text: str) -> tuple[float, str]:
        # Extract score
        score_match = re.search(r'SCORE:\s*(\d)', text)
        score = int(score_match.group(1)) / 5.0 if score_match else 0.5
        
        # Extract issues
        issues_match = re.search(r'ISSUES:\s*(.+)', text, re.IGNORECASE)
        feedback = issues_match.group(1).strip() if issues_match else ""
        
        return score, feedback


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------

@step(inputs=[], outputs=["conversation"])
def LoadConversations(
    input_file: str = str(INPUT_FILE),
) -> Generator[list[dict], None, None]:
    """Load conversations from JSON file."""
    
    with open(input_file) as f:
        conversations = json.load(f)
    
    # Yield in batches
    batch_size = 32
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        yield [{"conversation": c} for c in batch]


class RuleBasedQualityFilter(Step):
    """Apply rule-based quality filtering."""
    
    min_score: float = 0.6
    
    @property
    def inputs(self) -> list[str]:
        return ["conversation"]
    
    @property
    def outputs(self) -> list[str]:
        return ["conversation", "quality_score", "quality_issues", "keep"]
    
    def process(self, inputs: list[dict]) -> Generator[list[dict], None, None]:
        results = []
        for item in inputs:
            score, issues = compute_quality_score(item["conversation"])
            results.append({
                **item,
                "quality_score": score,
                "quality_issues": issues,
                "keep": score >= self.min_score,
            })
        yield results


def run_quality_filter(
    input_file: Path = INPUT_FILE,
    output_file: Path = OUTPUT_FILE,
    min_score: float = 0.6,
    use_llm_judge: bool = False,
):
    """Run the quality filtering pipeline."""
    
    print(f"Loading conversations from {input_file}...")
    
    with open(input_file) as f:
        conversations = json.load(f)
    
    print(f"Loaded {len(conversations)} conversations")
    
    # Apply rule-based filtering
    kept = []
    filtered = []
    
    for conv in conversations:
        score, issues = compute_quality_score(conv)
        conv["quality_score"] = score
        conv["quality_issues"] = issues
        
        if score >= min_score:
            kept.append(conv)
        else:
            filtered.append(conv)
    
    print(f"\nQuality filtering results:")
    print(f"  Kept: {len(kept)} ({100*len(kept)/len(conversations):.1f}%)")
    print(f"  Filtered: {len(filtered)} ({100*len(filtered)/len(conversations):.1f}%)")
    
    # Show distribution of issues
    all_issues = []
    for conv in filtered:
        all_issues.extend(conv.get("quality_issues", []))
    
    if all_issues:
        print(f"\nTop issues in filtered conversations:")
        issue_counts = Counter(all_issues)
        for issue, count in issue_counts.most_common(5):
            print(f"  {issue}: {count}")
    
    # Save filtered conversations
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    
    print(f"\nSaved {len(kept)} quality conversations to {output_file}")
    
    # Show some examples of filtered ones
    if filtered:
        print(f"\n--- Examples of filtered conversations ---")
        for conv in filtered[:2]:
            print(f"\nRelationship: {conv['relationship_type']}")
            print(f"Score: {conv['quality_score']:.2f}")
            print(f"Issues: {conv['quality_issues']}")
            for msg in conv['messages'][:4]:
                sender = "A" if msg['sender_uuid'] == conv.get('main_uuid') else "B"
                print(f"  {sender}: {msg['text'][:60]}...")
    
    return kept


if __name__ == "__main__":
    run_quality_filter(min_score=0.6)
