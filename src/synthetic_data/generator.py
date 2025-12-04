"""
SMS conversation generator using distilabel + vLLM.

This module provides the core conversation generation logic,
refactored for clean separation of concerns.
"""

import json
import random
import re
from pathlib import Path

from tqdm import tqdm

from src.schemas import Message, Conversation
from src.synthetic_data.config import (
    PERSONAS_FILE,
    RELATIONSHIPS_FILE,
    CONVERSATIONS_FILE,
    MODEL_NAME,
    VLLM_CONFIG,
    GENERATION_CONFIG,
    CONVERSATIONS_PER_RELATIONSHIP,
    TURN_RANGES,
    SCENARIO_TEMPLATES,
    RELATIONSHIP_IDENTITIES,
)
from src.synthetic_data.text_utils import (
    clean_message,
    check_message_quality,
    should_end_conversation,
    infer_sms_style,
)


# =============================================================================
# Helper Functions
# =============================================================================


def get_turn_count(relationship_type: str) -> int:
    """Get random turn count based on relationship type."""
    min_t, max_t = TURN_RANGES.get(relationship_type, (3, 8))
    return random.randint(min_t, max_t)


def get_contact_identity(rel_type: str, service_type: str | None = None) -> str:
    """Get a clear, natural description of the relationship for the prompt."""
    if service_type:
        return f"{service_type} (professional service)"
    return RELATIONSHIP_IDENTITIES.get(rel_type, "contact")


def generate_scenario(relationship_type: str, service_type: str | None = None) -> str:
    """Pick a scenario template based on relationship."""
    if service_type:
        templates = SCENARIO_TEMPLATES.get("professionals", ["General inquiry"])
        return f"Contacting {service_type} - {random.choice(templates)}"
    templates = SCENARIO_TEMPLATES.get(relationship_type, ["General conversation"])
    return random.choice(templates)


# =============================================================================
# Prompt Building
# =============================================================================


def build_prompt(
    persona: dict,
    is_main: bool,
    scenario: str,
    contact_identity: str,
    service_type: str | None,
    history: list[dict],
    turn_number: int = 0,
    total_turns: int = 10,
) -> tuple[list[dict], str]:
    """
    Build chat messages for LLM.

    Returns (messages, cache_key) where cache_key groups prompts with same prefix.

    Prompt structure optimized for prefix caching:
    - Static persona info FIRST (cached across conversations)
    - Dynamic scenario/history LAST (varies per conversation)
    """
    style = infer_sms_style(persona)
    persona_text = (
        persona.get("professional_persona")
        if service_type and not is_main
        else persona.get("persona", "")
    )

    # Extract first name from persona if available
    name_match = re.match(r"^([A-Z][a-z]+)", persona_text)
    first_name = name_match.group(1) if name_match else "Person"

    # Cache key = persona UUID + is_main (same persona in same role = same prefix)
    cache_key = f"{persona.get('uuid', '')}_{is_main}"

    # Determine if near end of conversation
    near_end = turn_number >= total_turns - 2
    ending_instruction = (
        "\n6. WRAP UP: Send a brief closer and END the conversation."
        if near_end
        else ""
    )

    # Extract what was recently said to avoid repetition
    recent_topics = ""
    if history:
        last_msg = history[-1]["text"].lower()
        time_match = re.search(r"(?:in|at|around)\s+\d+", last_msg)
        if time_match:
            recent_topics = (
                f"\nDO NOT REPEAT: '{time_match.group()}' - this was already said."
            )

    system = f"""You are {first_name}.

BACKGROUND: {persona_text[:200]}...

TEXTING STYLE: {style}

---
You're texting your {contact_identity}.
SITUATION: {scenario}

CRITICAL RULES:
1. MAX 20 words. Real texts are SHORT.
2. NO greetings or sign-offs
3. NEVER repeat what was just said - move the conversation forward
4. Use contractions (I'm, don't, gonna)
5. Don't sign your name{ending_instruction}{recent_topics}

Write ONLY the text message. Nothing else."""

    # Only show last 4 messages to reduce repetition
    history_text = "\n".join(
        f"{'You' if m['is_main'] == is_main else 'Them'}: {m['text']}"
        for m in history[-4:]
    )

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": f"{history_text or '(Start the conversation)'}\n\nYour text:",
        },
    ], cache_key


# =============================================================================
# Conversation Generation
# =============================================================================


def generate_conversations(
    personas: dict,
    relationships: list,
    llm,  # distilabel LLM instance
    limit: int | None = None,
) -> list[Conversation]:
    """
    Generate all conversations with batched LLM calls.

    Optimized for vLLM prefix caching:
    - Conversations sorted by pair_key so same persona pairs are processed together
    - Batches sorted by cache_key to maximize prefix reuse
    """
    # Prepare conversation states
    convs = []
    for rel in relationships[:limit]:
        main = personas.get(rel["from_uuid"])
        contact = personas.get(rel["to_uuid"])
        if not main or not contact:
            continue

        rel_type = rel["relationship_type"]
        service_type = rel.get("service_type")
        contact_identity = get_contact_identity(rel_type, service_type)
        pair_key = f"{main['uuid']}_{contact['uuid']}"

        # Generate multiple conversations per relationship
        num_convs = CONVERSATIONS_PER_RELATIONSHIP.get(rel_type, 1)
        available_scenarios = SCENARIO_TEMPLATES.get(rel_type, ["General conversation"])

        for conv_idx in range(num_convs):
            scenario = available_scenarios[conv_idx % len(available_scenarios)]
            if service_type:
                scenario = f"Contacting {service_type} - {scenario}"

            convs.append(
                {
                    "main": main,
                    "contact": contact,
                    "rel_type": rel_type,
                    "scenario": scenario,
                    "service_type": service_type,
                    "contact_identity": contact_identity,
                    "turns": get_turn_count(rel_type),
                    "current_turn": 0,
                    "history": [],
                    "time_offset": 0,
                    "pair_key": pair_key,
                }
            )

    # Sort by pair_key to maximize prefix cache hits
    convs.sort(key=lambda c: c["pair_key"])

    total_messages = sum(c["turns"] for c in convs)
    max_turns = max(c["turns"] for c in convs) if convs else 0

    print(f"Total conversations: {len(convs)}")
    print(f"Total messages to generate: {total_messages}")
    print(f"Max turns per conversation: {max_turns}")
    print(
        f"Unique persona pairs: {len(set(c['pair_key'] for c in convs))} (sorted for cache efficiency)"
    )

    pbar = tqdm(total=total_messages, desc="Generating SMS", unit="msg")

    for turn in range(max_turns):
        active_convs = [
            (i, c) for i, c in enumerate(convs) if c["current_turn"] < c["turns"]
        ]
        if not active_convs:
            break

        # Build batch with cache keys for sorting
        batch_data = []
        for i, conv in active_convs:
            if should_end_conversation(conv["history"]):
                conv["current_turn"] = conv["turns"]
                continue

            is_main_turn = conv["current_turn"] % 2 == 0
            persona = conv["main"] if is_main_turn else conv["contact"]

            messages, cache_key = build_prompt(
                persona,
                is_main_turn,
                conv["scenario"],
                conv["contact_identity"],
                conv["service_type"],
                conv["history"],
                turn_number=conv["current_turn"],
                total_turns=conv["turns"],
            )
            batch_data.append((messages, cache_key, i, is_main_turn))

        if not batch_data:
            continue

        # Sort by cache_key for prefix reuse
        batch_data.sort(key=lambda x: x[1])
        batch_inputs = [d[0] for d in batch_data]
        batch_meta = [(d[2], d[3]) for d in batch_data]

        # Generate
        results = llm.generate(batch_inputs)

        # Process results
        retry_needed = []
        for (i, is_main_turn), result in zip(batch_meta, results):
            conv = convs[i]
            text = result["generations"][0].strip().strip('"').strip("'")
            text = clean_message(text)

            is_valid, issue = check_message_quality(
                text, conv["history"], conv["rel_type"]
            )
            if not is_valid:
                retry_needed.append((i, is_main_turn, issue))
                continue

            _add_message_to_conv(conv, is_main_turn, text)
            pbar.update(1)

        # Retry failed generations
        if retry_needed:
            _retry_failed_generations(convs, retry_needed, llm, pbar)

    pbar.close()

    # Build final conversation objects
    return [
        Conversation(
            main_uuid=c["main"]["uuid"],
            contact_uuid=c["contact"]["uuid"],
            relationship_type=c["rel_type"],
            scenario=c["scenario"],
            messages=[
                Message(
                    sender_uuid=m["sender_uuid"],
                    text=m["text"],
                    timestamp_offset_minutes=m["timestamp_offset_minutes"],
                )
                for m in c["history"]
            ],
        )
        for c in convs
    ]


def _add_message_to_conv(conv: dict, is_main_turn: bool, text: str) -> None:
    """Add a message to a conversation state."""
    gap = (
        random.randint(1, 15)
        if conv["rel_type"] in ("partner", "close_friends")
        else random.randint(2, 60)
    )
    conv["time_offset"] += gap

    conv["history"].append(
        {
            "is_main": is_main_turn,
            "sender_uuid": conv["main"]["uuid"]
            if is_main_turn
            else conv["contact"]["uuid"],
            "text": text,
            "timestamp_offset_minutes": conv["time_offset"],
        }
    )
    conv["current_turn"] += 1


def _retry_failed_generations(convs: list, retry_needed: list, llm, pbar) -> None:
    """Retry failed message generations with hints."""
    retry_inputs = []
    retry_meta = []

    for i, is_main_turn, issue in retry_needed:
        conv = convs[i]
        persona = conv["main"] if is_main_turn else conv["contact"]

        messages, _ = build_prompt(
            persona,
            is_main_turn,
            conv["scenario"],
            conv["contact_identity"],
            conv["service_type"],
            conv["history"],
            turn_number=conv["current_turn"],
            total_turns=conv["turns"],
        )
        messages[-1]["content"] += (
            f"\n\n(Be different. Previous was {issue}. Keep it SHORT and FRESH.)"
        )
        retry_inputs.append(messages)
        retry_meta.append((i, is_main_turn))

    if retry_inputs:
        retry_results = llm.generate(retry_inputs)
        for (i, is_main_turn), result in zip(retry_meta, retry_results):
            conv = convs[i]
            text = result["generations"][0].strip().strip('"').strip("'")
            text = clean_message(text)
            _add_message_to_conv(conv, is_main_turn, text)
            pbar.update(1)


# =============================================================================
# Data I/O
# =============================================================================


def load_data(
    personas_file: Path = PERSONAS_FILE,
    relationships_file: Path = RELATIONSHIPS_FILE,
) -> tuple[dict, list]:
    """Load personas and relationships from JSON files."""
    with open(personas_file, encoding="utf-8") as f:
        personas = json.load(f)
    with open(relationships_file, encoding="utf-8") as f:
        relationships = json.load(f)
    return personas, relationships


def save_conversations(
    conversations: list[Conversation],
    output_file: Path = CONVERSATIONS_FILE,
) -> None:
    """Save conversations to JSON."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            [c.model_dump() for c in conversations], f, indent=2, ensure_ascii=False
        )
    print(f"Saved {len(conversations)} conversations to {output_file}")


# =============================================================================
# LLM Initialization
# =============================================================================


def create_llm():
    """Create and initialize the vLLM instance."""
    from distilabel.models import vLLM

    llm = vLLM(
        model=MODEL_NAME,
        dtype=VLLM_CONFIG["dtype"],
        extra_kwargs={k: v for k, v in VLLM_CONFIG.items() if k != "dtype"},
        generation_kwargs=GENERATION_CONFIG,
    )
    llm.load()
    return llm


# =============================================================================
# Main Entry Point
# =============================================================================


def main(limit: int | None = None):
    """Generate SMS conversations."""
    print("Loading data...")
    personas, relationships = load_data()
    print(f"Loaded {len(personas)} personas and {len(relationships)} relationships")

    print("\nInitializing LLM...")
    llm = create_llm()

    print("\nGenerating conversations...")
    conversations = generate_conversations(personas, relationships, llm, limit=limit)

    save_conversations(conversations)

    # Print samples
    print("\n" + "=" * 60)
    print("SAMPLE CONVERSATIONS")
    print("=" * 60)
    for conv in conversations[:3]:
        print(f"\n{'─' * 50}")
        print(f"📱 {conv.relationship_type.upper()} | {conv.scenario}")
        print("─" * 50)
        for msg in conv.messages:
            sender = "→" if msg.sender_uuid == conv.main_uuid else "←"
            print(f"  {sender} [{msg.timestamp_offset_minutes:3d}m] {msg.text}")

    return conversations


if __name__ == "__main__":
    main()
