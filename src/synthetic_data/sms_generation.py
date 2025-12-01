"""
SMS conversation generator using personas and relationships.
Uses CAMEL-style multi-agent simulation with distilabel + vLLM (Qwen3).

This file is a backward-compatible wrapper around the refactored modules.
See:
- config.py: Configuration constants
- text_utils.py: Message cleaning and validation
- generator.py: Core generation logic
"""

# Re-export everything from modules for backward compatibility
from src.synthetic_data.config import (
    DATA_DIR,
    PERSONAS_FILE,
    RELATIONSHIPS_FILE,
    CONVERSATIONS_FILE as OUTPUT_FILE,
    MODEL_NAME,
    CONVERSATIONS_PER_RELATIONSHIP,
    SCENARIO_TEMPLATES,
    TURN_RANGES,
    VLLM_CONFIG,
    GENERATION_CONFIG,
)

from src.synthetic_data.text_utils import (
    clean_message,
    check_message_quality,
    should_end_conversation,
    infer_sms_style,
    extract_key_phrases,
)

from src.synthetic_data.generator import (
    generate_conversations,
    load_data,
    save_conversations,
    create_llm,
    build_prompt,
    get_turn_count,
    get_contact_identity,
    generate_scenario,
)

# Re-export schemas for backward compatibility
from src.schemas import Message, Conversation

# Main entry point
if __name__ == "__main__":
    from src.synthetic_data.generator import main
    main()
