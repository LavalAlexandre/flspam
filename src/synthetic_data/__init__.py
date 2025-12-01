"""
Synthetic data generation for FLspam.

Modules:
- config: Configuration constants and templates
- text_utils: SMS text cleaning and validation
- generator: Conversation generation using vLLM
- personas: Persona network generation
"""

from src.synthetic_data.config import (
    DATA_DIR,
    PERSONAS_FILE,
    RELATIONSHIPS_FILE,
    CONVERSATIONS_FILE,
    MODEL_NAME,
    CONVERSATIONS_PER_RELATIONSHIP,
    SCENARIO_TEMPLATES,
    TURN_RANGES,
    RELATIONSHIP_IDENTITIES,
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
    main as generate_sms,
)

__all__ = [
    # Config
    "DATA_DIR",
    "PERSONAS_FILE",
    "RELATIONSHIPS_FILE",
    "CONVERSATIONS_FILE",
    "MODEL_NAME",
    "CONVERSATIONS_PER_RELATIONSHIP",
    "SCENARIO_TEMPLATES",
    "TURN_RANGES",
    "RELATIONSHIP_IDENTITIES",
    "VLLM_CONFIG",
    "GENERATION_CONFIG",
    # Text utils
    "clean_message",
    "check_message_quality",
    "should_end_conversation",
    "infer_sms_style",
    "extract_key_phrases",
    # Generator
    "generate_conversations",
    "load_data",
    "save_conversations",
    "create_llm",
    "build_prompt",
    "get_turn_count",
    "get_contact_identity",
    "generate_scenario",
    "generate_sms",
]
