"""
SMS generation configuration and constants.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PERSONAS_FILE = DATA_DIR / "personas.json"
RELATIONSHIPS_FILE = DATA_DIR / "relationships.json"
CONVERSATIONS_FILE = DATA_DIR / "conversations.json"

# =============================================================================
# Model Configuration
# =============================================================================

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"  # Optimized for T4 (16GB VRAM)

# vLLM settings for T4 GPU
VLLM_CONFIG = {
    "dtype": "float16",  # T4 doesn't support bfloat16
    "tensor_parallel_size": 1,
    "gpu_memory_utilization": 0.92,
    "max_model_len": 2048,
    "swap_space": 0,
    "enforce_eager": False,
    "max_num_seqs": 64,
    "enable_chunked_prefill": True,
    "enable_prefix_caching": True,
}

GENERATION_CONFIG = {
    "max_tokens": 60,
    "temperature": 0.9,
    "top_p": 0.95,
    "stop": [
        "\n\n",
        "Them:",
        "You:",
        "THEM:",
        "ME:",
        "Best regards",
        "Regards,",
        "Sincerely",
    ],
}

# =============================================================================
# Conversation Generation Config
# =============================================================================

# Number of conversations to generate per relationship type
CONVERSATIONS_PER_RELATIONSHIP = {
    "partner": 8,
    "close_friends": 5,
    "friends": 3,
    "family": 4,
    "colleagues": 3,
    "professionals": 2,
    "businesses": 2,
    "neighbors": 2,
    "casual": 2,
    "other": 1,
}

# Turn count ranges per relationship type
TURN_RANGES = {
    "partner": (8, 20),
    "close_friends": (6, 15),
    "friends": (4, 12),
    "family": (4, 10),
    "colleagues": (3, 8),
    "professionals": (2, 6),
    "businesses": (2, 4),
    "neighbors": (2, 6),
    "casual": (3, 8),
    "other": (2, 6),
}

# =============================================================================
# Scenario Templates
# =============================================================================

SCENARIO_TEMPLATES = {
    "partner": [
        "Asking for help with something",
        "Planning dinner tonight",
        "Checking in during work day",
        "Discussing weekend plans",
        "Small argument about chores",
    ],
    "family": [
        "Asking for help with something",
        "Catching up after not talking for a few days",
        "Planning a family gathering",
        "Sharing news about a relative",
    ],
    "close_friends": [
        "Making plans to hang out",
        "Sharing gossip or news",
        "Venting about work or life",
        "Asking for help with something",
    ],
    "friends": [
        "Casual catch-up",
        "Sharing a meme or link",
        "Making loose plans",
        "Asking for advice",
    ],
    "colleagues": [
        "Quick work question",
        "Coordinating on a project",
        "Office gossip",
        "Asking for help with something",
    ],
    "professionals": [
        "Scheduling an appointment",
        "Following up on a service",
        "Asking for a quote",
        "Confirming arrival time",
    ],
    "businesses": [
        "Confirming a reservation",
        "Checking order status",
        "Asking about hours or availability",
    ],
    "neighbors": [
        "Asking about a package delivery",
        "Noise complaint (polite)",
        "Borrowing something",
    ],
    "casual": [
        "Planning a group activity",
        "Sharing hobby-related info",
    ],
}

# =============================================================================
# Relationship Identity Descriptions
# =============================================================================

RELATIONSHIP_IDENTITIES = {
    "partner": "spouse/romantic partner (you live together, you're a couple)",
    "close_friends": "close friend (you hang out often, very casual)",
    "friends": "friend",
    "family": "family member",
    "colleagues": "coworker",
    "professionals": "professional contact",
    "businesses": "business/service",
    "neighbors": "neighbor",
    "casual": "acquaintance",
    "other": "contact",
}
