"""
Shared data schemas for FLspam.

Pydantic models for personas, relationships, conversations, and messages.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# =============================================================================
# Enums
# =============================================================================


class RelationshipType(str, Enum):
    """Types of relationships between personas."""

    PARTNER = "partner"
    FAMILY = "family"
    CLOSE_FRIENDS = "close_friends"
    FRIENDS = "friends"
    COLLEAGUES = "colleagues"
    NEIGHBORS = "neighbors"
    PROFESSIONALS = "professionals"
    BUSINESSES = "businesses"
    CASUAL = "casual"
    OTHER = "other"


class MessageLabel(int, Enum):
    """Message classification labels."""

    HAM = 0
    SPAM = 1


# =============================================================================
# Persona & Relationship Models
# =============================================================================


class Persona(BaseModel):
    """A persona with demographic and personality information."""

    uuid: str
    age: int
    gender: str
    occupation: str = ""
    education_level: str = ""
    marital_status: str = ""
    city: str = ""
    state: str = ""
    persona: str = ""  # Full persona description
    professional_persona: str = ""  # Professional description (for service providers)
    hobbies_and_interests_list: list[str] = Field(default_factory=list)

    class Config:
        extra = "allow"  # Allow extra fields from dataset


class Relationship(BaseModel):
    """A relationship between two personas."""

    from_uuid: str
    to_uuid: str
    relationship_type: RelationshipType
    service_type: str | None = None  # For professionals (e.g., "plumber")
    business_type: str | None = None  # For businesses (e.g., "restaurant")
    business_name: str | None = None  # For businesses (e.g., "Joe's Pizza")


# =============================================================================
# Conversation Models
# =============================================================================


class Message(BaseModel):
    """A single SMS message in a conversation."""

    sender_uuid: str
    text: str
    timestamp_offset_minutes: int = 0  # Minutes from conversation start


class Conversation(BaseModel):
    """A complete SMS conversation between two personas."""

    main_uuid: str
    contact_uuid: str
    relationship_type: RelationshipType
    scenario: str
    messages: list[Message] = Field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.messages) == 0

    def add_message(self, sender_uuid: str, text: str, offset_minutes: int = 0) -> None:
        self.messages.append(
            Message(
                sender_uuid=sender_uuid,
                text=text,
                timestamp_offset_minutes=offset_minutes,
            )
        )


# =============================================================================
# Dataset Models (for FL training)
# =============================================================================


class LabeledMessage(BaseModel):
    """A message with spam/ham label and optional persona assignment."""

    text: str
    label: MessageLabel
    main_uuid: str | None = (
        None  # Persona this message belongs to (for FL partitioning)
    )


class ClientDataset(BaseModel):
    """Dataset for a single FL client (persona)."""

    persona_uuid: str
    messages: list[LabeledMessage] = Field(default_factory=list)

    @property
    def ham_count(self) -> int:
        return sum(1 for m in self.messages if m.label == MessageLabel.HAM)

    @property
    def spam_count(self) -> int:
        return sum(1 for m in self.messages if m.label == MessageLabel.SPAM)


# =============================================================================
# Configuration Models
# =============================================================================


class ContactDistribution(BaseModel):
    """Configuration for contact distribution per relationship type."""

    partner: int = 1
    family: int = 15
    close_friends: int = 10
    friends: int = 20
    colleagues: int = 15
    neighbors: int = 8
    professionals: int = 20
    businesses: int = 10
    casual: int = 8
    other: int = 5

    def get(self, rel_type: RelationshipType) -> int:
        return getattr(self, rel_type.value, 0)

    def total(self) -> int:
        return sum(getattr(self, rt.value) for rt in RelationshipType)


class ConversationConfig(BaseModel):
    """Configuration for conversation generation."""

    conversations_per_relationship: dict[str, int] = Field(
        default_factory=lambda: {
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
    )

    turn_ranges: dict[str, tuple[int, int]] = Field(
        default_factory=lambda: {
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
    )

    def get_turn_count(self, rel_type: str) -> int:
        import random

        min_t, max_t = self.turn_ranges.get(rel_type, (3, 8))
        return random.randint(min_t, max_t)

    def get_num_conversations(self, rel_type: str) -> int:
        return self.conversations_per_relationship.get(rel_type, 1)


# =============================================================================
# Serialization Helpers
# =============================================================================


def personas_to_dict(personas: list[Persona]) -> dict[str, dict]:
    """Convert list of Personas to UUID-keyed dict for JSON serialization."""
    return {p.uuid: p.model_dump() for p in personas}


def relationships_to_list(relationships: list[Relationship]) -> list[dict]:
    """Convert list of Relationships to list of dicts for JSON serialization."""
    return [r.model_dump() for r in relationships]


def conversations_to_list(conversations: list[Conversation]) -> list[dict]:
    """Convert list of Conversations to list of dicts for JSON serialization."""
    return [c.model_dump() for c in conversations]


def load_personas_from_json(path: str) -> dict[str, Persona]:
    """Load personas from JSON file."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {uuid: Persona(**p) for uuid, p in data.items()}


def load_relationships_from_json(path: str) -> list[Relationship]:
    """Load relationships from JSON file."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Relationship(**r) for r in data]


def load_conversations_from_json(path: str) -> list[Conversation]:
    """Load conversations from JSON file."""
    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Conversation(**c) for c in data]
