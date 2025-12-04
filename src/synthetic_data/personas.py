"""
Persona network generator for synthetic SMS conversations.

This module creates contact networks for main personas, with relationships
distributed across family, friends, colleagues, professionals, and businesses.
"""

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from datasets import load_dataset


# =============================================================================
# Configuration
# =============================================================================

# Output directory for saved data
OUTPUT_DIR = Path("data")
PERSONAS_FILE = OUTPUT_DIR / "personas.json"
RELATIONSHIPS_FILE = OUTPUT_DIR / "relationships.json"

# Relationship categories and their distribution per main persona
# Note: "partner" is handled separately based on marital_status
CONTACT_DISTRIBUTION = {
    "partner": 1,  # Spouse/partner (only if married, max 1)
    "family": 15,  # Parents, siblings, children, in-laws, etc.
    "close_friends": 10,  # Best friends, childhood friends
    "friends": 20,  # Regular friends, acquaintances
    "colleagues": 15,  # Coworkers, boss, professional contacts (same occupation)
    "neighbors": 8,  # People living nearby
    "professionals": 20,  # Service providers (plumber, doctor, mechanic, etc.)
    "businesses": 10,  # Restaurants, stores, delivery services
    "casual": 8,  # Gym buddies, hobby groups, etc.
    "other": 5,  # Misc contacts
}
# CONTACT_DISTRIBUTION = {
#     "partner": 1,          # Spouse/partner (only if married, max 1)
#     "family": 4,           # Parents, siblings, children, in-laws, etc.
#     "close_friends": 2,   # Best friends, childhood friends
#     "friends": 4,         # Regular friends, acquaintances
#     "colleagues": 3,      # Coworkers, boss, professional contacts (same occupation)
#     "neighbors": 2,        # People living nearby
#     "professionals": 5,   # Service providers (plumber, doctor, mechanic, etc.)
#     "businesses": 5,      # Restaurants, stores, delivery services
#     "casual": 2,          # Gym buddies, hobby groups, etc.
#     "other": 2,            # Misc contacts
# }

# Marital statuses that indicate a partner
MARRIED_STATUSES = {"married_present", "married"}

# Minimum age for personas
MIN_AGE = 18

# Professional service types mapped to occupation field values from the dataset
# Keys are service types (for display), values are occupation field patterns to match
PROFESSIONAL_OCCUPATION_MAP = {
    "plumber": ["plumber", "pipefitter"],
    "electrician": ["electrician"],
    "mechanic": ["mechanic", "automotive", "auto_body"],
    "doctor": ["physician", "surgeon", "doctor"],
    "dentist": ["dentist"],
    "hairdresser": ["hairdresser", "hairstylist", "barber", "cosmetologist"],
    "veterinarian": ["veterinarian"],
    "accountant": ["accountant", "auditor"],
    "lawyer": ["lawyer", "attorney"],
    "realtor": ["real_estate", "realtor"],
    "contractor": ["construction", "contractor", "carpenter", "roofer"],
    "landscaper": ["landscap", "groundskeep", "gardener"],
    "tutor": ["tutor", "teacher", "instructor", "professor"],
    "personal_trainer": ["fitness", "trainer", "coach", "athletic"],
    "therapist": ["therapist", "counselor", "psychologist", "social_worker"],
    "childcare": ["childcare", "nanny", "preschool", "daycare"],
    "pet_care": ["veterinar", "animal", "pet"],
    "cleaner": ["janitor", "housekeeper", "cleaning", "maid"],
    "handyman": ["maintenance", "repair", "handyman"],
    "insurance_agent": ["insurance"],
    "nurse": ["nurse", "nursing", "rn", "lpn"],
    "pharmacist": ["pharmac"],
    "optometrist": ["optometrist", "optician"],
    "chiropractor": ["chiropractor"],
    "massage_therapist": ["massage"],
    "financial_advisor": ["financial_advisor", "financial_planner", "investment"],
    "tax_preparer": ["tax_preparer", "tax_examiner"],
    "auto_mechanic": ["automotive", "auto_mechanic", "auto_body"],
    "hvac_technician": ["hvac", "heating", "air_condition"],
    "locksmith": ["locksmith"],
    "painter": ["painter"],
    "moving_service": ["mover", "moving"],
}

# List of professional types (for backwards compatibility)
PROFESSIONAL_TYPES = list(PROFESSIONAL_OCCUPATION_MAP.keys())

# Business types
BUSINESS_TYPES = [
    "pharmacy",
    "restaurant",
    "pizza_delivery",
    "grocery_store",
    "bank",
    "gym",
    "salon",
    "auto_shop",
    "dry_cleaner",
    "pet_store",
    "dentist_office",
    "medical_clinic",
    "school",
    "daycare",
    "veterinary_clinic",
]

# Columns to remove from the dataset (not needed for SMS generation)
COLUMNS_TO_REMOVE = [
    "sports_persona",
    "arts_persona",
    "travel_persona",
    "culinary_persona",
    "skills_and_expertise",
    "hobbies_and_interests",  # Keep hobbies_and_interests_list for friend matching
    "career_goals_and_ambitions",
    # Keep professional_persona for service provider matching
]


# =============================================================================
# Data classes
# =============================================================================


@dataclass
class Contact:
    """Represents a contact in a persona's network."""

    persona: dict[str, Any]
    relationship: str
    service_type: str | None = None
    business_type: str | None = None
    business_name: str | None = None


@dataclass
class PersonaNetwork:
    """Represents a main persona with their contact network."""

    main_persona: dict[str, Any]
    contacts: dict[str, list[Contact]] = field(default_factory=dict)

    @property
    def total_contacts(self) -> int:
        return sum(len(c) for c in self.contacts.values())


# =============================================================================
# Core functions
# =============================================================================


def load_personas(n: int = 1500, min_age: int = MIN_AGE) -> list[dict[str, Any]]:
    """
    Load personas from the Nemotron dataset using streaming.
    Only includes adults (age >= min_age).

    Args:
        n: Number of personas to load.
        min_age: Minimum age filter (default 18).

    Returns:
        List of persona dictionaries (adults only).
    """
    print(
        f"Loading {n} adult personas (age >= {min_age}) from nvidia/Nemotron-Personas-USA..."
    )
    ds = load_dataset("nvidia/Nemotron-Personas-USA", split="train", streaming=True)
    ds = ds.remove_columns(COLUMNS_TO_REMOVE)

    # Filter for adults only
    personas = []
    for persona in ds:
        age = persona.get("age", 0)
        if isinstance(age, str):
            try:
                age = int(age)
            except ValueError:
                continue
        if age >= min_age:
            personas.append(persona)
        if len(personas) >= n:
            break

    print(f"Loaded {len(personas)} adult personas")
    return personas


def select_main_personas(
    personas: list[dict[str, Any]],
    n: int = 10,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """
    Select diverse main personas from the pool.

    Args:
        personas: Pool of all available personas.
        n: Number of main personas to select.
        seed: Random seed for reproducibility.

    Returns:
        List of selected main personas.
    """
    if seed is not None:
        random.seed(seed)

    # Shuffle and pick first n
    shuffled = random.sample(personas, min(len(personas), n * 10))
    return shuffled[:n]


def _occupation_matches_service(occupation: str, service_type: str) -> bool:
    """
    Check if a persona's occupation matches a service type.

    Args:
        occupation: The persona's occupation field.
        service_type: The professional service type to match.

    Returns:
        True if occupation matches the service type.
    """
    if not occupation or service_type not in PROFESSIONAL_OCCUPATION_MAP:
        return False

    occupation_lower = occupation.lower()
    keywords = PROFESSIONAL_OCCUPATION_MAP[service_type]
    return any(keyword in occupation_lower for keyword in keywords)


def _has_shared_hobbies(
    persona1: dict[str, Any], persona2: dict[str, Any], min_shared: int = 1
) -> bool:
    """
    Check if two personas share at least min_shared hobbies/interests.

    Args:
        persona1: First persona.
        persona2: Second persona.
        min_shared: Minimum number of shared hobbies required.

    Returns:
        True if they share at least min_shared hobbies.
    """
    hobbies1 = persona1.get("hobbies_and_interests_list", [])
    hobbies2 = persona2.get("hobbies_and_interests_list", [])

    if not hobbies1 or not hobbies2:
        return False

    # Convert to sets for efficient intersection
    set1 = set(h.lower().strip() for h in hobbies1 if isinstance(h, str))
    set2 = set(h.lower().strip() for h in hobbies2 if isinstance(h, str))

    shared = set1 & set2
    return len(shared) >= min_shared


def _select_random_from_pool(
    pool: list[dict[str, Any]],
    used: set[int],
    filter_fn: callable = None,
) -> dict[str, Any] | None:
    """
    Select a random persona from pool that hasn't been used.

    Args:
        pool: List of personas to select from.
        used: Set of already used persona indices (by id()).
        filter_fn: Optional filter function for additional criteria.

    Returns:
        Selected persona or None if none available.
    """
    candidates = [
        (i, p)
        for i, p in enumerate(pool)
        if id(p) not in used and (filter_fn is None or filter_fn(p))
    ]

    if not candidates:
        return None

    idx, persona = random.choice(candidates)
    used.add(id(persona))
    return persona


def create_contact_network(
    main_persona: dict[str, Any],
    persona_pool: list[dict[str, Any]],
    used_personas: set[int],
    distribution: dict[str, int] | None = None,
) -> PersonaNetwork:
    """
    Create a contact network for a main persona.

    Args:
        main_persona: The main persona (phone owner).
        persona_pool: Pool of personas to draw contacts from.
        used_personas: Set of already used persona IDs (shared across all networks).
        distribution: Contact distribution by relationship type.

    Returns:
        PersonaNetwork with contacts organized by relationship type.
    """
    if distribution is None:
        distribution = CONTACT_DISTRIBUTION.copy()

    network = PersonaNetwork(main_persona=main_persona)
    main_occupation = main_persona.get("occupation", "")
    main_marital_status = main_persona.get("marital_status", "")

    # Check if main persona is married
    is_married = main_marital_status.lower() in MARRIED_STATUSES

    # Define filter functions outside the loop to avoid E731
    def _filter_colleagues(p: dict) -> bool:
        return p.get("occupation", "") == main_occupation and main_occupation != ""

    def _filter_partner(p: dict) -> bool:
        return p.get("marital_status", "").lower() in MARRIED_STATUSES

    def _filter_friends(p: dict, min_hobbies: int) -> bool:
        return _has_shared_hobbies(main_persona, p, min_shared=min_hobbies)

    def _filter_professional(p: dict, st: str) -> bool:
        return _occupation_matches_service(p.get("occupation", ""), st)

    for relationship_type, count in distribution.items():
        network.contacts[relationship_type] = []

        # Handle partner specially - only if married, max 1
        if relationship_type == "partner":
            if not is_married:
                continue  # Skip partner if not married
            count = 1  # Can only have 1 partner

        for _ in range(count):
            # Define filter based on relationship type
            filter_fn = None
            matched_service_type = None

            if relationship_type == "colleagues":
                # Colleagues must have the same occupation
                filter_fn = _filter_colleagues
            elif relationship_type == "partner":
                # Partner should be adult (already filtered) and not same person
                filter_fn = _filter_partner
            elif relationship_type in ("close_friends", "friends", "casual"):
                # Friends should share at least one hobby/interest
                min_hobbies = 2 if relationship_type == "close_friends" else 1

                def _friends_filter(p: dict, mh: int = min_hobbies) -> bool:
                    return _filter_friends(p, mh)

                filter_fn = _friends_filter
            elif relationship_type == "professionals":
                # Try to find a persona whose occupation matches a professional service type
                # Shuffle service types to get variety
                shuffled_services = random.sample(
                    PROFESSIONAL_TYPES, len(PROFESSIONAL_TYPES)
                )
                for service_type in shuffled_services:

                    def _prof_filter(p: dict, st: str = service_type) -> bool:
                        return _filter_professional(p, st)

                    filter_fn = _prof_filter
                    candidate = _select_random_from_pool(
                        persona_pool, used_personas, filter_fn
                    )
                    if candidate is not None:
                        matched_service_type = service_type
                        contact = Contact(
                            persona=candidate,
                            relationship=relationship_type,
                            service_type=service_type,
                        )
                        break

                # If we found a matching professional, skip the normal selection
                if matched_service_type is not None:
                    location = candidate.get("location", "Local").split(",")[0]
                    contact.business_name = (
                        f"{location} {contact.service_type.title()} Services"
                    )
                    network.contacts[relationship_type].append(contact)
                    continue

                # Fallback: no matching occupation found, use random persona with random service
                filter_fn = None

            contact_persona = _select_random_from_pool(
                persona_pool, used_personas, filter_fn
            )

            if contact_persona is None:
                # If no matching colleague found, try without filter for colleagues
                if relationship_type == "colleagues":
                    contact_persona = _select_random_from_pool(
                        persona_pool, used_personas
                    )
                # If no matching friend found by hobbies, try without filter
                elif relationship_type in ("close_friends", "friends", "casual"):
                    contact_persona = _select_random_from_pool(
                        persona_pool, used_personas
                    )
                if contact_persona is None:
                    continue

            # Create contact with appropriate metadata
            contact = Contact(
                persona=contact_persona,
                relationship=relationship_type,
            )

            # Add professional/business specific metadata
            location = contact_persona.get("location", "Local").split(",")[0]

            if relationship_type == "professionals":
                # Service type was already determined during selection
                if contact.service_type is None:
                    contact.service_type = random.choice(PROFESSIONAL_TYPES)
                contact.business_name = (
                    f"{location} {contact.service_type.title()} Services"
                )
            elif relationship_type == "businesses":
                contact.business_type = random.choice(BUSINESS_TYPES)
                contact.business_name = (
                    f"{location} {contact.business_type.replace('_', ' ').title()}"
                )

            network.contacts[relationship_type].append(contact)

    return network


def build_all_networks(
    main_personas: list[dict[str, Any]],
    persona_pool: list[dict[str, Any]],
) -> list[PersonaNetwork]:
    """
    Build contact networks for all main personas.
    Ensures no persona is used as a contact for multiple main personas.

    Args:
        main_personas: List of main personas.
        persona_pool: Pool of personas to draw contacts from.

    Returns:
        List of PersonaNetwork objects.
    """
    networks = []

    # Track all used personas globally to prevent duplicates
    used_personas: set[int] = set()

    # Mark main personas as used
    for main_p in main_personas:
        used_personas.add(id(main_p))

    for i, main_p in enumerate(main_personas):
        marital = main_p.get("marital_status", "unknown")
        occupation = main_p.get("occupation", "unknown")
        network = create_contact_network(main_p, persona_pool, used_personas)
        networks.append(network)

        partner_count = len(network.contacts.get("partner", []))
        colleague_count = len(network.contacts.get("colleagues", []))
        print(
            f"Persona {i + 1}: {network.total_contacts} contacts "
            f"(partner: {partner_count}, colleagues: {colleague_count}, "
            f"marital: {marital}, occupation: {occupation})"
        )

    return networks


def print_network_summary(network: PersonaNetwork) -> None:
    """Print a summary of a persona's contact network."""
    print("=" * 60)
    print("MAIN PERSONA:")
    print(f"  UUID: {network.main_persona.get('uuid')}")
    print(f"  Age: {network.main_persona.get('age')}")
    print(
        f"  Location: {network.main_persona.get('city')}, {network.main_persona.get('state')}"
    )
    print(f"  Occupation: {network.main_persona.get('occupation')}")
    print(f"  Marital Status: {network.main_persona.get('marital_status')}")
    persona_text = network.main_persona.get("persona", "")[:200]
    print(f"  Persona: {persona_text}...")
    print("=" * 60)

    print("\nCONTACT NETWORK SUMMARY:")
    for rel_type, contacts in network.contacts.items():
        if not contacts:
            continue
        print(f"\n{rel_type.upper()} ({len(contacts)} contacts):")
        for c in contacts[:3]:  # Show first 3 of each type
            if c.service_type:
                occupation = c.persona.get("occupation", "N/A")
                print(
                    f"  - {c.business_name} ({c.service_type}) - occupation: {occupation}"
                )
            elif c.business_type:
                print(f"  - {c.business_name} ({c.business_type})")
            elif rel_type == "partner":
                print(
                    f"  - Age: {c.persona.get('age')}, Occupation: {c.persona.get('occupation')}"
                )
            elif rel_type == "colleagues":
                print(
                    f"  - Age: {c.persona.get('age')}, Occupation: {c.persona.get('occupation')}"
                )
            else:
                print(
                    f"  - Age: {c.persona.get('age')}, Location: {c.persona.get('city')}, {c.persona.get('state')}"
                )


def save_personas_and_relationships(
    networks: list[PersonaNetwork],
    personas_file: Path = PERSONAS_FILE,
    relationships_file: Path = RELATIONSHIPS_FILE,
) -> None:
    """
    Save all personas and their relationships to JSON files.

    Args:
        networks: List of PersonaNetwork objects.
        personas_file: Path to save personas JSON.
        relationships_file: Path to save relationships JSON.
    """
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all unique personas (main + contacts)
    all_personas: dict[str, dict[str, Any]] = {}

    # Collect all relationships
    relationships: list[dict[str, Any]] = []

    for network in networks:
        main_uuid = network.main_persona.get("uuid")

        # Add main persona
        if main_uuid and main_uuid not in all_personas:
            all_personas[main_uuid] = network.main_persona

        # Process contacts and relationships
        for rel_type, contacts in network.contacts.items():
            for contact in contacts:
                contact_uuid = contact.persona.get("uuid")

                # Add contact persona
                if contact_uuid and contact_uuid not in all_personas:
                    all_personas[contact_uuid] = contact.persona

                # Create relationship record
                relationship = {
                    "from_uuid": main_uuid,
                    "to_uuid": contact_uuid,
                    "relationship_type": rel_type,
                }

                # Add optional fields for professionals/businesses
                if contact.service_type:
                    relationship["service_type"] = contact.service_type
                    relationship["business_name"] = contact.business_name
                if contact.business_type:
                    relationship["business_type"] = contact.business_type
                    relationship["business_name"] = contact.business_name

                relationships.append(relationship)

    # Save personas
    with open(personas_file, "w", encoding="utf-8") as f:
        json.dump(all_personas, f, indent=2, ensure_ascii=False)

    # Save relationships
    with open(relationships_file, "w", encoding="utf-8") as f:
        json.dump(relationships, f, indent=2, ensure_ascii=False)

    print(f"\n✓ Saved {len(all_personas)} personas to {personas_file}")
    print(f"✓ Saved {len(relationships)} relationships to {relationships_file}")


# =============================================================================
# Main
# =============================================================================


def main():
    """Main entry point for generating persona networks."""
    # Load personas
    all_personas = load_personas(n=30000)

    # Select 10 main personas
    main_personas = select_main_personas(all_personas, n=20, seed=42)

    print("\nSelected main personas:")
    for i, p in enumerate(main_personas):
        print(
            f"  {i + 1}. Age: {p.get('age', 'N/A')}, "
            f"Occupation: {p.get('occupation', 'N/A')}, "
            f"Marital: {p.get('marital_status', 'N/A')}"
        )

    # Build contact networks
    print("\nBuilding contact networks...")
    networks = build_all_networks(main_personas, all_personas)

    # Show summary of first network
    print("\n")
    print_network_summary(networks[0])

    # Save to files
    save_personas_and_relationships(networks)

    print(f"\n✓ Created {len(networks)} persona networks")
    print(f"  Total contacts per persona: {sum(CONTACT_DISTRIBUTION.values())}")

    return networks


if __name__ == "__main__":
    networks = main()
