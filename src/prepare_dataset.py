#!/usr/bin/env python3
"""Prepare ham_messages.json and spam_messages.json from raw data.

Run this script before starting federated learning training.
It processes:
- data/conversations.json -> data/ham_messages.json (messages received by personas)
- data/super_sms_dataset.csv -> data/spam_messages.json (spam messages for distribution)

Usage:
    python -m src.prepare_dataset
    # or
    python src/prepare_dataset.py
"""

import json
from pathlib import Path

import pandas as pd


def main():
    # Determine data directory (works from repo root or src/)
    script_dir = Path(__file__).parent
    if script_dir.name == "src":
        data_dir = script_dir.parent / "data"
    else:
        data_dir = Path("data")

    print(f"Data directory: {data_dir.resolve()}")

    # Check required files exist
    conversations_file = data_dir / "conversations.json"
    spam_csv_file = data_dir / "super_sms_dataset.csv"

    if not conversations_file.exists():
        raise FileNotFoundError(f"Missing: {conversations_file}")
    if not spam_csv_file.exists():
        raise FileNotFoundError(f"Missing: {spam_csv_file}")

    # -------------------------------------------------------------------------
    # 1. Process ham messages from conversations
    # -------------------------------------------------------------------------
    print("\n=== Processing Ham Messages ===")

    with open(conversations_file, "r", encoding="utf-8") as f:
        conversations = json.load(f)

    # Extract only messages RECEIVED by main persona (sender != main_uuid)
    ham_messages = []
    for conv in conversations:
        main_uuid = conv["main_uuid"]
        for msg in conv["messages"]:
            if msg["sender_uuid"] != main_uuid:  # Only received messages
                ham_messages.append(
                    {
                        "text": msg["text"],
                        "label": 0,
                        "main_uuid": main_uuid,  # Track which persona received it
                    }
                )

    print(f"Ham messages (received by personas): {len(ham_messages)}")
    unique_personas = len(set(m["main_uuid"] for m in ham_messages))
    print(f"Unique main personas: {unique_personas}")

    # Save ham messages
    ham_output = data_dir / "ham_messages.json"
    with open(ham_output, "w", encoding="utf-8") as f:
        json.dump(ham_messages, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {ham_output}")

    # -------------------------------------------------------------------------
    # 2. Process spam messages from CSV
    # -------------------------------------------------------------------------
    print("\n=== Processing Spam Messages ===")

    df = pd.read_csv(spam_csv_file, encoding="latin-1")
    spam_df = df[df["Labels"] == 1][["SMSes"]].copy()
    spam_df.columns = ["text"]
    spam_df["label"] = 1

    print(f"Spam messages: {len(spam_df)}")

    # Save spam messages
    spam_output = data_dir / "spam_messages.json"
    spam_messages = spam_df[["text", "label"]].to_dict(orient="records")
    with open(spam_output, "w", encoding="utf-8") as f:
        json.dump(spam_messages, f, ensure_ascii=False, indent=2)
    print(f"Saved to: {spam_output}")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n=== Dataset Summary ===")
    print(f"Ham messages: {len(ham_messages)}")
    print(f"Spam messages: {len(spam_df)}")
    print(f"Total: {len(ham_messages) + len(spam_df)}")
    print(f"\nUnique personas (FL clients): {unique_personas}")
    print(f"Avg ham per persona: {len(ham_messages) / unique_personas:.1f}")
    print(f"Avg spam per persona (if IID): {len(spam_df) / unique_personas:.1f}")

    print("\n✅ Dataset preparation complete!")
    print("You can now run: flwr run .")


if __name__ == "__main__":
    main()
