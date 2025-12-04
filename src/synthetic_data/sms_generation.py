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


# Re-export schemas for backward compatibility

# Main entry point
if __name__ == "__main__":
    from src.synthetic_data.generator import main

    main()
