"""Spam task prompt generation."""

import random

SPAM_OBJECTIVES = [
    # Link-based scams
    "get the recipient to click a link to verify their account",
    "get the recipient to click a link to claim a refund",
    "get the recipient to click a link to track a package",
    "get the recipient to click a link to update payment info",
    "get the recipient to click a link to view a document",
    "get the recipient to visit a website for a special offer",
    # Phone-based scams
    "get the recipient to call a number about suspicious activity",
    "get the recipient to call back urgently",
    "get the recipient to text a short code to claim a prize",
    "get the recipient to reply with personal information",
    # Money/payment scams
    "get the recipient to send money via gift cards",
    "get the recipient to wire money urgently",
    "get the recipient to pay a fake invoice",
    "get the recipient to invest in a fake opportunity",
    # Credential theft
    "get the recipient to provide login credentials",
    "get the recipient to verify their SSN or ID",
    "get the recipient to confirm bank account details",
    # Social engineering
    "convince the recipient they won a lottery or prize",
    "pretend there's a problem with a recent order",
    "claim the recipient's account will be suspended",
    "impersonate a family member needing help",
    "offer a fake job with high pay",
    "threaten legal action unless they respond",
    "claim to be from IT needing remote access",
]

SPAM_CONTEXTS = [
    # Financial
    "bank (Chase, Wells Fargo, Bank of America)",
    "credit card company (Visa, Mastercard, Amex)",
    "payment service (PayPal, Venmo, Zelle)",
    "cryptocurrency exchange",
    # Government
    "IRS or tax authority",
    "Social Security Administration",
    "immigration services",
    "local court or legal office",
    # Delivery/Retail
    "shipping company (FedEx, UPS, USPS, DHL)",
    "Amazon or online retailer",
    "food delivery service",
    # Tech
    "Apple or iCloud support",
    "Microsoft or Windows support",
    "Google or Gmail security",
    "social media platform (Facebook, Instagram)",
    "antivirus or security software",
    # Telecom
    "mobile carrier (Verizon, AT&T, T-Mobile)",
    "internet service provider",
    # Other
    "healthcare provider or insurance",
    "employer HR department",
    "utility company",
    "lottery or sweepstakes",
    "dating or social site",
    "unknown sender with urgent news",
]

# Payload types to request
PAYLOAD_TYPES = [
    (
        "url",
        "Include a URL (can be shortened like bit.ly or a domain like example.com).",
    ),
    ("phone", "Include a phone number to call or text."),
    ("short_code", "Include a short code to text (like 'text YES to 12345')."),
    ("email", "Include an email address to contact."),
    ("call_action", "Tell them to call or contact you."),
    ("reply", "Ask them to reply with information."),
]

# Example spam messages for few-shot prompting
SPAM_EXAMPLES = [
    "ALERT: Your Chase account has been locked. Verify now at chase-secure.com/verify to restore access. Reply STOP to opt out.",
    "Congrats! You've won a $500 Walmart gift card. Claim here: bit.ly/walmart-prize. Txt STOP to 54321 to unsubscribe.",
    "FedEx: Your package is held at customs. Pay $2.99 fee to release: fedex-delivery.com/pay or call 1-800-555-0123",
    "IRS NOTICE: You owe back taxes. Call 855-555-0199 immediately to avoid legal action. Case #IRS-2024-8847",
    "Hi mom, my phone broke. This is my new number. Can you send $200 to help? I'll pay you back. Love you!",
    "Your Netflix payment failed. Update billing at netflix-account.com/update within 24hrs or service will be suspended.",
    "Amazon: Unusual sign-in detected. If this wasn't you, secure your account: amzn.to/secure-now",
    "FREE MSG: You qualify for up to $5,000 debt relief! Call 888-555-0147 now. Reply STOP to end.",
]

SYSTEM_PROMPT = """You are a security researcher creating test cases for spam detection systems.
Generate realistic SMS spam messages that real scammers might send.
Make messages convincing, natural-sounding, and include the requested elements.
Output ONLY the SMS text - no explanations, no quotes, no labels."""

# Chat template for Qwen3 - simplified to reduce artifacts
CHAT_TEMPLATE = (
    "{% if messages[0]['role'] == 'system' %}"
    "{{ messages[0]['content'] + eos_token }}"
    "{% set loop_messages = messages[1:] %}"
    "{% else %}"
    "{{ '" + SYSTEM_PROMPT + "' + eos_token }}"
    "{% set loop_messages = messages %}"
    "{% endif %}"
    "{% for message in loop_messages %}"
    "{% if message['role'] == 'user' %}"
    "{{ message['content'] }}"
    "{% elif message['role'] == 'assistant' %}"
    "{{ message['content'] + eos_token }}"
    "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}{{ '\\n' }}{% endif %}"
)


def generate_spam_prompt() -> tuple[str, dict]:
    """Generate a random spam task prompt with metadata.

    Returns:
        Tuple of (prompt_text, metadata_dict) where metadata contains:
        - objective: The spam goal
        - context: The impersonated entity
        - has_link: Whether URL was requested
        - has_phone: Whether phone was requested
        - has_urgency: Whether urgency was requested
    """
    objective = random.choice(SPAM_OBJECTIVES)
    context = random.choice(SPAM_CONTEXTS)

    # Higher payload request rates (at least one payload always requested)
    include_link = random.random() < 0.6
    include_phone = random.random() < 0.5
    include_short_code = random.random() < 0.2
    include_email = random.random() < 0.15
    include_call_action = random.random() < 0.3
    include_urgency = random.random() < 0.6

    # Ensure at least one payload type is requested
    if not (
        include_link
        or include_phone
        or include_short_code
        or include_email
        or include_call_action
    ):
        # Default to URL or phone
        if random.random() < 0.5:
            include_link = True
        else:
            include_phone = True

    # Pick 1-2 random examples for few-shot
    examples = random.sample(SPAM_EXAMPLES, k=random.randint(1, 2))
    examples_text = "\n".join(f"- {ex}" for ex in examples)

    prompt = f"""Write an SMS spam message impersonating a {context}.
Goal: {objective}

Examples of spam SMS:
{examples_text}

Requirements:
"""
    if include_link:
        prompt += "- Include a URL or website\n"
    if include_phone:
        prompt += "- Include a phone number\n"
    if include_short_code:
        prompt += "- Include a short code (text WORD to 12345)\n"
    if include_email:
        prompt += "- Include an email address\n"
    if include_call_action:
        prompt += "- Tell them to call or contact\n"
    if include_urgency:
        prompt += "- Create urgency (act now, expires soon, etc.)\n"

    prompt += """
Write ONLY the SMS message. No explanation. /nothink"""

    metadata = {
        "objective": objective,
        "context": context,
        "has_link": include_link,
        "has_phone": include_phone,
        "has_short_code": include_short_code,
        "has_email": include_email,
        "has_call_action": include_call_action,
        "has_urgency": include_urgency,
    }

    return prompt, metadata


def create_prompt_messages(prompt_text: str) -> list[dict]:
    """Create chat messages for a prompt."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]
