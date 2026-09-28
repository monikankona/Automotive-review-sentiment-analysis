import re
from typing import Dict, List, Tuple

# Aspect vocabulary follows the Module 3 project brief.
ASPECT_KEYWORDS: Dict[str, List[str]] = {
    "Vehicle": ["vehicle", "car", "automobile", "model"],
    "Engine": ["engine", "motor", "powertrain", "horsepower", "torque"],
    "Battery": ["battery", "charging", "charge", "ev battery", "battery pack"],
    "Mileage": ["mileage", "fuel economy", "range", "kmpl", "efficiency"],
    "Safety": ["safety", "airbag", "brake", "braking", "abs", "adas", "crash"],
    "Comfort": ["comfort", "seat", "seats", "ride", "suspension", "cabin"],
    "Service": ["service", "maintenance", "dealer", "workshop", "support"],
    "Infotainment": ["infotainment", "screen", "display", "audio", "speaker", "navigation", "bluetooth"],
    "Price": ["price", "cost", "expensive", "affordable", "value", "emi"],
}


def find_aspects(text: str) -> List[Tuple[str, str]]:
    """Return (aspect, keyword) matches, preserving first-match order."""
    lowered = text.lower()
    found: List[Tuple[str, str]] = []
    for aspect, keywords in ASPECT_KEYWORDS.items():
        for keyword in keywords:
            if re.search(r"(?<!\w)" + re.escape(keyword.lower()) + r"(?!\w)", lowered):
                found.append((aspect, keyword))
                break
    return found


def aspect_context(clause: str, keyword: str) -> str:
    """Keep the clause local so sentiment is less likely to bleed across aspects."""
    tokens = clause.split()
    k_tokens = keyword.split()
    lower = [t.lower().strip(".,;:!?()") for t in tokens]
    try:
        start = next(i for i in range(len(lower)) if lower[i:i+len(k_tokens)] == k_tokens)
    except StopIteration:
        return clause
    left = max(0, start - 12)
    right = min(len(tokens), start + len(k_tokens) + 12)
    return " ".join(tokens[left:right])
