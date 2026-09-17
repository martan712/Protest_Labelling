"""Versioned, explainable trigger rules used for weak label generation."""

from __future__ import annotations

import re

RULE_VERSION = "rules-v2"

CLASSES_DIC = {
    "blm": ["black lives matter", "black lives", "anti-black"],
    "lgbtq": ["lgb", "lesbian", "gay", "homosexual", "transsexual", "queer", "homophobia", "transphobia", "biphobia", "trans rights", "lgbtq"],
    "women rights": ["women's rights", "feminism", "feminist", "against women", "women protested", "abortion", "sexual violence", "sexual assault", "sexual harassment", "sexual abuse"],
    "immigration": ["migrants", "immigration", "against migration", "deportation detention", "deportation", "asylum seeker"],
    "unjust law enforcement": ["police brutality", "criminalize protests", "criminalize demonstrations", "police misconduct", "police repression", "police violence", "police abuse"],
    "discrimination": ["discrimination", "racism", "racial discrimination"],
    "climate": ["climate change", "fossil fuels", "greenwashing", "climate agenda", "global warming", "climate crisis", "climate protection", "emissions"],
    "palestine-israel conflict": ["gaza", "palestine", "israel", "hamas", "palestinian"],
    "animal welfare": ["species extinction", "animal welfare", "animal rights", "animal protection", "bullfighting", "animals locked", "wildlife", "cruelty to animals"],
    "farmers": ["farmers", "agriculture", "agricultural", "intensive farming"],
    "labor rights": ["labor agreement", "labour agreement", "wages", "wage increase", "pay rise", "pay increase", "rights of workers", "labor rights", "labour rights", "higher salaries", "working conditions", "labor conditions", "labour conditions", "commission fees", "pension", "salary equalization", "unfairly dismissed", "unfair dismissal", "dismissed workers", "contract workers", "employment contract", "factory closure", "workload", "working hours", "suspension of workers"],
    "health care": ["healthcare", "health care", "hospital", "hospitals", "emergency clinics", "emergency care", "medical care"],
    "environment": ["environmental", "the environment", "pfas", "nitrogen", "planned felling", "biodiversity", "park project", "pollution", "deforestation"],
    "public services": ["collapse of a concrete canopy", "canopy collapse", "bus", "traffic accidents", "railway station", "train station", "bike lanes", "road connection", "public service", "pedestrianization", "child-safe intersections", "bike street", "play street", "reasonable mobility", "cycling conditions", "urban development", "free transport"],
    "ukraine-russia war": ["russia", "ukrain", "ukraine", "war in ukraine", "peace in ukraine"],
    "housing": ["residential complex", "dignified housing", "evict", "eviction", "rent increase", "social housing"],
    "culture": ["tourism", "tourists", "cultural sector", "cultural workers"],
    "policies & politics": ["social welfare", "social services", "social assistance", "economic justice", "economic sovereignty", "economic independence", "adoption of the euro", "euro adoption", "council's plan", "nightlife noise", "municipality", "regional government", "political criticism", "political opposition", "against the pm", "resignation of the president", "political rights", "political prisoners", "anti-eu", "pro-eu", "democratic", "referendums", "urgent elections", "distinct autonomy"],
    "pandemic": ["pandemic", "covid", "coronavirus"],
    "education": ["education", "teacher", "academic", "professor", "university", "student loan", "school closure"],
}


def _pattern(phrase: str) -> re.Pattern[str]:
    return re.compile(r"(?<!\w)" + re.escape(phrase.lower().strip()) + r"(?!\w)")


RULE_PATTERNS = {label: [(phrase, _pattern(phrase)) for phrase in phrases] for label, phrases in CLASSES_DIC.items()}
RULE_ANY = {
    label: re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p.lower().strip()) for p in phrases) + r")(?!\w)")
    for label, phrases in CLASSES_DIC.items()
}


def match_rules(text: str) -> dict[str, list[str]]:
    """Return every class and phrase matched in the original note."""
    value = str(text or "").lower()
    return {
        label: [phrase for phrase, pattern in RULE_PATTERNS[label] if pattern.search(value)]
        for label in RULE_PATTERNS
        if RULE_ANY[label].search(value)
    }


def rule_label(matches: dict[str, list[str]]) -> str | None:
    """Return an unambiguous rule label; conflicts deliberately remain unresolved."""
    return next(iter(matches)) if len(matches) == 1 else None


def strip_triggers(text: str) -> str:
    """Remove all matched trigger phrases while preserving other note text."""
    output = str(text or "")
    phrases = sorted({phrase for values in CLASSES_DIC.values() for phrase in values}, key=len, reverse=True)
    for phrase in phrases:
        output = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", " ", output, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", output).strip()
