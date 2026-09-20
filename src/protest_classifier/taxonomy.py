"""The active, single-label annotation contract."""

from __future__ import annotations

from typing import Iterable

TAXONOMY_VERSION = "2026-09-20-v1"
OTHER_REASONS = ("outside_taxonomy", "insufficient_information")

CORE_RULE = (
    "Assign exactly one label: What change are the protesters primarily demanding, "
    "opposing, or defending? Do not classify from words such as farmers, students, "
    "migrants, or police alone. If several grievances appear, choose the one that "
    "is most explicitly described or receives the most emphasis. Use other only "
    "when no listed class fits or the note is too vague."
)

CLASS_NAMES = (
    "animal welfare",
    "blm",
    "women rights",
    "lgbtq",
    "climate and environment",
    "culture",
    "education",
    "farmers",
    "healthcare",
    "housing and rents",
    "immigration",
    "labor rights and wages",
    "palestine-israel conflict",
    "ukraine-russia war",
    "war and foreign policy",
    "pandemic",
    "unjust law enforcement",
    "anti-government and anti-establishment",
    "cost of living and inflation",
    "democratic institutions, corruption, elections, and reforms",
    "other",
)

DESCRIPTIONS = {
    "animal welfare": "Treatment, protection, slaughter, breeding, hunting, testing, or exploitation of animals.",
    "blm": "Anti-Black racism or Black liberation, especially but not limited to protests connected to Black Lives Matter.",
    "women rights": "Women’s equality, reproductive rights, gender-based violence, or discrimination against women.",
    "lgbtq": "Rights, safety, recognition, or discrimination involving LGBTQ people.",
    "climate and environment": "Climate change, pollution, ecological damage, conservation, land use, environmental health, or opposition to environmentally harmful projects.",
    "culture": "Arts, cultural heritage, monuments, language, cultural identity, or cultural institutions.",
    "education": "Schools, universities, teachers, students, curricula, exams, tuition, or education policy.",
    "farmers": "Agricultural producers’ livelihoods, subsidies, crop/livestock prices, agricultural regulation, nitrogen regulation, fuel for farming, farmland access, or farmer working conditions.",
    "healthcare": "Hospitals, doctors, medical access, healthcare funding, treatment, public health services, or healthcare workers as such.",
    "housing and rents": "Rent, eviction, housing affordability, homelessness, housing supply, or residential development affecting housing access.",
    "immigration": "Migration status, asylum, refugees, deportation, borders, migrant rights, or immigration policy.",
    "labor rights and wages": "Pay, working conditions, unions, strikes, employment security, pensions, working hours, or labor law.",
    "palestine-israel conflict": "Events primarily concerning the Israel–Palestine conflict or solidarity with either side.",
    "ukraine-russia war": "Events primarily concerning Russia’s war against Ukraine or solidarity with either side.",
    "war and foreign policy": "Other wars, military interventions, sanctions, international diplomacy, foreign occupation, or external relations not covered by the two dedicated conflict classes.",
    "pandemic": "COVID-19 or another epidemic/pandemic, including lockdowns, vaccination mandates, quarantine, or pandemic restrictions.",
    "unjust law enforcement": "Police abuse, arrests, prosecution, court treatment, detention, surveillance, or direct misuse of coercive legal authority.",
    "anti-government and anti-establishment": "Explicit generalized opposition to the government, state, political system, or establishment when no more specific grievance is stated.",
    "cost of living and inflation": "Prices or affordability of everyday necessities such as food, fuel, energy, utilities, or taxation when framed as household economic pressure.",
    "democratic institutions, corruption, elections, and reforms": "Elections, electoral integrity, corruption, constitutional reform, democratic institutions, political accountability, or governance reform.",
    "other": "A genuine topic outside the taxonomy, or notes too vague to identify a central grievance.",
}

BOUNDARY_RULES = (
    "Residents opposing a pig farm because of pollution -> climate and environment.",
    "Pig owners opposing compulsory slaughter -> animal welfare.",
    "Farmers demanding better milk prices -> farmers.",
    "Farm workers demanding higher wages -> labor rights and wages.",
    "Rent increases -> housing and rents, even if described as part of the cost-of-living crisis.",
    "Fuel or food prices affecting the general public -> cost of living and inflation.",
    "A protest against a specific government policy -> that policy’s substantive class, not automatically anti-government and anti-establishment.",
    "General “down with the government” or anti-system protests with no concrete issue -> anti-government and anti-establishment.",
    "Corruption, election manipulation, or constitutional reform -> democratic institutions, corruption, elections, and reforms.",
    "Police violence or unjust prosecution -> unjust law enforcement, even if protesters also demand political reform.",
    "Environmental climate activism -> climate and environment; these are intentionally one class.",
    "Palestine/Israel and Ukraine/Russia always use their dedicated labels when that conflict is central; otherwise use war and foreign policy.",
)


def validate_label(label: str, other_reason: str | None = None) -> None:
    if label not in CLASS_NAMES:
        raise ValueError(f"Unknown label {label!r}; expected one of {CLASS_NAMES}")
    if label == "other" and other_reason not in OTHER_REASONS:
        raise ValueError("other requires outside_taxonomy or insufficient_information")
    if label != "other" and other_reason not in (None, ""):
        raise ValueError("other_reason is only valid for other")


def accepted(prediction: str, primary: str, alternatives: Iterable[str] = ()) -> bool:
    """Lenient scoring: exact primary or an independently recorded alternative."""
    return prediction == primary or prediction in set(alternatives)
