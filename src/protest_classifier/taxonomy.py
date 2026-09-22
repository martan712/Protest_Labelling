"""The active, single-label annotation contract."""

from __future__ import annotations

from typing import Iterable

TAXONOMY_VERSION = "2026-09-21-v3"
OTHER_REASONS = ("outside_taxonomy", "insufficient_information")

CORE_RULE = (
    "Assign exactly one primary label for the substantive domain the protest concerns, "
    "and optionally one secondary label in alternative_labels when another domain "
    "is also supported by the note as a secondary theme or a defensible alternative "
    "interpretation. Apply the same rules to both labels; leave alternative_labels "
    "empty when no second label is justified. "
    "Use the stated demand to identify that domain. When the demand is vague or "
    "unspecified, use the protesting group's role or the affected institution as "
    "the default domain. When an explicit grievance points to another domain, "
    "that grievance takes precedence over the group or institution. For example, "
    "students protesting for funding -> education; students protesting corruption "
    "-> democratic institutions, corruption, elections, and reforms. If several "
    "grievances appear, choose the one that is most explicitly described or receives "
    "the most emphasis, applying the category-specific boundary rules. Use other "
    "when no listed class fits or neither the demand nor the group/institution "
    "provides enough information to identify a domain."
)

CLASS_NAMES = (
    "animal welfare",
    "racism",
    "anti-fascism and extremism",
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
    "crime, violence and victim justice",
    "civil liberties and censorship",
    "cost of living and inflation",
    "democratic institutions, corruption, elections, and reforms",
    "other",
)

DESCRIPTIONS = {
    "animal welfare": "Treatment, protection, slaughter, breeding, hunting, testing, or exploitation of animals.",
    "racism": "Opposition to racism, racial discrimination, racist violence, or racist representation, including anti-Black racism and Black liberation.",
    "anti-fascism and extremism": "Opposition to fascist, neo-Nazi, far-right, or extremist organizations, movements, events, or ideology.",
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
    "crime, violence and victim justice": "Opposition to crime, violence, human trafficking, or organized crime; solidarity with and commemoration of victims; demands for investigation, justice for victims, accountability of perpetrators, or protection from crime.",
    "civil liberties and censorship": "Freedom of speech, the press, assembly, association, religion, privacy, or access to information; opposition to censorship or restrictions on these freedoms.",
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
    "A protest against a specific local policy, regulation, closure, restriction, or decision, where no broader taxonomy domain is the subject -> other.",
    "Shopkeepers opposing a street closure because they lose customers, or fishers opposing catch-reporting or catch-weighing regulations because of business burdens -> other (outside_taxonomy). Economic loss alone does not establish wages, farmers, or household inflation.",
    "A specific closure with a covered domain retains that domain: emergency room -> healthcare; theatre -> culture; school -> education. Explicit corruption or pollution in a local decision establishes the corresponding domain.",
    "Anti-government is not an available label. Classify the substantive domain when supported; otherwise use other. Opposition to a government or politician alone does not establish corruption or a democratic-institutions grievance.",
    "Opposition to fascist, neo-Nazi, far-right, or extremist groups, events, or ideology -> anti-fascism and extremism.",
    "Corruption, election manipulation, or constitutional reform -> democratic institutions, corruption, elections, and reforms.",
    "Police violence or unjust prosecution -> unjust law enforcement, even if protesters also demand political reform.",
    "Human trafficking, justice for a murder victim, or commemorating Giovanni Falcone while opposing the mafia -> crime, violence and victim justice. Asking for justice alone does not imply misconduct by police or courts.",
    "Alleged corrupt judges, biased trials, wrongful arrests, or deliberate judicial cover-ups -> unjust law enforcement when the protest concerns abuse in the handling of justice; broader institutional corruption or judicial independence -> democratic institutions, corruption, elections, and reforms.",
    "Censorship, media bans, or restrictions on speech, assembly, association, or religion -> civil liberties and censorship. Specific abusive arrests, detention, or prosecution -> unjust law enforcement. Apply the primary and optional secondary label rules when both themes are supported.",
    "Surveillance opposed as a restriction on privacy -> civil liberties and censorship; surveillance opposed as abuse in a specific police investigation -> unjust law enforcement.",
    "Explicit gender-based violence, racist violence, or anti-fascist mobilization retains its specific domain label when central; crime, violence and victim justice may be secondary when justified.",
    "Environmental climate activism -> climate and environment; these are intentionally one class.",
    "A protest whose primary target or subject is a foreign state, regime, government, foreign official, diaspora solidarity movement, foreign occupation, sanctions, debt, or other international relationship -> war and foreign policy, even when the foreign event involves repression or rights violations. Palestine/Israel and Ukraine/Russia use their dedicated labels when those conflicts are central.",
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
