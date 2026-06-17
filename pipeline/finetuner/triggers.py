"""Strip the trigger words (from 1-preprocess.ipynb) out of the training text.

The labeler tags each event by the first trigger keyword found in its notes, so every
labeled row literally contains its class's keyword. Training on that text lets the model
keyword-match instead of learning the topic, and the (also leaked) val/test splits hide
it. Removing the keywords forces the model to use real context. Applied in the finetuner
only — inference (notebook 3) keeps the full text.
"""

import re

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

for _pkg in ("wordnet", "stopwords"):
    nltk.download(_pkg, quiet=True)

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))

# Same spelling normalization the labeler applied before matching (1-preprocess.ipynb),
# so US-spelled triggers strip British-spelled tokens too.
BRITISH_TO_US = {
    'labour': 'labor', 'criminalis': 'criminaliz', 'organis': 'organiz',
    'mobilis': 'mobiliz', 'nationalis': 'nationaliz', 'privatis': 'privatiz',
    'recognis': 'recogniz', 'neighbour': 'neighbor', 'behaviour': 'behavior',
    'colour': 'color', 'honour': 'honor', 'favour': 'favor',
    'defence': 'defense', 'licence': 'license', 'centre': 'center', 'theatre': 'theater',
}

# Trigger words per class, copied verbatim from 1-preprocess.ipynb (Classes_dic).
CLASSES_DIC = {
    "blm": ["black lives matter"],
    "lgbtq": ["lgb", "lesbian", "gay", "homosexual", "transsexual", "queer", "homophobia", "transphobia", "biphobia", "trans rights"],
    "women rights": ["women's rights", "feminism", "feminist", "against women", "women protested", "abortion", "sexual violence", "sexual assault", "sexual harassment", "sexual abuse"],
    "immigration": ["migrants", "immigration", "against migration", "deportation detention"],
    "unjust law enforcement": ["police brutality", "criminalize protests", "criminalize demonstrations", "police misconduct", "police repression"],
    "discrimination": ["discrimination", "racism"],
    "climate": ["climate change", "fossil fuels", "greenwashing", "climate agenda", "global warming"],
    "palestine-israel conflict": ["gaza", "palestine", "israel", "hamas", "palestinian"],
    "animal welfare": ["species extinction", "animal welfare", "animal rights", "animal protection", "bullfighting", "animals locked", "wildlife", "cruel"],
    "farmers": ["farmers", "agriculture", "agricultural", "intensive farming"],
    "labor rights": ["labor agreement", " wages", "rights of workers", "labor rights", "higher salaries", "working conditions", "labor conditions", "commission fees", " pension", "salary equalization", "unfairly dismissed"],
    "health care": ["healthcare", "hospital ", "hospitals", "emergency clinics", "emergency care"],
    "environment": ["environmental", "the environment", "pfas", "nitrogen", "planned felling", "biodiversity", "park project"],
    "public services": ["collapse of a concrete canopy", "canopy collapse", " bus ", "traffic accidents", "railway station", "bike lanes", "road connection", "public service", "pedestrianization", "child-safe intersections", "bike street", "play street", "reasonable mobility", "cycling conditions", "urban development", "free transport"],
    "ukraine-russia war": [" russia", "ukrain"],
    "housing": ["residential complex", "dignified housing", "evict"],
    "culture": ["tourism", "tourists", "cultural sector"],
    "policies & politics": ["social welfare", "social services", "social assistance", "economic justice", "economic sovereignty", "economic independence", "adoption of the euro", "euro adoption", "council's plan", "nightlife noise", "municipality", "regional government", "political criticism", "political opposition", "against the pm", "resignation of the president", "political rights", "political prisoners", "anti-eu", "pro-eu", "democratic", "referendums", "urgent elections", "distinct autonomy"],
    "pandemic": ["pandemic", "covid", "coronavirus"],
    "education": ["education", "teacher", "academic", "education", "professor", "university", "student loan"],
}


def _normalize_spelling(text):
    for brit, us in BRITISH_TO_US.items():
        text = text.replace(brit, us)
    return text


def _process(text):
    """Same token pipeline as preprocess() in 1-preprocess.ipynb."""
    text = _normalize_spelling(str(text).lower())
    text = re.sub(r"\W+", " ", text)
    return [_lemmatizer.lemmatize(w) for w in text.split() if w not in _stop_words]


# Build matchers once from the processed triggers.
_exact = set()            # single tokens matched whole-word
_prefix = set()           # single tokens matched as stems (e.g. 'ukrain' -> ukraine)
_phrases = {}             # token-count -> set of contiguous token tuples
for _words in CLASSES_DIC.values():
    for _phrase in _words:
        _toks = _process(_phrase)
        if not _toks:
            continue
        if len(_toks) == 1:
            # A trailing space ('hospital ', ' bus ') guards the right boundary -> whole
            # word. Everything else is a stem; startswith already anchors the left side,
            # so leading-space triggers (' russia', ' wages') still match russian/wages.
            (_exact if _phrase != _phrase.rstrip() else _prefix).add(_toks[0])
        else:
            _phrases.setdefault(len(_toks), set()).add(tuple(_toks))

_prefix_t = tuple(sorted(_prefix))
_max_phrase = max(_phrases, default=1)


def strip_triggers(text):
    """Remove trigger keywords (and the phrases they came from) from one clean_notes."""
    toks = str(text).split()
    n = len(toks)
    out = []
    i = 0
    while i < n:
        matched = False
        for length in range(min(_max_phrase, n - i), 1, -1):
            if tuple(toks[i:i + length]) in _phrases.get(length, ()):
                i += length
                matched = True
                break
        if matched:
            continue
        tok = toks[i]
        if tok not in _exact and not (_prefix_t and tok.startswith(_prefix_t)):
            out.append(tok)
        i += 1
    return " ".join(out)
