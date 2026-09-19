# A from-scratch approach to labeling European protest events

## Decision

The immediate assignment is to label the themes of protest events across
Europe. The model must assign reliable themes to unseen protest notes and
produce an auditable event-level dataset. Geographic and temporal trend
analysis is a later phase and is not required to complete the labeling system.

Do not continue optimizing the current 21-class classifier. Redefine the task
as **evidence-grounded extraction of protest claims**, followed by multi-label
coding on separate analytical dimensions.

The desired output is not only "the topic of this event is X." It is a
traceable answer to questions such as:

- What grievance or demand was expressed?
- Which issue domains and specific issues does it concern?
- Does the event express more than one theme?
- Which exact words in the note support each assigned theme?
- Is the note informative enough to make those claims?

This is a different objective, data model, annotation task, model, and
evaluation protocol. Keywords can remain useful for finding cases and auditing
coverage, but must no longer generate training targets. The final product is a
versioned event-level label dataset with evidence and confidence, not merely a
checkpoint containing a classifier.

## Problem recovered from the project report

The report defines the relevant part of the intended product more precisely
than the repository README:

- automatically classify the main themes of European protests;
- provide a reusable labeled dataset from which Clingendael can study patterns
  later.

The present phase contains two tasks that should be evaluated separately:

| Intended question | Proper task |
| --- | --- |
| What did protesters express? | Text extraction and multi-label coding |
| Does the system work on later periods and new contexts? | Forward-time and geographic generalization testing |

Here, “prediction” means assigning themes to unseen notes. Aggregating those
labels into trends, forecasting future protest counts, and predicting violent
outcomes are outside the present phase.

## Why the current objective should be retired

### The labels do not describe one coherent variable

The taxonomy mixes several kinds of concepts:

- issue domains: `housing`, `education`, `environment`;
- participant or sector identities: `farmers`, `women rights`, `blm`;
- specific events or periods: `pandemic`, `ukraine-russia war`,
  `palestine-israel conflict`;
- state response: `unjust law enforcement`;
- a broad residual: `policies & politics`.

These are neither mutually exclusive nor at the same level of abstraction.
For example, a protest can simultaneously concern workers, climate policy, and
government subsidies. Forcing it into one class discards information and makes
the correct answer depend on arbitrary precedence rules. The accepted-pair
metric compensates for this after the fact, but does not repair the target.

### The training objective is mainly rule imitation

The current generated candidate artifact contains 218,108 events. It reports
106,119 unambiguous rule labels, 111,489 unresolved rows, and 26,453 rule
conflicts. The selected release then balances the unambiguous rule outputs
rather than representing the event population.

Of 46,849 training rows:

- 46,349 are keyword/rule-labelled;
- 500 are independently reviewed by an LLM;
- none of those 500 are recorded as human-adjudicated;
- `other` has only 48 examples.

The model can therefore improve its training loss primarily by reproducing the
keyword system. More training on this release cannot establish that it recovers
the themes actually expressed in protest notes.

These counts describe the current repository artifacts, not independently
verified facts about the source dataset. The redesign does not depend on their
exact values; it depends on the documented fact that keyword rules generate
almost all training targets.

### The evaluation supports the diagnosis, but is not yet gold standard

The legacy model scores 57.5% strict and 69.0% accepted-pair accuracy on the old
200-event development set. The stored `release21-head-500` report scores 28.5%
strict and 41.5% accepted on those same 200 rows. The error reviews identify
participant shortcuts (students causing `education`), venue/object shortcuts
(`house` causing `housing`), and categories with no fitting answer.

The newer 500-row development file is useful for exploration but every row is
marked `independent_llm_review`; it is not human gold. The fresh 1,000-event test
queue is still unannotated. There is also a reporting inconsistency to resolve:
`current_approach.md` states 64.0%/68.8% on 500 rows, while the stored scorer
artifact for the named release contains 28.5%/41.5% on 200 rows. Every future
score must identify the exact model, manifest, schema, and gold-file hashes.

### “Other” reveals missing dimensions, not just a missing class

The reviewed `other` examples include nuclear disarmament, anti-mafia action,
hostage release, human trafficking, military intervention, public safety, and
justice for individual deaths. Adding a larger residual class would hide these
analytically meaningful claims. The right response is to design dimensions from
the research questions and observed data, not to make the residual absorb them.

## Unit of labeling: an event with grounded claims

The released unit remains the ACLED event. Because one event may contain zero,
one, or several themes, represent its labels through grounded claims and roll
their theme tags up to the event. Store each claim as a child record linked to
the event, then produce one event row with the union of its validated theme
labels. A minimal schema is:

| Field | Meaning |
| --- | --- |
| `event_id` and `claim_id` | Stable event and within-event claim identifiers |
| `information_status` | `motive_stated`, `motive_not_stated`, `stated_out_of_scope`, or `unclear` |
| `evidence_span` | Exact text supporting the extracted claim |
| `claim_summary` | Short normalized grievance or demand, faithful to the note |
| `stance` (supporting) | `support`, `oppose`, `demand_change`, `seek_redress`, `commemorate`, or `unclear` |
| `issue_domains` | Zero or more broad, stable analytical domains |
| `issue_tags` | Zero or more specific issues under those domains |
| `affected_groups` (supporting) | Groups whose interests or rights are at issue |
| `claimant` (supporting) | Protester/organizer identity when stated |
| `target` (supporting) | Government, employer, police, company, foreign state, etc. |
| `confidence` | Calibrated confidence for automatic use |
| `review_status` | Automatic, human-reviewed, or adjudicated |

The required event-level output contains `event_id`, zero or more
`theme_labels`, per-theme probabilities, `information_status`, references to
the supporting claim/evidence rows, schema/model versions, and `review_status`.
That is the primary deliverable; claim rows provide its audit trail.

The supporting fields are not separate headline products. They keep participant
identity, affected group, issue, target, and event context apart so the theme is
assigned correctly and can be audited. Thus “nurses demanded higher pay” is a
labor/pay claim made by health workers; it need not become a health-care issue.
“Students opposed a coal mine” is an environment/climate claim made by students;
it need not become an education issue.

For example:

```json
{
  "information_status": "motive_stated",
  "evidence_span": "workers ... protested the closure of their factory",
  "claim_summary": "prevent the factory closure",
  "stance": "oppose",
  "issue_domains": ["work_and_livelihoods"],
  "issue_tags": ["job_loss", "workplace_closure"],
  "claimant": ["workers"],
  "target": ["employer"]
}
```

## Build the vocabulary from the research questions and the data

Do not translate the existing 21 labels directly into a new list. Use this
process instead:

1. Write the labeling questions first: which expressed themes must be
   distinguishable, which combinations are valid, and when should the system
   return no label or abstain? A field belongs in the schema only if it supports
   reliable theme assignment or quality control.
2. Open-code a representative pilot of roughly 300 events. Include all years
   and major countries, plus deliberately selected thin, multi-claim, rare, and
   non-English/source-translation cases.
3. Have two domain-informed humans independently mark evidence spans and
   describe claims without seeing keywords, old labels, or model predictions.
4. Cluster the open codes into a shallow hierarchy. Broad domains should be
   stable and mutually intelligible; specific issue tags may evolve. Keep
   identities, geopolitical contexts, and tactics in their own dimensions.
5. Test the guide on another blind batch. Revise definitions where annotators
   disagree systematically, then freeze schema version 1 before building the
   benchmark.

A plausible starting set of broad domains—not a final taxonomy—is:

- work and livelihoods;
- public goods and services;
- governance and political institutions;
- rights, equality, and justice;
- environment and climate;
- security, war, and foreign affairs;
- place, community, and culture;
- other stated claim;
- motive not stated.

Specific issues such as wages, pensions, hospital access, abortion, rent,
police violence, fossil fuels, migration policy, Gaza, or Ukraine sit below or
beside these domains as multi-label tags. Time-bound conflicts should be context
tags unless the research question explicitly treats each as an issue class.

## Annotation and benchmark design

### Sampling

Create two kinds of data and never conflate their metrics:

1. A probability sample from the natural event population for headline label
   quality.
2. Enriched diagnostic samples for rare issues, countries, years, ambiguous
   notes, multi-claim events, and likely model failures.

Split by semantic duplicate/campaign group, not only exact note hash. Add a
forward-time test slice so that performance on new events is visible. Freeze
the raw snapshot and all split IDs before model development.

The existing 1,000 fresh test IDs may be reused if their sampling is suitable,
but they must be annotated from scratch under the new schema. The 500 LLM
annotations and the old error reviews should inform the guide and audit strata,
not serve as final gold labels.

### Human labeling

- Double-label the pilot, development set, final test set, and a meaningful
  sample of training rows.
- Adjudicate all dev/test disagreements and retain the pre-adjudication labels
  to measure agreement.
- Require an evidence span for every positive issue or extracted claim.
- Allow multiple claims and multiple tags without “primary label” precedence.
- Mark absent information as `information_status=motive_not_stated`; do not
  guess a topic.
- Version the guide and retain annotator, timestamp, note hash, and adjudication
  history.

Use agreement as a schema diagnostic, reported separately for evidence spans,
claim existence, broad domains, and specific tags. Low agreement is a reason to
revise or merge a concept, not a reason to add lenient scoring pairs.

## Concrete modeling plan

### Models

Use [`answerdotai/ModernBERT-base`](https://huggingface.co/answerdotai/ModernBERT-base)
as the encoder. It is an English, encoder-only model with 149 million
parameters. The ACLED notes in the current snapshot are English and short: the
median is 41 words, 99% are at most 115 words, and the maximum is 430 words.
An encoder classifier is therefore a better fit than generative T5.

Train two models sequentially from the same base checkpoint:

1. **Event classifier:** one ModernBERT encoder with a four-class information
   status head and a multi-label theme head.
2. **Evidence extractor:** a theme-conditioned ModernBERT token classifier that
   marks the note tokens supporting each predicted theme.

The classifier is the required labeling model. The evidence extractor makes
its labels auditable and acts as a final validation step. The two models are
kept separate so classification and evidence extraction can be evaluated and
retrained independently. Pin the downloaded model revision in every run
manifest rather than relying on the moving model name.

Also train one non-neural baseline: word and character TF-IDF features with a
one-vs-rest logistic regression classifier. ModernBERT must beat this baseline
on the locked development set to justify its additional complexity.

### Input

The event classifier receives only the original ACLED `notes` value:

```text
On 28 May 2022, workers protested outside the factory against its planned
closure and the expected loss of 300 jobs.
```

Do not prepend country, year, actor, the old class, matched keywords, or model
predictions. Those fields create shortcuts and are unnecessary for identifying
what the note says. They remain in the output for provenance and later analysis.

Do not use `clean_notes`. Do not lowercase, stem, lemmatize, remove stopwords,
remove keywords, or discard the first tokens. Preserve negation, word order,
names, and the exact text needed for evidence offsets.

Preprocessing is limited to:

1. reject or separately mark missing notes;
2. normalize line endings and invalid control characters;
3. store a SHA-256 hash of the untouched note;
4. tokenize with the ModernBERT tokenizer;
5. dynamically pad each batch rather than padding the dataset to a fixed size.

Use `max_length=256`. Notes longer than 256 tokens are split into overlapping
windows of 256 tokens with 32 tokens of overlap. Most events have one window.
For long events, take the maximum probability for each theme across windows;
keep the window and offsets that produced the supporting evidence. Annotation
evidence determines which training windows contain each positive theme, so a
theme is not incorrectly copied onto unrelated chunks.

### Targets

Freeze an ordered list of `K` specific theme tags in `schema/themes.yaml`.
Represent an event's tags as a length-`K` multi-hot vector: several entries may
be 1, and all may be 0. Each specific tag maps to one broad domain in the same
file. Derive broad domains from predicted tags rather than training another
potentially inconsistent output head. Include a `*_general` tag within a domain
when the note supports the broad theme but not a narrower tag.

The information-status target is one of:

- `motive_stated`: at least one in-scope grievance or demand is expressed;
- `motive_not_stated`: the event is a protest but the note gives no motive;
- `stated_out_of_scope`: a clear motive is expressed but no frozen theme fits;
- `unclear`: the note is too ambiguous to decide without review.

For every positive theme, annotators must select one or more supporting
character spans. These are converted to binary token targets using the
tokenizer's offset mapping. Query and special tokens receive target `-100` and
are ignored by the evidence loss.

### Event-classifier architecture and loss

Pass each tokenized window through `AutoModel.from_pretrained(...)`. Apply
dropout (`p=0.1`) to the pooled first-token representation, then use:

- a linear `hidden_size -> 4` information-status head with softmax;
- a linear `hidden_size -> K` theme head with independent sigmoid outputs.

Train both heads jointly:

```text
total_loss = BCEWithLogits(theme_logits, multi_hot_tags)
           + 0.25 * CrossEntropy(status_logits, information_status)
```

Start with unweighted binary cross-entropy. If rare-tag recall is unusable,
compare inverse-square-root class weights capped at 5 on development data;
never select weighting from the locked test. Do not oversample the evaluation
sets or manufacture balanced prevalence.

### Evidence-extractor architecture and loss

For each positive event-theme pair, construct a paired input:

```text
sequence A: Theme: workplace closure. Includes opposition to closing a factory
            or work site and consequent job loss.
sequence B: <original ACLED note>
```

The text and inclusion/exclusion definition come from `themes.yaml`, not from a
keyword list. Feed the pair to a second ModernBERT encoder with a binary token
head. Train with token-level cross-entropy to classify tokens from sequence B
as `evidence` or `not_evidence`; ignore sequence A and special tokens. Include
negative event-theme pairs sampled from plausible confusions so the extractor
can return no evidence. Weight evidence tokens by the inverse square root of
their training frequency, capped at 10, because most note tokens are not part
of an evidence span.

At inference, run the extractor only for themes passing a low classifier
candidate threshold (initially 0.15). Merge adjacent evidence tokens and map
them back to exact character spans. An automatically published theme must have
at least one evidence span; otherwise route it to review.

### Human data and splits

Use only human-reviewed labels as supervised targets. Legacy keyword labels and
the current LLM-only reviews may select cases for annotation or define audit
slices, but they do not enter the loss.

Build the first dataset as follows:

1. **Schema pilot:** 300 events, independently labeled by two humans and
   adjudicated. Revise the schema, then relabel these under the frozen version;
   they may subsequently enter training.
2. **Initial training set:** 1,200 additional events. Double-label a 20% random
   overlap and adjudicate disagreements, all unclear cases, and all new tags.
3. **Development set:** 400 independently double-labeled and adjudicated events.
4. **Locked test set:** 400 independently double-labeled and adjudicated events,
   not inspected until the recipe and thresholds are frozen.
5. **Forward-time diagnostic:** 200 later-period events, kept separate from the
   headline test to measure temporal transfer.

Sample train/dev/test from the natural event population, then add a documented
enrichment layer for rare themes and difficult cases. Report headline metrics
on the natural sample and diagnostic metrics on the enriched rows separately.

Keep exact duplicates and near-duplicate campaign reports in one split. Start
with exact note hashes plus character 3–5-gram TF-IDF cosine similarity, join
pairs at similarity `>= 0.90`, and manually audit large connected components.
Freeze group IDs before fitting the model.

### Training configuration

Use the following initial configuration for both ModernBERT models:

| Setting | Value |
| --- | --- |
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Weight decay | `0.01` |
| Effective batch size | `32` via gradient accumulation |
| Maximum length | `256` tokens |
| Epochs | maximum `10` |
| Early stopping | patience `2` on development macro-F1 |
| Warm-up | first `10%` of optimizer steps |
| Gradient clipping | norm `1.0` |
| Precision | BF16/FP16 only when supported; otherwise FP32 |
| Seeds | `17`, `42`, and `83` for the final candidate |

Use dynamic padding and length-bucketed batches to fit the existing AMD/CPU
environment. Save a checkpoint after every epoch. Choose the epoch and all
thresholds on development data only. Record the resolved base-model revision,
schema version, data hashes, split manifest, seed, and hyperparameters.

### Calibration, thresholds, and abstention

Fit per-theme temperature scaling on the development set after training. For
each tag, choose the lowest threshold that reaches development precision of at
least 90% while maximizing coverage. A tag with fewer than 30 positive
development examples or one that cannot reach that precision remains
`review_only`; do not silently use a global `0.5` threshold.

Inference for one event is:

1. tokenize and run the event classifier;
2. aggregate long-window probabilities;
3. calibrate probabilities and apply per-theme thresholds;
4. run the evidence extractor for candidate themes;
5. publish themes that pass both the classifier threshold and evidence check;
6. otherwise emit `needs_review`, `motive_not_stated`,
   `stated_out_of_scope`, or `unclear` as appropriate.

The event must never be forced into a substantive theme. Store all calibrated
probabilities, thresholds, evidence spans, and the reason for abstention.

One event prediction has this shape:

```json
{
  "event_id": "DEU12345",
  "information_status": "motive_stated",
  "themes": [
    {
      "tag": "workplace_closure",
      "domain": "work_and_livelihoods",
      "probability": 0.94,
      "threshold": 0.81,
      "evidence": ["against its planned closure and the expected loss of 300 jobs"]
    }
  ],
  "decision": "auto_labeled",
  "schema_version": "themes-v1",
  "model_version": "<checkpoint hash>"
}
```

### Files and commands to implement

Keep the new pipeline separate from the legacy notebooks:

```text
schema/themes.yaml
labeling_v2/build_annotation_queue.py
labeling_v2/validate_annotations.py
labeling_v2/build_splits.py
labeling_v2/model.py
labeling_v2/train_classifier.py
labeling_v2/train_evidence.py
labeling_v2/calibrate.py
labeling_v2/predict.py
labeling_v2/evaluate.py
tests/test_labeling_v2_*.py
```

`predict.py` writes:

- `event_labels.parquet`: one row per event with status, predicted themes,
  probabilities, evidence, and provenance;
- `claim_evidence.parquet`: one row per event-theme-evidence span;
- `review_queue.csv`: abstentions and failed evidence checks ordered by
  uncertainty;
- `run_manifest.json`: exact model, data, schema, and code versions.

Implementation is complete only when a single command can rebuild splits,
train, calibrate, score the locked manifest, and label the full event snapshot.

## Phase II — trend analysis (deferred)

The event-labeling release should retain event IDs, dates, countries, theme
probabilities, evidence, schema/model versions, and review status so a later
phase can aggregate labels into geographic and temporal trends. No trend
dashboard, prevalence correction, or trend validation is required in the
current phase.

When that phase begins, it will need separate requirements and evaluation.
Multi-label counts will not sum to the number of events, model errors can bias
changes over time, and changes in ACLED coverage can resemble real changes in
public concern. Those issues should not complicate the current labeling MVP,
but the event-level output must preserve enough provenance to address them
later.

## Evaluation that matches the product

Replace accepted-pair accuracy with component-level labeling measures:

| Component | Primary measures |
| --- | --- |
| Information status | Macro-F1, per-status precision/recall, and calibration |
| Evidence extraction | Exact match and token-overlap F1 |
| Claims per event | Claim detection precision/recall after adjudicated matching |
| Broad domains | Micro/macro multi-label F1 and per-domain precision/recall |
| Specific tags | Per-tag average precision, F1, and coverage |
| Abstention | Risk-versus-coverage curve and error rate among auto-published rows |

Also report performance for thin notes, multi-claim events, later years,
countries, languages/source styles, rare tags, and campaign-held-out events.
Bootstrap confidence intervals at the campaign group level.

The operational release gate should be tied to use. For example, an
auto-published tag might require at least 90% precision on locked human test
data, with all remaining cases abstained or reviewed. Coverage must always be
reported beside that precision. Do not set a universal 95% accuracy target
before label prevalence, annotation agreement, and error costs are known.

Trend accuracy is deliberately not a release criterion for the current phase.

## Phase I — event labeling implementation sequence

### Step 0 — labeling contract

Agree with the intended users on the required theme granularity, valid
multi-theme combinations, abstention behavior, and the cost of false positives,
false negatives, and manual review. This fixes the labeling objective before
any taxonomy or model choice.

**Deliverable:** labeling requirements, output schema draft, and success
criteria.

### Step 1 — schema pilot

Open-code and double-annotate the pilot, then freeze the first guide. Implement
schema validation and a small annotation interface that hides all legacy
predictions.

**Deliverable:** versioned schema/guide, adjudicated pilot, agreement report.

### Step 2 — locked benchmark

Construct population and diagnostic samples, cluster campaign duplicates, and
human-annotate development and test partitions. Keep test sealed.

**Deliverable:** manifests, human gold, and one scorer.

### Step 3 — baselines

Evaluate a motive-only baseline, a constrained extraction baseline, and a
multi-label classifier. Select models on the new development measures and
review burden, not legacy class accuracy.

**Deliverable:** comparable model cards and error analysis.

### Step 4 — scale human supervision

Use active learning plus a continuing random sample. Train or fine-tune a local
model only when the learning curve shows that more reviewed examples improve
the locked objective. Weak rules may be auxiliary features or noisy-label
sources only after their precision is measured against humans.

**Deliverable:** versioned training release with no unlabeled row presented as
gold.

### Step 5 — publish event labels

Run calibrated inference, route abstentions, and generate claim-level and
event-level tables with evidence, probabilities, review status, and clear
schema/model versions.

**Deliverable:** auditable, versioned labels for the in-scope protest events.

## What to retain and what to stop

Retain:

- stable event IDs, note hashes, provenance, and frozen raw snapshots;
- the separation of training and held-out manifests;
- error reviews as qualitative design evidence;
- duplicate/campaign grouping work, extended beyond exact hashes;
- tests and reports that enforce provenance and leakage controls.

Stop:

- expanding the keyword dictionary as the main labeling strategy;
- balancing to 3,000 examples per legacy class;
- training a 21-way softmax to fill every unknown event;
- treating LLM-only reviews as human gold;
- using accepted pairs to conceal taxonomy overlap;
- selecting checkpoints against a metric that does not measure the eventual
  analytical use.

The first new code should therefore support the schema pilot and benchmark—not
another T5 training run.
