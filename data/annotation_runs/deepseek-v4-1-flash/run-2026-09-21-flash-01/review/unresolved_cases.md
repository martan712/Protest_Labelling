# Unresolved / boundary cases — run-2026-09-21-flash-01

Recorded separately from the annotation CSVs. These are decisions made with a defensible
alternative; none were changed post hoc. They are the categories most likely to differ
between annotators. No taxonomy change was made.

## Review counts
- 7,337 rows schema-validated across 147 chunks (train 120, dev 10, test 17) and 6 assembled files: all PASS.
- 605 `other` rows checked for the two-reason contract: 564 `outside_taxonomy`, 41 `insufficient_information`.
- Pilot: all 11 `other` rows semantically reviewed against the reason definitions and group/institution fallback (11 accepted) — see `pilot_review.md`.
- Alternating-row spot review performed during review passes; no format/consistency faults found after re-validation.

## Recurring boundary categories (defensible either way)
1. **Foreign-regime / diaspora solidarity (rule 2) vs the concrete rights theme.** Repression-abroad,
   foreign-election, foreign-official and diaspora events → `war and foreign policy` primary with the
   concrete theme (women rights, lgbtq, climate, elections) secondary. A theme-first reading would invert
   primary/secondary for many of these (e.g. Iran Woman-Life-Freedom, Belarus, Venezuela, Serbian/Polish
   diaspora rows).
2. **Far-right / anti-Islamic *own* mobilizations.** The taxonomy defines `racism` as opposition to
   racism and `anti-fascism` as opposition to the far right, so far-right rallies are often `other`
   (`outside_taxonomy`), sometimes `immigration` where an anti-immigration subject is stated. Some rows
   were given `racism`/`lgbtq` topically. Directional-vs-topical reading is the main source of variance.
3. **Protest vs counter-protest ordering (rule 6).** Where a far-right rally is described first and an
   anti-fascist counter second, notes split between "first mobilization" primary and the anti-fascist
   counter as primary.
4. **Pandemic restrictions vs affected sector.** School/culture/hospitality closures: `pandemic` when the
   restriction is the grievance; `education`/`culture`/`labor` when the sector demand dominates.
   Boundary not fully settled.
5. **Unjust law enforcement vs crime/victim justice.** Court-treatment/prosecution/arrest grievances →
   `unjust law enforcement`; victims'-justice/accountability → `crime, violence and victim justice`.
   Alleged judicial cover-ups and disaster accountability (Novi Sad, floods, Tempe) split between these
   two and `democratic institutions…`.
6. **Business/regulatory/local-policy `other`.** Haulers/fishers/taxi/boat/bank-branch/road-routing/
   parking/infrastructure rows → `other/outside_taxonomy` (fisher analogy). Some could be read as
   `labor`, `immigration` or a sector class.
7. **Conscription / military-service law (domestic).** Mapped to `war and foreign policy` as the nearest
   class; `other` or `education` defensible.
8. **Disability / social welfare / general governance.** No disability or welfare class exists; mapped to
   `healthcare`, `education`, or `other/outside_taxonomy` on a best-fit basis.
9. **NIMBY siting.** Energy projects → `climate and environment` (land use); proximity/nuisance-only
   objections → `other`. Line is soft.
10. **Education/labor override.** Teacher and health-worker pay/conditions → `labor rights and wages`
    primary (explicit-wage override), with the sector as secondary only where a service demand is stated.

## Administrative note
`docs/annotation_handoff.md` shows an uncommitted working-tree modification that predates this run
(last commit 2026-09-21 00:32); it was read-only for this run and was not written by this annotator.
