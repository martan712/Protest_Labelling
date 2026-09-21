# Pilot review — run-2026-09-21-flash-01

- Annotator string: `deepseek-v4.1-flash-run-2026-09-21-flash-01`
- Taxonomy: `2026-09-21-v3`
- Prompt snapshot SHA-256: `19c2da75017c047b26cd1d88d6b8b1aa49932646d8b1d927561674af18ea8250` (matches frozen)
- Pilot size: 100 rows from `train_6000` (75 reproducible random, seed 20260921; 25 ambiguous-boundary selections)
- Output: `pilot/train_pilot_100_output.csv`
- Schema validation: PASS (exact IDs/order, no dups/missing, all primary labels allowed, <=1 alternative per row and always != primary, `other_reason` rules hold, evidence and annotator nonempty on every row).

## `other` review (all 11 `other` rows checked against the two reason definitions and the group/institution fallback)

| event | other_reason | verdict |
| --- | --- | --- |
| ITA10110 | outside_taxonomy | ACCEPT — public-transport funding cut; no covered sector domain; unions present but no wage/working-condition demand stated (explicit demand overrides group default). |
| BIH190 | outside_taxonomy | ACCEPT — resident objection to hotel construction; not housing access/supply. |
| MKD1474 | outside_taxonomy | ACCEPT — transport operators vs EU entry/exit rules = regulatory/business burden (fisher analogy). |
| BGR4845 | outside_taxonomy | ACCEPT — pro-government counter-mobilization with no covered substantive issue. |
| DEU23613 | insufficient_information | ACCEPT (borderline) — wind-energy opponents criticizing "various political and media policies" with no stated demand; no domain identifiable from demand or group/institution. See unresolved. |
| DEU24651 | outside_taxonomy | ACCEPT — pedestrian-zone opposition over parking/inconvenience (explicitly listed out-of-scope). |
| DEU26485 | outside_taxonomy | ACCEPT — conscription/military-service law grievance; domestic defence law, not war/foreign policy or education (explicit grievance overrides student default). |
| ESP9121 | outside_taxonomy | ACCEPT — bank-branch closure; no covered domain (distinct from ER/theatre/school closures). |
| NLD4772 | outside_taxonomy | ACCEPT — boat-size/mooring restrictions = administrative rule/inconvenience. |
| POL3151 | outside_taxonomy | ACCEPT — municipal income/tax reform; not household cost-of-living. |
| SVN623 | outside_taxonomy | ACCEPT — disability support; no disability class exists. |

Review counts: 11/11 `other` rows reviewed; 11 accepted; 0 re-labelled.

## Unresolved / flagged cases (not changed; to be escalated rather than guessed)

1. DEU23613 — `other`/`insufficient_information` vs `climate and environment` (wind-energy subject is energy policy, but the note states only vague "political and media policies"). Borderline.
2. GBR9029 / GBR9729 / GBR7698 — far-right anti-immigration mobilizations labelled `immigration` (first-described mobilization) with `anti-fascism and extremism` / `racism` / `palestine-israel conflict` as secondary. The far-right affiliation rule and the "first-described mobilization" tie-break both apply; defensible but boundary-sensitive.
3. FRA28893 — journalist strike over a broadcast format: `labor rights and wages` (strike = covered labour action) vs `other` (editorial-format grievance). Borderline.
4. ITA10110 — `other` vs a possible `labor rights and wages` reading via transport unions; explicit demand (transit funding) chosen over group default.
5. ESP19417 / SVN623 / ITA25893 — no disability/welfare class exists; mapped to `healthcare` (home-care/state-funded care) or `other` on a best-fit basis.

## Distribution (primary labels, pilot, n=100)

labor rights and wages 23; other 11; climate and environment 10; women rights 8; immigration 7; healthcare 6;
farmers 5; war and foreign policy 4; crime, violence and victim justice 4; pandemic 4; palestine-israel conflict 3;
animal welfare 3; anti-fascism and extremism 2; democratic institutions, corruption, elections, and reforms 2;
civil liberties and censorship 2; racism 1; cost of living and inflation 1; education 1; lgbtq 1; housing and rents 1; ukraine-russia war 1.
