# Pilot review notes — run-2026-09-21-pro-01

100-row pilot (80 random seed=42 + 20 boundary cases by note-keyword selection).
Validated: exact IDs, no duplicates, all labels in taxonomy, other_reason only for
other, evidence/annotator nonempty. No label/rule modifications were made.

## Unresolved interpretation issues (recorded, not resolved by new rules)

- `POL8539` flood-barrier installation: treated as `other` (specific local public
  works). Borderline `climate and environment` (environmental health); note has no
  climate/environment framing.
- `ROU775` dual grievance (government attacks on judiciary + police brutality):
  `democratic institutions` primary (first/stronger), `unjust law enforcement`
  secondary. The "police violence -> unjust law enforcement" rule competes here.
- `ESP16534` rural-path heritage: `culture` (heritage). Borderline environment.
- `ITA20467` basic-income welfare criteria: `other` (no welfare/benefits domain).
- `NLD5075` anti-asylum mobilization ("AZC away"): `immigration` (asylum subject),
  despite crime/insecurity framing by protesters.
- `FRA7484` "more democratic system and social equality": `democratic institutions`;
  demand is somewhat vague/broad.
- `ESP13005` firefighters vs dismantling of services: `other` (no public-safety
  domain). `BEL5271` public-transport budget cuts likewise `other`.
- `GBR8717` Wimbledon expansion: `other` (no stated grievance domain in note).
- `ITA6206` bar/restaurant owners COVID relief: `pandemic` (restrictions are the
  grievance) though the demand is monetary relief. `DEU5037` shop owners pandemic
  hardship likewise `pandemic`.
- `ITA18351` prison inmates vs lack of hot water: `other` (facility condition, not
  misuse of coercive authority). Borderline `unjust law enforcement` (detention).
- `SRB6266` pro-government counter-protest: `other` (general support/opposition).
- `FRA9714` far-right Action Francaise anti-Islam action: `other` (no covered
  opposition subject; not anti-fascism since it is the far-right acting).
- `ITA21258` tax crimes/false invoices in asylum project: `democratic institutions`
  (corruption). Borderline `other` (private-sector fraud vs governance corruption).
- `MDA2226` fuel prices + administrative-territorial reform: `cost of living`
  primary, `democratic institutions` (governance reform) secondary.
- `ESP6925` Catalan independence referendum commemoration: `other` (no separatism /
  self-determination domain).
- `FRA16112` fishers vs gas price: `other` (fishers boundary rule; not farmers,
  not general-public cost-of-living).
- `GRC5882` rail-disaster government cover-up: `democratic institutions` primary,
  `crime, violence and victim justice` (justice for victims) secondary.
- `ITA26050` neo-fascism + homophobic crimes: `anti-fascism and extremism` primary,
  `lgbtq` secondary.

## Pilot summary
- 100 rows validated; label distribution recorded in merge script output.
- 11 `other` (all `outside_taxonomy`), 12 rows with one alternative label.
- No taxonomy or rule changes. Proceeding with the frozen prompt for the full run.
