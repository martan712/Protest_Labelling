# Unresolved review cases — run-2026-09-21-v3-flash

Recorded separately from the annotation CSVs. These are judgment calls that a
human reviewer should adjudicate; no taxonomy rules were changed to resolve them.

## Pilot (100 rows)
- ESP8238 public-employment exam cancellation: labour vs education vs other.
- NLD3884 local canal permit rule: other.
- SRB789 anti-government motive after attack on opposition leader: crime/victim
  justice vs democratic institutions.
- SRB1149 same movement, no issue stated: other.
- FRA43792 compulsory livestock culling: animal welfare primary, farmers secondary.
- NLD868 / FRA2826 pandemic-measures vs work consequences: pandemic/labour split.
- ITA2800 local inaction on crime: crime/victim justice vs other.

## Train batches (representative; IDs by subagent report)
- Public-employment/driver/transport rule burdens -> other: ITA25278, POL6067,
  ESP12153, ESP7698, MKD1471/1473/1475, SRB6777.
- Local infrastructure/siting objections -> other vs climate and environment:
  ITA25199, ITA13490, GBR7640, ESP10213, NOR1936, GBR3320/GBR8058, ITA8716,
  ESP16760, BEL1345, BEL627, NLD155.
- Far-right/NRM rallies with no covered subject -> other vs anti-fascism:
  SWE2544, SWE5741, DEU236, DEU4207, DEU24525, DNK93, AUT1773.
- Disaster-accountability after infrastructure failure: crime/victim justice vs
  democratic institutions: SRB4477, SRB4615, SRB3734, SRB4334, SRB3887, GRC5962,
  GRC5914, GRC6000, SRB6453, SRB3966.
- Health-worker working-conditions: labour vs healthcare: ESP20400, FRA30465,
  FRA36581, FRA10126, ITA5863, FRA18909, ESP3360, ITA25029.
- Police/prison-worker disputes: crime/victim justice vs labour: BEL1086,
  FRA3445, ESP11000, ITA3193, ESP8436, FRA8201, ESP23334.
- Anti-government / resignation-only demands -> other: BGR1724, SRB457, SRB812,
  ROU115, ROU972, SRB542, SRB406, SRB1526, ESP14703, ESP15259, ESP14861, ALB882.
- Disability / social-payment grievances with no matching class -> other:
  SRB1131, ITA25135, ITA25893, ROU260, BIH1077, ESP10170, ITA20493.
- Cross-national solidarity vs women rights: DNK494, DNK483.
- Bioethics / ART bills: lgbtq vs women rights: FRA6156.
- Health & safety deaths of students: labour vs education: ITA13694, ITA13641,
  ITA18038, ITA14433.
- Wind/hydro energy siting vs environment: DEU3205, FRA37477, HRV662, ALB1101.
- Anti-conscription / peace demos: war and foreign policy vs other: DEU27895,
  DEU27233, ITA28447, ITA22206.

## Dev
- BGR279 insult/disability, resignation demand alone: other vs democratic
  institutions.
- DEU21530 PEGIDA motives unspecified: other vs anti-fascism.
- MDA413/1980/2075 rallies backing officials on trial: unjust law enforcement vs
  democratic institutions.
- ESP18689 ETA amnesty: war and foreign policy vs unjust law enforcement.
- FRA44820 far-right tribute with victim-justice theme: crime/victim justice vs other.

## Test (locked)
- POL5950 truckers vs Ukrainian carriers: other vs labour.
- POL8018 Azov protest: ukraine-russia war vs anti-fascism / foreign policy.
- NOR138 Stop Islamisation rally: other.
- GRC5914 / GRC6000 disaster justice vs cover-up.
- ITA13427 / ITA7602 / ITA5345 school reopening: pandemic vs education.
- SRB2122 city-wide heat outage: housing and rents vs other.

## Provenance / integrity
- Concurrent uncoordinated writer in the same run directory (see `progress.md`).
  Per-row writer attribution within this run is not guaranteed.
