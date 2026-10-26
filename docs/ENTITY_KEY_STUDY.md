# IEEE-CIS entity key study

## Evidence state

**Actual dataset study unavailable: both real local CSVs are absent.** No real entity count,
singleton rate, collision estimate or recommended grouping is claimed. The implementation
generates a measured report under `ml/artifacts/ieee_cis/ENTITY_KEY_STUDY.md` after a successful
real-data run. Synthetic fixture findings are test evidence only.

## Proposed candidates

1. `card1`–`card6` tuple: broad card-profile grouping.
2. `card1`–`card6`, `addr1`, `addr2`: card/address transaction profile.
3. `card1`, `card2`, `addr1`, `addr2`, `P_emaildomain`: lighter card/address/email grouping.
4. `card1`, `card2`, `addr1`, `addr2`, `DeviceInfo`: device-conditioned transaction profile.

Operator candidates may use distinct approved proxy fields; outcome, time and transaction
ID cannot form entity definitions. At most 16 candidates are evaluated. No field combination
is declared a real customer/person. Shared attributes collide; changing attributes fragment
history. Device-conditioned keys cannot independently evaluate novelty of that same device.

## Measurement rules

Use exact observed tuples, not delimiter-concatenated strings (avoids artificial key
collisions). Exclude rows with any missing component **from that candidate's study only**;
never delete those transactions or collapse all missing keys into a fictitious entity.
Report eligible/missing rows, missing percentage/per-field counts, unique groups, mean/
median/max transactions per group, singleton entity percentage and repeated-history row
percentage. Entity metrics use complete-key groups as the denominator; percentages distinguish
entity counts from row counts. Absent fields yield unavailable statistics, not zero entities.

Inspect whether groups contain multiple observed DeviceInfo/address/email values. These
variation percentages are diagnostic and may reflect legitimate changes; they do **not**
estimate true-identity collision rates. Collision risk remains unknown without ground truth.
Also report the percentage of complete-key rows having a strictly earlier elapsed observation;
simultaneous rows do not create prior history. This is feasibility analysis, not a produced
history feature or model input.

## Recommendation policy

Provisional screening thresholds: at least 20 complete groups, at least 20% non-singleton
groups and at least 50% transaction coverage. They are configurable engineering thresholds,
not statistical significance, identity validation or predictive usefulness. Rank passing
candidates by repeated-history row share multiplied by complete-key coverage; prefer fewer
fields to break ties. Publish all passing candidates for review, not a hidden automatic choice.
Temporal behavioural feasibility additionally requires strictly prior elapsed observations.

After actual profiling, review collisions, coverage, missingness, temporal persistence and
stability across chronological splits. Approve one or multiple **behavioural_entity** or
**transaction_profile** strategies only if measured useful. Consider card profiles for
device novelty, rather than a key already containing the device. No recommendation can be
made for this local snapshot until the real files are supplied.
