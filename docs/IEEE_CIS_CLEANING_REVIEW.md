# IEEE-CIS cleaning review

Real files are absent: actual extreme/moderate missingness lists are unavailable. After
profiling, `ieee_cis_integration.json` and the ignored cleaning-review artifact list extreme
missingness (at least 90%), moderate missingness (20% to below 90%) and candidate indicators
for every observed field with missing values. These are descriptive review thresholds, not
automatic drop rules.

Identity absence is structural for some transactions. Distinguish unmatched identity keys
from matched records with missing attributes. Preserve all transactions through the left
join. Missingness may convey collection/process information; review drift and whether it is
available at assessment time before using indicators in future experiments.

Recommended Phase 2 proposal: train-only numeric imputation appropriate to each family,
explicit missing states for categoricals, and evaluated missingness indicators. Leave the
choice to later data review; no imputation, encoding, normalization, record deletion or
column dropping runs here. Do not aggressively complete cases or delete high-missingness
columns solely on percentages. Invalid values, outliers and suspicious features stay in
the profile with explanations until a reviewed plan is authorized.

Categorical storage optimization is reversible representation, not cleaning: it preserves
every exact observed string. Numeric source strings retain precision; float statistics are
analysis-only. No joined/cleaned source CSV is written in Phase 1.5.
