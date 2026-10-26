# RiskByte statistical intelligence — Synthetic-data findings

All measurements below describe the IEEE-CIS-inspired synthetic financial transaction dataset. They are not IEEE-CIS observations, banking-population estimates or model performance.

Dataset fingerprint: `sha256:737d55a6ed587403b2914203505a290bab6cffe16c1ea71a16b045abe34a2d66`.
Feature version: `synthetic-features-v1:entity-v1:strict-prior-v1:encoding-v1`.
Frozen temporal split: `sha256:30e32bb1ed91c4f11699d2ee4d9a5d37cfe29f5f6197de6dd844d87e5387efd4`.
Statistical artifact version: `statistical-intelligence-v1`.
Generated UTC: `2026-10-13T16:46:19.396649+00:00`.

## Execution and statistical design

Read-only source verification hashes both CSVs plus generator configuration/version and the independently hashed evaluation sidecar. Frozen feature/split IDs, split hash, disjoint/exhaustive indexes, chronological boundaries, row-order hash and exact target/split counts are verified before analysis. A mismatch aborts execution without publishing new results. No models, imputers, encoders, SHAP or ABRF are fitted.

Primary six hypotheses use only the frozen training partition. Descriptives, scenario/hard-negative exploration and elapsed-time summaries cover all records. Correlations use training observations. Holdout aggregates are exploratory and cannot be used for selecting later models or thresholds.

Sample variance/std use ddof=1; history features retain Phase 2 population std semantics. Percentiles use linear interpolation. Skewness and Fisher excess kurtosis are bias adjusted where defined; constant/small/all-missing groups return null for undefined moments. Missing/invalid/infinite values are excluded with counts, never imputed for tests. Zero means/std in history remain unavailable. Group quartiles support box plots; no individual raw records or huge matrices enter APIs.

Two-sided Mann-Whitney U handles unequal skewed/discrete groups; asymptotic tests correct ties and continuity. Small untied samples use exact tests, small tied samples seeded permutations. Group shapes are reported instead of claiming normality or equal variances. This is not a median-only test when shapes differ. Practical equivalence in the requested RQ1 wording is not testable without a prespecified equivalence margin; the operational H0 is equality of distributions.

Binary novelty tests exclude unavailable context, use Pearson chi-square with expected-count checks and Fisher exact for sparse 2x2 tables. Cliff's delta is P(fraud > reference)-P(fraud < reference), with ties zero. Cramer's V is unsigned; novelty odds ratios compare fraud odds for new versus seen context. Zero-denominator odds are null with reason. No arbitrary pseudocount is added.

Alpha=0.05. Benjamini-Hochberg FDR is applied once across all six prespecified questions (unavailable questions remain in the family with p=1 for adjustment). Report raw/adjusted p-values and effects. Repeated entities/episodes violate IID assumptions: all decisions below are nominal exploratory decisions. FDR cannot repair invalid independence assumptions; significance is not business utility.
A computed p-value of 0 denotes floating-point underflow, not a mathematical zero. The machine-readable p_value_underflow flag records this explicitly.

## Research questions and results

| Question | Feature                    | Test                                                                | Raw p       | BH adjusted p | Effect                 | Nominal decision    |
| -------- | -------------------------- | ------------------------------------------------------------------- | ----------- | ------------- | ---------------------- | ------------------- |
| RQ1      | TransactionAmt             | Mann-Whitney U; asymptotic; tie/continuity correction if asymptotic | 1.19662e-36 | 1.43595e-36   | Cliffs_delta 0.189503  | REJECT_H0_NOMINALLY |
| RQ2      | failed_attempts_before     | Mann-Whitney U; asymptotic; tie/continuity correction if asymptotic | 7.0123e-303 | 1.40246e-302  | Cliffs_delta 0.381059  | REJECT_H0_NOMINALLY |
| RQ3      | transactions_prior_30m     | Mann-Whitney U; asymptotic; tie/continuity correction if asymptotic | 0           | 0             | Cliffs_delta 0.636058  | REJECT_H0_NOMINALLY |
| RQ4      | device_seen_before         | Chi-square independence; Pearson, no continuity correction          | 0           | 0             | Cramers_V 0.243777     | REJECT_H0_NOMINALLY |
| RQ5      | merchant_seen_before       | Chi-square independence; Pearson, no continuity correction          | 2.26399e-42 | 3.39599e-42   | Cramers_V 0.0523133    | REJECT_H0_NOMINALLY |
| RQ6      | amount_ratio_to_prior_mean | Mann-Whitney U; asymptotic; tie/continuity correction if asymptotic | 0.000118557 | 0.000118557   | Cliffs_delta 0.0681154 | REJECT_H0_NOMINALLY |

### RQ1: Do transaction amount distributions differ by fraud status?

H0: The two feature distributions are equal; practical equivalence is not tested.

H1: The two feature distributions differ.

Selection: Skewed, discrete and unequal groups: two-sided rank test with ties; no normal/equal variance claim.

Synthetic-data finding: nominal evidence of association/distribution difference after FDR. Review effect magnitude, unequal shapes, missing-context selection and clustered observations; not a risk probability or causal claim.

Statistic: 6.1788e+07; group counts: `{'first': 1517, 'second': 68483}`; excluded: `{'first': 0, 'second': 0}`.

Assumption checks: Independent observations are not established: repeated entities and attack episodes cluster rows. P-values are nominal exploratory evidence, not calibrated population inference. Ordinal/numeric ranks; no equal-variance or normality requirement ties_present=True; finite-only group sizes=1517,68483 Different distribution shapes prevent interpreting U solely as a median difference

first: mean 15050.6, median 3711.37, variance 1.22421e+09, std 34988.8. Strongly skewed; mean and median describe different aspects.

second: mean 4554.77, median 2008.81, variance 8.74695e+07, std 9352.51. Strongly skewed; mean and median describe different aspects.

### RQ2: Are failed attempts associated with fraud status?

H0: The two feature distributions are equal; practical equivalence is not tested.

H1: The two feature distributions differ.

Selection: Skewed, discrete and unequal groups: two-sided rank test with ties; no normal/equal variance claim.

Synthetic-data finding: nominal evidence of association/distribution difference after FDR. Review effect magnitude, unequal shapes, missing-context selection and clustered observations; not a risk probability or causal claim.

Statistic: 7.17382e+07; group counts: `{'first': 1517, 'second': 68483}`; excluded: `{'first': 0, 'second': 0}`.

Assumption checks: Independent observations are not established: repeated entities and attack episodes cluster rows. P-values are nominal exploratory evidence, not calibrated population inference. Ordinal/numeric ranks; no equal-variance or normality requirement ties_present=True; finite-only group sizes=1517,68483 Different distribution shapes prevent interpreting U solely as a median difference

first: mean 3.00791, median 0, variance 16.0646, std 4.00806. Strongly skewed; mean and median describe different aspects.

second: mean 0.214433, median 0, variance 0.260653, std 0.510542. Strongly skewed; mean and median describe different aspects.

### RQ3: Does prior 30-minute transaction velocity differ by fraud status?

H0: The two feature distributions are equal; practical equivalence is not tested.

H1: The two feature distributions differ.

Selection: Skewed, discrete and unequal groups: two-sided rank test with ties; no normal/equal variance claim.

Synthetic-data finding: nominal evidence of association/distribution difference after FDR. Review effect magnitude, unequal shapes, missing-context selection and clustered observations; not a risk probability or causal claim.

Statistic: 8.4984e+07; group counts: `{'first': 1517, 'second': 68483}`; excluded: `{'first': 0, 'second': 0}`.

Assumption checks: Independent observations are not established: repeated entities and attack episodes cluster rows. P-values are nominal exploratory evidence, not calibrated population inference. Ordinal/numeric ranks; no equal-variance or normality requirement ties_present=True; finite-only group sizes=1517,68483 Different distribution shapes prevent interpreting U solely as a median difference

first: mean 2.06131, median 1, variance 4.42302, std 2.1031. Inspect quartiles and tails; shape alone does not establish normality.

second: mean 0.109326, median 0, variance 0.390387, std 0.624809. Strongly skewed; mean and median describe different aspects.

### RQ4: Is new-device behaviour associated with fraud?

H0: Novelty and fraud status are independent.

H1: Novelty and fraud status are associated.

Selection: Binary context with observed-cell/expected-cell checks; Fisher exact for sparse 2x2 tables.

Synthetic-data finding: nominal evidence of association/distribution difference after FDR. Review effect magnitude, unequal shapes, missing-context selection and clustered observations; not a risk probability or causal claim.

Statistic: 2790.04; group counts: `{'fraud': 1042, 'legitimate': 45907}`; excluded: `{'fraud': 475, 'legitimate': 22576}`.

Assumption checks: Independent observations are not established: repeated entities and attack episodes cluster rows. P-values are nominal exploratory evidence, not calibrated population inference. minimum expected cell count=58.5929

Odds ratio: 14.9752. Table: `[[447, 595], [2193, 43714]]`; rows fraud,legitimate; columns new,seen; unavailable context excluded.

### RQ5: Is new-merchant behaviour associated with fraud?

H0: Novelty and fraud status are independent.

H1: Novelty and fraud status are associated.

Selection: Binary context with observed-cell/expected-cell checks; Fisher exact for sparse 2x2 tables.

Synthetic-data finding: nominal evidence of association/distribution difference after FDR. Review effect magnitude, unequal shapes, missing-context selection and clustered observations; not a risk probability or causal claim.

Statistic: 186.094; group counts: `{'fraud': 1517, 'legitimate': 66483}`; excluded: `{'fraud': 0, 'legitimate': 2000}`.

Assumption checks: Independent observations are not established: repeated entities and attack episodes cluster rows. P-values are nominal exploratory evidence, not calibrated population inference. minimum expected cell count=318.927

Odds ratio: 2.07489. Table: `[[533, 984], [13763, 52720]]`; rows fraud,legitimate; columns new,seen; unavailable context excluded.

### RQ6: Do prior-mean amount ratios differ between fraud and hard-negative legitimate transactions?

H0: The two feature distributions are equal; practical equivalence is not tested.

H1: The two feature distributions differ.

Selection: Skewed, discrete and unequal groups: two-sided rank test with ties; no normal/equal variance claim.

Synthetic-data finding: nominal evidence of association/distribution difference after FDR. Review effect magnitude, unequal shapes, missing-context selection and clustered observations; not a risk probability or causal claim.

Statistic: 2.89229e+06; group counts: `{'first': 1517, 'second': 3570}`; excluded: `{'first': 0, 'second': 94}`.

Assumption checks: Independent observations are not established: repeated entities and attack episodes cluster rows. P-values are nominal exploratory evidence, not calibrated population inference. Ordinal/numeric ranks; no equal-variance or normality requirement ties_present=False; finite-only group sizes=1517,3570 Different distribution shapes prevent interpreting U solely as a median difference

first: mean 2.7281, median 1.35938, variance 9.99426, std 3.16137. Strongly skewed; mean and median describe different aspects.

second: mean 2.14292, median 0.985195, variance 8.04842, std 2.83697. Strongly skewed; mean and median describe different aspects.

## Descriptive group comparison — Synthetic-data findings

Full-group statistics include all requested raw and history variables, quartiles, percentiles, variance, standard deviation, skewness and kurtosis in descriptive_statistics.json.

| Group                    | Rows   | Amount mean | Amount median | Amount variance | Amount std |
| ------------------------ | ------ | ----------- | ------------- | --------------- | ---------- |
| all                      | 100000 | 4795.03     | 2021.53       | 1.12134e+08     | 10589.3    |
| legitimate               | 98000  | 4589.08     | 2006.85       | 8.96499e+07     | 9468.36    |
| fraudulent               | 2000   | 14886.2     | 3898.12       | 1.11049e+09     | 33324      |
| normal_legitimate        | 93000  | 4299.81     | 1975.7        | 5.50531e+07     | 7419.78    |
| hard_negative_legitimate | 5000   | 9969.52     | 2762.09       | 7.02779e+08     | 26510      |

## Fraud prevalence — Synthetic-data findings

| Partition  | Rows   | Fraud | Prevalence | Nominal 95% Wilson CI |
| ---------- | ------ | ----- | ---------- | --------------------- |
| overall    | 100000 | 2000  | 2.0000%    | 1.9151%–2.0886%       |
| train      | 70000  | 1517  | 2.1671%    | 2.0619%–2.2777%       |
| validation | 15000  | 266   | 1.7733%    | 1.5741%–1.9972%       |
| test       | 15000  | 217   | 1.4467%    | 1.2676%–1.6506%       |

Synthetic-data finding: nominal 95% Wilson intervals describe a hypothetical repeated independent sample under this simulation, not banking populations. Generator fixes overall class counts; the realized full-sample proportion is exact. Entity/episode dependence invalidates an IID uncertainty claim; intervals are teaching approximations, not generator-parameter uncertainty.

## Hard negatives — Synthetic-data findings

| Signal                          | Normal legitimate median | Hard-negative median | Fraud median | Fraud vs hard-negative delta |
| ------------------------------- | ------------------------ | -------------------- | ------------ | ---------------------------- |
| TransactionAmt                  | 1975.7                   | 2762.09              | 3898.12      | 0.074775                     |
| failed_attempts_before          | 0                        | 0                    | 0            | 0.303656                     |
| account_age_days                | 1602                     | 1564                 | 1603.5       | -0.0098053                   |
| identity_age_days               | 1555.5                   | 1531.5               | 1547.5       | -0.0237139                   |
| amount_minus_prior_mean         | -212.595                 | -27.2528             | 666.999      | 0.135644                     |
| amount_ratio_to_prior_mean      | 0.845079                 | 0.976986             | 1.40731      | 0.0713102                    |
| amount_zscore_prior             | -0.254991                | -0.0538089           | 0.596608     | 0.0796559                    |
| transactions_prior_5m           | 0                        | 0                    | 1            | 0.10398                      |
| transactions_prior_30m          | 0                        | 0                    | 1            | 0.150153                     |
| transactions_prior_1h           | 0                        | 0                    | 2            | 0.228325                     |
| transactions_prior_24h          | 1                        | 1                    | 3            | 0.247678                     |
| time_since_previous_transaction | 70479                    | 3583.5               | 30           | -0.190234                    |

| Group                    | Device novelty fraction (available rows) | Merchant novelty | Region novelty |
| ------------------------ | ---------------------------------------- | ---------------- | -------------- |
| normal_legitimate        | 0.0295341                                | 0.159055         | 0.00014271     |
| hard_negative_legitimate | 0.135047                                 | 0.439258         | 0.0990624      |
| fraudulent               | 0.414777                                 | 0.324            | 0.2755         |

Synthetic-data finding: aggregate hard negatives overlap fraud in amount deviation. The training RQ6 delta is negligible despite a small nominal p-value. Device/region novelty and failed-attempt distributions show group differences, but no signal alone establishes fraud or classifier separability.

| Hard-negative subtype | Amount median | Prior-mean amount ratio median | 30-minute velocity median |
| --------------------- | ------------- | ------------------------------ | ------------------------- |
| LAPTOP_PURCHASE       | 15349.6       | 6.883                          | 0                         |
| MEDICAL_EXPENSE       | 14182.4       | 6.83991                        | 0                         |
| NEW_LEGITIMATE_DEVICE | 1905.21       | 0.799004                       | 0                         |
| NEW_MERCHANT_PURCHASE | 1895.11       | 0.826704                       | 0                         |
| SHOPPING_FESTIVAL     | 1939.96       | 0.848143                       | 3                         |
| TRAVEL_PURCHASE       | 2182.73       | 0.817571                       | 0                         |

Synthetic-data finding: laptop and medical hard negatives have high amount ratios; shopping-festival hard negatives have high velocity. Legitimate new-device and travel events share novelty signals with attacks. These subtype summaries demonstrate why future evaluation must measure hard-negative false positives.

Subtype-level amount/velocity/deviation/novelty summaries in hard_negative_analysis.json identify which unusual legitimate mechanisms resemble fraud. Similar medians or small deltas indicate overlap, not classifier separability; distribution tails and missing evidence matter.

## Scenarios — Synthetic-data findings

| Scenario                      | Count | Fraud share | Amount median | 30-minute velocity median |
| ----------------------------- | ----- | ----------- | ------------- | ------------------------- |
| ACCOUNT_TAKEOVER              | 401   | 0.2005      | 11304.4       | 3                         |
| AMOUNT_SPIKE                  | 80    | 0.0400      | 14406.3       | 0                         |
| INTERNATIONAL_ANOMALY         | 79    | 0.0395      | 3511.47       | 0                         |
| LOW_AND_SLOW                  | 367   | 0.1835      | 545.38        | 0                         |
| MIXED_MODE                    | 371   | 0.1855      | 8312.86       | 3                         |
| NEW_DEVICE_HIGH_AMOUNT        | 80    | 0.0400      | 16309.2       | 0                         |
| NEW_MERCHANT_AMOUNT_DEVIATION | 80    | 0.0400      | 15455.7       | 0                         |
| REGION_CHANGE                 | 80    | 0.0400      | 1715.95       | 0                         |
| REPEATED_FAILED_ATTEMPTS      | 80    | 0.0400      | 1769.06       | 0                         |
| VELOCITY_BURST                | 382   | 0.1910      | 1696.29       | 3                         |

Scenario labels describe synthetic generating mechanisms only; they never enter feature inputs. Episode sizes explain unequal row shares. Novelty summaries retain available/unavailable denominators.

## Correlations — Synthetic-data findings

Pearson and Spearman are reported for 66 fixed meaningful training pairs with finite paired counts. Spearman is preferred for skewed monotonic behaviour; neither implies causation. No identifiers, target or evaluation metadata enter the matrix. Strongest absolute Spearman pairs:

- account_age_days / identity_age_days: Spearman 0.999824, Pearson 0.999851, n=48948.
- amount_ratio_to_prior_mean / amount_zscore_prior: Spearman 0.918954, Pearson 0.370627, n=66001.
- amount_minus_prior_mean / amount_ratio_to_prior_mean: Spearman 0.891414, Pearson 0.520217, n=68000.
- transactions_prior_30m / transactions_prior_1h: Spearman 0.861727, Pearson 0.973875, n=70000.
- transactions_prior_24h / time_since_previous_transaction: Spearman -0.857917, Pearson -0.44964, n=68000.

## Temporal exploration — Synthetic-data findings

Elapsed-day volume, prevalence, scenario counts and partition feature summaries are saved in temporal_analysis.json. Only relative synthetic time is used; no dates/weekdays/months.

Supplementary partition prevalence association: Chi-square independence; Pearson, no continuity correction, statistic 37.3414, raw p=7.78805e-09; Cramer's V=0.0193239. This exploratory diagnostic is outside the six-question FDR family and is not a confirmatory claim.

## Reproduction, API and artifacts

```powershell
.\.venv\Scripts\python.exe -m riskbyte_ml.statistics --config ml/config/synthetic_riskbyte.example.json
```

Default artifact root is ml/artifacts; output is ml/artifacts/statistics. Individual JSON files carry identical provenance; analysis_bundle.json is published last atomically as the coherent API snapshot. One writer per output directory. API rechecks exact source hashes and frozen provenance; no analysis runs in a request, no local paths or raw rows are returned. Missing configuration is dataset_not_configured; a verified source without a bundle is statistics_not_generated.

GET /api/v1/statistics/summary, /hypotheses, /correlations, /prevalence, /scenarios, /hard-negatives and /temporal provide typed finite responses. Quartiles, effect/statistic/p-value fields, pairwise coefficients, group/scenario summaries and elapsed-day buckets support later visualizations. Undefined statistics remain null with counts/reasons; no frontend redesign.

## Limitations

- All results are Synthetic-data findings; cannot demonstrate production banking performance.
- Generator rules influence observed associations; deterministic scenarios can make later performance optimistic.
- Repeated fictional entities and attack episodes violate independent-row inference; p-values/FDR and Wilson intervals are nominal teaching approximations.
- BH assumes independent or suitably positively dependent valid p-values; neither condition is established for this clustered simulation. Correction does not repair dependence.
- No cluster-robust p-values or entity/block bootstrap are implemented; use those before confirmatory population claims.
- Finite-only complete-case analyses report exclusions; device missingness/cold start changes denominators and can bias comparisons.
- Mann-Whitney tests rank/distribution differences, not practical equivalence or necessarily a difference in medians.
- No meaningful effect threshold was prespecified; conventional magnitude labels are heuristics, not operational utility.
- Overall fraud count was fixed by the generator, not sampled IID; full-dataset prevalence is exact for these records.
- Aggregated validation/test data are inspected for authorized research, so they are no longer unseen for exploratory purposes. Do not tune later model/threshold choices to these summaries.
- Synthetic entities are not bank customers; concept drift is approximated. Future ABRF results can describe this simulation only.
- Fraud-scenario and hard-negative fields remain evaluation-only; no labels enter prior-history calculations or predictive features.

## References

Test implementations follow [SciPy Mann-Whitney U](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.mannwhitneyu.html), [chi-square](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.contingency.chi2_contingency.html) and [Fisher exact](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html).

## Verification performed

Full statistical generation completed on the frozen MEDIUM dataset. **275 offline tests
passed** (220 existing and 55 new). New coverage checks descriptive moments/percentiles,
missing/zero-variance/extreme values, exact/permutation/asymptotic rank tests, categorical
test selection, effect sizes, FDR, Wilson intervals, group/scenario summaries, strict
serialization, source/split/version tampering, API readiness/privacy and missing-artifact
states. Test fixtures are small and synthetic; no test loads MEDIUM or requires the internet.

All frontend lint, typecheck, build and format checks passed. Python Ruff lint/format,
compilation and dependency checks passed. Import verification loaded **76 modules**.
Actual MEDIUM smoke requests returned HTTP 200 for health and all seven statistical
routes, with the verified fingerprint, finite JSON, no raw records or local source paths.
The frozen CSVs, feature manifest and temporal split were read without modification.

```powershell
npm run lint
npm run typecheck
npm run build
npm run format:check
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m compileall -q apps/api/src ml/src tests
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m riskbyte_ml.statistics --config ml/config/synthetic_riskbyte.example.json
```

## Exact next recommended phase

Review these frozen exploratory findings, then separately authorize Phase 4 supervised baseline classification using the existing temporal split and train-only preprocessing. Report scenario recall and hard-negative false-positive rates; freeze model decisions without tuning against inspected test summaries. No Phase 4 work is performed here.
