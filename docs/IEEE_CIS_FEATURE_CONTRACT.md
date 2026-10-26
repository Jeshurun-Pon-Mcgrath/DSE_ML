# Proposed Phase 2 IEEE-CIS feature contract

This is a review proposal, **not fitted preprocessing or produced features**. Real files
are absent, so exact observed feature lists/availability are not claimed. After inspection,
the adapter writes `ml/artifacts/ieee_cis/feature_contract.json` with actual present columns,
support states and explicit source/processing constraints.

| Group                     | Proposed fields / rules                                                                                                                                          |
| ------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A. Raw predictive         | TransactionAmt; inspected suitable numeric measures such as C/D/dist/id families only after availability/meaning review                                          |
| B. Temporal derived       | Relative elapsed hour/day indexes; no calendar dates/weekdays/months. Raw TransactionDT reserved for chronology pending predictive review                        |
| C. Behavioural derived    | Past-only amount deviation, velocity, strictly previous-event gap, device-string/address-code novelty under independently chosen proxies; no fitted features yet |
| D. Missingness indicators | Present fields with actual missing values, including identity coverage patterns; candidates only, later train-only analysis                                      |
| E. Categorical            | Present ProductCD/card/addr/email/M/device/id_12–id_38 fields; numeric codes are categories, not scalar magnitude                                                |
| F. Anonymized engineered  | Present V1–V339; numerical usability may be evaluated, explanations must acknowledge unavailable business meaning                                                |
| G. Identifier/excluded    | TransactionID; raw TransactionDT; target/copies/post-outcome/leakage candidates pending review. Target kept separately for evaluation                            |

Runtime contract sections are separate, with exclusion candidates kept out of proposed
predictive lists; source frames still retain every column. Unknown fields are conservative
ANONYMIZED review entries, never automatically promoted to business meaning. Interpretability
tags may overlap with behavioural-key/exclusion tags. The target is `isFraud` with explicit
positive `1` and negative `0`, supported by the dataset-specific source definition; this
does not change generic positive-class configuration.

Candidate missingness indicators and categorical missing states require later validation.
Threshold-only dropping, global imputation, target encoding, resampling and general model
preprocessing are not authorized in this phase. Later fitting belongs inside the training
split, and chronological/grouped evaluation must be documented before experimenting.

History must use strictly prior elapsed events, no future/full-dataset aggregates or target
history, explicit cold-start/minimum-history rules and simultaneous-event handling. Novelty
must use an entity definition that does not include the field whose novelty is evaluated.
No real customer identity, merchant novelty or geographic travel feature can be asserted.

Future SHAP output for an anonymous column must say, for example, “anonymized engineered
feature V127 contributed to the model score,” not “V127 caused fraud.” Neither attribution
nor score proves causation/fraudulent intent. SHAP is not implemented in Phase 1.5.
