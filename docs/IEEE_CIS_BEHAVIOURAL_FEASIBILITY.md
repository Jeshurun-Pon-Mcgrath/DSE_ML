# IEEE-CIS behavioural feasibility

## Evidence state

Real files are absent; no actual repeated behavioural history or entity key is established.
The following table describes **conditional schema feasibility**, not measured success.
The runtime report under `ml/artifacts/ieee_cis/IEEE_CIS_BEHAVIOURAL_FEASIBILITY.md` verifies
present/non-missing prerequisites and useful proxy history. It returns UNSUPPORTED if no
qualifying group or valid temporal observations exist. No derived feature is materialized.

| Feature                               | Conditional support | Requirements / limitation                                                                |
| ------------------------------------- | ------------------- | ---------------------------------------------------------------------------------------- |
| Amount deviation                      | PARTIALLY_SUPPORTED | Amount plus strictly prior history in a reviewed proxy; cold start unavailable           |
| Transaction velocity                  | PARTIALLY_SUPPORTED | Counts in elapsed windows under proxy, not verified customer activity                    |
| Time since previous transaction       | PARTIALLY_SUPPORTED | Difference to strictly earlier elapsed event; simultaneous events do not establish order |
| Merchant novelty                      | UNSUPPORTED         | No documented merchant ID; ProductCD/domain is not a merchant identity                   |
| Device novelty                        | PARTIALLY_SUPPORTED | DeviceInfo string change in a non-device-conditioned proxy; sparse identity coverage     |
| Location/address novelty              | PARTIALLY_SUPPORTED | Prior anonymized addr-code tuple; not actual geographic location                         |
| Repeated behavioural history          | PARTIALLY_SUPPORTED | Complete proxy keys with repeated, temporally distinguishable observations               |
| Relative elapsed hour/day index       | SUPPORTED           | Finite non-negative elapsed seconds; relative index, no wall-clock claim                 |
| Calendar hour/weekday/month           | UNSUPPORTED         | Reference calendar instant is unknown                                                    |
| Geographic distance/impossible travel | UNSUPPORTED         | No documented event coordinates or sufficient dist-field meaning                         |
| Verified customer history             | UNSUPPORTED         | No clean documented customer_id                                                          |

Amount deviation, velocity, previous-event gap, device/address novelty and repeated history
require a constructed behavioural entity. All require reinterpretation as **transaction-profile
behaviour** rather than person behaviour. DeviceInfo is not a verified physical-device ID;
addr1/addr2 are not GPS; email-domain novelty cannot substitute for merchant novelty.
Missing fields/keys produce unavailable signals, never zero novelty or fabricated history.

Choose a grouping only after the entity study. A key containing DeviceInfo cannot measure
new devices relative to itself; use a reviewed card/profile key. A key containing addr1/addr2
similarly cannot independently establish changes of those fields. The proposed contract
must keep context-field novelty separate from fields that define the entity.

Future temporal features must read strictly prior events, use elapsed-time windows and
explicit simultaneous-event/cold-start policies. No inferred starting date or fake weekdays.
[IEEE-CIS official temporal description](https://www.kaggle.com/c/ieee-fraud-detection/data).
