# Success/failure validation diagnostic

These files evaluate one prediction per episode from the 0814 expanded dataset's
checkpoint validation split. The split contains 18 episodes: 13 successes and
5 failures. All three seeds use the same episode IDs. This is the validation
split used for checkpoint selection, not an untouched test set.

`h60_*` and `h120_*` use the trained risk head. The score is
`sigmoid(risk_logits)`, where failure is the positive class. The decision
threshold is 0.5. `baseline_*_value_*` uses the original WCM value head; its
failure score is `-V(s)`. For each baseline seed and evaluation point, the
threshold was chosen to maximize balanced accuracy on training episodes, then
applied unchanged to validation episodes. The two score types should not be
interpreted as identically calibrated probabilities.

## Fixed frame 2

This is the earliest common prediction frame for original WCM and SpaceTime.
Numbers are correct episodes out of 18; AUROC is for failure as positive.

| Model | Seed 3072 | Seed 42 | Seed 1337 |
| --- | --- | --- | --- |
| Original WCM value accuracy | 11/18 | 13/18 | 13/18 |
| Original WCM value AUROC | 0.600 | 0.646 | 0.785 |
| SpaceTime H60 risk accuracy | 18/18 | 18/18 | 18/18 |
| SpaceTime H120 risk accuracy | 18/18 | 18/18 | 18/18 |

Original WCM fails to identify 5/5 failures at frame 2 for seeds 3072 and 42;
seed 1337 identifies 1/5. Both SpaceTime variants identify 5/5 failures and
13/13 successes at this frame for each seed.

## Episode progress and final endpoint

The `_f0`, `_f05`, `_f075`, and default or `_f1` suffixes select the first,
50%, 75%, and last available prediction window within each episode. These are
fractions of the completed episode length, not numbers of input frames. At
`_f0`, SpaceTime's historical offsets clamp to the first observed frame, so
the eight history slots do not provide eight independent observations. A fixed
frame index is more appropriate for an operational early-prediction claim.

| Evaluation point | Original WCM value accuracy, seeds 3072 / 42 / 1337 | SpaceTime H60 and H120 risk accuracy, all seeds |
| --- | --- | --- |
| First available window | 11/18, 13/18, 13/18 | 18/18 |
| 50% of episode | 18/18, 18/18, 18/18 | 18/18 |
| 75% of episode | 18/18, 18/18, 18/18 | 18/18 |
| Last available window | 13/18, 13/18, 13/18 | 18/18 |

## Interpretation limit

The expanded dataset merges three sources. In this validation split, all five
failures come from the `0703_f` source; 11 of 13 successes come from the
`0814` tail source. A model can separate source appearance at frame 0 without
forecasting the task outcome. The current perfect SpaceTime risk accuracy is
therefore a source-confounded validation diagnostic, not evidence of general
early failure prediction. A decisive test needs held-out successes and failures
from the same collection source, with episode-disjoint calibration and testing.

The per-episode JSON files contain predictions, labels, frame indices,
thresholds, confusion matrices, macro-F1, balanced accuracy, and AUROC.
