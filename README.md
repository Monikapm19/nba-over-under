# Predicting OVER vs UNDER the Sportsbook Total Line (NBA)

UE24CS352A Machine Learning Mini-Project - Problem 15: *The Bank is Open: AI in Sports Gambling*

**Team:** Monika P M (PES1UG24CS275), More Atharv Sachin (PES1UG24CS276)
**Repository:** `nba-over-under` (private) - shared with faculty `<FACULTY-GITHUB-USERNAME-PLACEHOLDER>` and TAs `<TA-USERNAMES-PLACEHOLDER>`

## Task
A simplified, classification version of the paper *The Bank is Open: AI in Sports Gambling* (Bucquet & Sarukkai). For each NBA game, predict whether the final combined score is **OVER (1)** or **UNDER (0)** the **sportsbook total line**. We do *not* predict the raw game total. Pushes (actual total == line) are dropped.

`line_total` (the sportsbook total line) is used as an input feature. All team-performance features use only games played **before** the game being predicted.

## Dataset
Kaggle: *NBA Betting Data | October 2007 to June 2026* (user `cviaxmiwnptr`)
https://www.kaggle.com/datasets/cviaxmiwnptr/nba-betting-data-october-2007-to-june-2024
File: `data/nba_2008-2026.csv` (24,440 games, 2007-10-30 to 2026-06-13, 27 columns). Per the Kaggle page, 2007-Jan 2023 data comes from Sportsbook Reviews Online, later data from ESPN and Sportsbook Review. The file does not say whether `total` is an opening or closing line, so we call it simply the "sportsbook total line". The CSV is included for reproducibility; please respect the original dataset's terms.

## Setup and run
```bash
pip install -r requirements.txt
python src/train.py          # run from the project root
```
Outputs go to `results/`: `metrics.csv` (validation + test), `test_metrics_with_ci.csv`, `confusion_matrices.png`, `rf_feature_importance.csv`, `run_summary.json` (split sizes, chosen hyper-parameters, feature list). Random seed: 42.

## Project structure
```
data/nba_2008-2026.csv   raw dataset
src/arenas.py            approximate city coordinates for the travel feature
src/features.py          feature engineering + label + chronological split
src/train.py             preprocessing, tuning, evaluation, plots
results/                 generated outputs
report/                  2-page PDF write-up
slides/                  review deck (PPTX)
```

## Method
- **Split (chronological):** train 2008-2022 (18,849 games), validation 2023 (1,311), test 2024-2026 (3,952).
- **Features (38):** per team (home `h_`, away `a_`): last-3-game averages (points for/against, total points, margin, win rate), last-3 average of opponents' season-to-date points for/against, season-to-date points for/against/total, rest days, back-to-back flag, games in last 7 days, travel km. Game-level: `line_total`, absolute spread, home-favoured, playoff flag, expected totals (last-3 and season-to-date), expected total minus line, both-teams-back-to-back, travel difference.
- **Travel:** Haversine distance between the cities of a team's previous and current game, using approximate city coordinates in `src/arenas.py` (written from general geographic knowledge, not a cited dataset, ~1 km precision).
- **Preprocessing:** median imputation (fit on train only); standard scaling for Logistic Regression and the neural network.
- **Models:** Logistic Regression, Random Forest (baseline model), XGBoost, Neural Network (scikit-learn MLP). Hyper-parameters selected on validation ROC-AUC; test set used only for final evaluation.
- **Reference baselines:** majority class (Under) and a seeded coin flip.

## Results (test set, 2024-2026, n = 3,952; 95% CI on accuracy about +/-1.6 points)
| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 49.65% | 0.502 | 0.476 | 0.489 | 0.494 |
| Random Forest (baseline) | 50.76% | 0.527 | 0.255 | 0.344 | 0.518 |
| XGBoost | 50.00% | 0.507 | 0.429 | 0.464 | 0.509 |
| Neural Network (MLP) | 50.53% | 0.515 | 0.380 | 0.437 | 0.505 |
| Majority-class baseline | 49.44% | 0.000 | 0.000 | 0.000 | n/a |
| Coin-flip baseline | 49.67% | 0.502 | 0.497 | 0.500 | n/a |

Random Forest has the best observed accuracy (50.76%), but this is within the estimated uncertainty of the baselines. **None of the models demonstrates a meaningful predictive advantage**, and none reaches the ~52.4% break-even accuracy at -110 odds. We do not claim to beat the sportsbook.

## Limitations
Team-level features only (no injuries/players); small validation set; approximate travel coordinates; unknown opening/closing status of the line; one chronological split.

## References
- A. Bucquet and V. Sarukkai, *The Bank is Open: AI in Sports Gambling* (paper provided with the assignment).
- Kaggle dataset by `cviaxmiwnptr` (link above); Sportsbook Reviews Online NBA odds archive (original source named in the paper and dataset page).
