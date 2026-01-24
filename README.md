# Remaining Useful Life Prediction (turbofan engines)

Predict how many operating cycles an engine has left before failure from its sensor history. Built for the NASA **C-MAPSS** run-to-failure benchmark; ships with a **synthetic fleet generator** with the same shape so everything runs without downloading anything.

> **Results below are on synthetic data.** My earlier project on the real C-MAPSS data (100+ engines) reached an RMSE of about 13 cycles; that dataset is NASA's and is not redistributed here, and this repo's numbers are **not** comparable to it. To reproduce on real data, download `train_FD001.txt` from the NASA Prognostics Data Repository and run `python run_rul.py --cmapss train_FD001.txt`.

## Method
- **Target:** RUL = cycles until the unit's last recorded cycle, **clipped at 125** (very early life carries no degradation signal; the standard C-MAPSS convention).
- **Features (causal, per engine):** raw sensors, rolling mean and std over 5 and 20 cycles, and the **least-squares slope over 20 cycles**. Rolling-window features carry most of the signal, because a single reading is dominated by unit-to-unit spread while the *trend* shows degradation. Constant sensors are dropped.
- **Models:** random forest and gradient boosting versus a predict-the-mean baseline. (An LSTM is the usual next step; PyTorch was not part of this environment, so it is not included.)
- **Evaluation done properly:** k-fold **grouped by engine**. A random row split leaks, because consecutive cycles of one engine are near duplicates, and looks far better than reality. Reported both over all cycles and over each engine's **last observed cycle**, which is the moment a maintenance decision is made, plus the asymmetric **NASA score** (late predictions are punished harder than early ones).

## Results (synthetic fleet: 60 engines, 3-fold, mean +/- std across folds)
```
model                           RMSE (all cycles)  RMSE (last cycle)  NASA score (last)
baseline: predict train mean          40.8 +/- 0.0        92.5 +/- 0.0             208655
random forest                         16.2 +/- 1.2         8.4 +/- 1.5                 27
gradient boosting                     15.9 +/- 1.2         7.9 +/- 2.5                 25
```
Reading it: the error over all cycles (~16) is larger than at the last cycle (~8) because far from failure the signal is weak and the model can only say "still healthy"; near failure the degradation is obvious. The synthetic degradation is smooth and low-noise, so **real engines will be harder**; treat these as a check that the pipeline and evaluation are sound, not as a benchmark.

## Run
```bash
pip install numpy pandas scikit-learn
python -m unittest discover -s tests          # 4 tests: RUL definition, feature causality, NASA score asymmetry, RMSE
python run_rul.py                             # synthetic fleet (about 5 minutes on one core)
python run_rul.py --cmapss train_FD001.txt    # real data
```
`tests/` includes a check that features are **causal**: changing an engine's future readings does not change earlier feature rows, so the same code is valid for online prediction.
