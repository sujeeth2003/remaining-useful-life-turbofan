# Remaining Useful Life Prediction (turbofan engines)

Predict how many operating cycles an engine has left before failure from its sensor history. Built for the NASA **C-MAPSS** run-to-failure benchmark; ships with a **synthetic fleet generator** with the same shape so everything runs without downloading anything.

> **Results below are on synthetic data.** My earlier project on the real C-MAPSS data (100+ engines) reached an RMSE of about 13 cycles; that dataset is NASA's and is not redistributed here, and this repo's numbers are **not** comparable to it. To reproduce on real data, download `train_FD001.txt` from the NASA Prognostics Data Repository and run `python run_rul.py --cmapss train_FD001.txt`.

## Method
- **Target:** RUL = cycles until the unit's last recorded cycle, **clipped at 125** (very early life carries no degradation signal; the standard C-MAPSS convention).
- **Features (causal, per engine):** raw sensors, rolling mean and std over 5 and 20 cycles, and the **least-squares slope over 20 cycles**. Rolling-window features carry most of the signal, because a single reading is dominated by unit-to-unit spread while the *trend* shows degradation. Constant sensors are dropped.
- **Models:** random forest and gradient boosting versus a predict-the-mean baseline. (An LSTM is the usual next step; PyTorch was not part of this environment, so it is not included.)
- **Evaluation done properly:** k-fold **grouped by engine**. A random row split leaks, because consecutive cycles of one engine are near duplicates, and looks far better than reality. Reported both over all cycles and over each engine's **last observed cycle**, which is the moment a maintenance decision is made, plus the asymmetric **NASA score** (late predictions are punished harder than early ones).

