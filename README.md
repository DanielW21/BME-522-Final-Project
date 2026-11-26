# BME 522 Final Project

Frame-level classification of CalMS21 Task 1 mouse behavior labels: attack, investigation, mount, and other.

## Set up

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas scikit-learn matplotlib pyyaml ijson
python scripts/setup_data.py
```

## Configure and run

Edit `src/configs/config.yaml`.

- Set feature sets and models to `true` or `false` under `run` to control which set of model and feature set you want to use.
- The majority baseline uses no features and runs once
- Set model parameters under `model_params`.
- `spatial_motion_treba_pca` applies PCA to TREBA only
- Set `treba_feature_params.n_features` to choose the number of PCA features.
- `run.save_results` controls saving. Results go to `results/runs/<run.name>/`

Then run the experiment:

```sh
python -m src.run_experiment
```

## Models and next steps

- Majority and MLP are implemented.
- KNN, random forest, and linear SVM are also wired in `src/models/build.py` ready to be built and tuned with settings in `model_params`. Tune then run the selected models.
