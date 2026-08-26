# German Credit Risk Prediction

An end-to-end machine learning pipeline that predicts whether a loan applicant is a good or bad credit risk, built on the [UCI Statlog German Credit Data](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data) (1,000 real loan applicants from a German bank).

Built as a learning project to practice real ML engineering practices — not just training a model, but building a reproducible, config-driven pipeline around it.

## What it does

- Loads and preprocesses the German Credit dataset (one-hot encoding, target binarization)
- Trains multiple candidate models (Logistic Regression, Random Forest) via a config-driven model factory
- Runs hyperparameter search with [Optuna](https://optuna.org/), using 5-fold cross-validation
- Optimizes for **F1 score** (not raw accuracy) to properly handle class imbalance in the dataset — a model that just predicts "good risk" most of the time would score well on accuracy while missing most actual bad-risk applicants
- Tracks every experiment run (parameters + metrics) with [MLflow](https://mlflow.org/)
- Saves the best-performing model to disk
- Provides a prediction script that loads the saved model and scores new applicants

## Project structure
german-credit-risk/
├── config/
│   └── config.yml          # data split, model search spaces, Optuna settings
├── src/
│   ├── data/
│   │   └── prep_data.py    # load + encode the dataset
│   ├── training/
│   │   ├── model_factory.py  # builds a model from a name + hyperparameters
│   │   └── train.py          # runs the full training + tuning + tracking pipeline
│   ├── inference/
│   │   └── predict.py        # loads saved model, scores new applicants
│   └── utils/
│       ├── config.py
│       └── paths.py
├── models/                  # saved model artifacts (gitignored)
├── notebooks/
│   └── 01_explore_data.ipynb
└── mlruns/                  # MLflow experiment logs (gitignored)

## Why F1 instead of accuracy

The dataset is imbalanced (~70% good risk, ~30% bad risk). Early experiments showed a model optimized for accuracy achieved ~0.78 accuracy but only caught 33-35% of actual bad-risk applicants (low recall on the minority class) — a real problem for a credit risk system, where missing a risky borrower is typically costlier than being overly cautious with a safe one. Switching the Optuna objective to F1 score (with `class_weight="balanced"`) raised bad-risk recall to ~0.73, at the cost of some overall accuracy — a deliberate, understood trade-off.

## How to run it

```bash
# Install dependencies
uv sync

# Train models (runs Optuna search, logs to MLflow, saves the best model)
uv run python -m src.training.train

# View experiment tracking dashboard
uv run mlflow ui --port 8080
# then open http://localhost:8080

# Score a new applicant using the saved model
uv run python -m src.inference.predict
```

## Tech stack

- **scikit-learn** — models and preprocessing
- **Optuna** — hyperparameter search
- **MLflow** — experiment tracking
- **uv** — dependency management
- **pandas** — data handling

## Status / Roadmap

- [x] Data pipeline
- [x] Config-driven multi-model training
- [x] Hyperparameter search
- [x] Experiment tracking
- [x] Model persistence + CLI prediction
- [ ] REST API for serving predictions
- [ ] Docker containerization
- [ ] Automated tests + CI

## Acknowledgments

Project structure and MLOps patterns (config-driven model factory, Optuna search, experiment tracking) were inspired by [Adeemy/end-to-end-ml](https://github.com/Adeemy/end-to-end-ml), adapted here for the German Credit Risk dataset and built from scratch as a personal learning project.