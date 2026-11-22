import os

files = {}

files["src/ml/train.py"] = '''import os
import joblib
import pandas as pd
import numpy as np
import optuna
import shap
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, accuracy_score

optuna.logging.set_verbosity(optuna.logging.WARNING)

def rps_score(y_true, y_prob):
    """
    Ranked Probability Score (RPS) for 3-class ordered outcomes (Home, Draw, Away)
    """
    n_samples, n_classes = y_prob.shape
    y_true_onehot = np.zeros((n_samples, n_classes))
    for i, val in enumerate(y_true):
        y_true_onehot[i, val] = 1.0

    cum_prob = np.cumsum(y_prob, axis=1)
    cum_true = np.cumsum(y_true_onehot, axis=1)

    rps = np.mean(np.sum((cum_prob[:, :-1] - cum_true[:, :-1]) ** 2, axis=1) / (n_classes - 1))
    return rps

def train_and_evaluate():
    features_path = "data/processed/epl_features.parquet"
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Missing {features_path}. Run features pipeline first.")

    df = pd.read_csv(features_path) if features_path.endswith('.csv') else pd.read_parquet(features_path)
    print(f"--> Loaded dataset with {len(df)} rows.")

    feature_cols = [
        'shin_prob_home', 'shin_prob_draw', 'shin_prob_away',
        'elo_home', 'elo_away', 'elo_diff',
        'home_xg_ewma_3', 'away_xg_ewma_3', 'home_xga_ewma_3', 'away_xga_ewma_3',
        'home_xg_ewma_5', 'away_xg_ewma_5', 'home_xga_ewma_5', 'away_xga_ewma_5',
        'home_xg_ewma_10', 'away_xg_ewma_10', 'home_xga_ewma_10', 'away_xga_ewma_10',
        'dc_prob_home', 'dc_prob_draw', 'dc_prob_away'
    ]

    # Time-based Train / Validation / Test split
    n = len(df)
    train_end = int(n * 0.75)
    val_end = int(n * 0.88)

    X_train, y_train = df.iloc[:train_end][feature_cols], df.iloc[:train_end]['target']
    X_val, y_val = df.iloc[train_end:val_end][feature_cols], df.iloc[train_end:val_end]['target']
    X_test, y_test = df.iloc[val_end:][feature_cols], df.iloc[val_end:]['target']

    print(f"Split sizes -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # 1. Optuna Hyperparameter Optimization for CatBoost (GPU)
    print("--> Running Optuna Optimization for CatBoost on GPU...")
    def objective(trial):
        params = {
            'iterations': trial.suggest_int('iterations', 150, 400),
            'depth': trial.suggest_int('depth', 4, 7),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
            'task_type': 'GPU',
            'verbose': 0,
            'random_seed': 42
        }
        model = CatBoostClassifier(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30)
        preds_val = model.predict_proba(X_val)
        return log_loss(y_val, preds_val)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=15)
    best_params = study.best_params
    best_params.update({'task_type': 'GPU', 'verbose': 0, 'random_seed': 42})
    print(f"Best CatBoost Params: {best_params}")

    # 2. Fit Final Ensemble Models
    print("--> Training CatBoost and LightGBM Models...")
    cb_model = CatBoostClassifier(**best_params)
    cb_model.fit(X_train, y_train)

    lgb_model = LGBMClassifier(n_estimators=250, max_depth=5, learning_rate=0.03, random_state=42, verbose=-1)
    lgb_model.fit(X_train, y_train)

    # 3. Probability Calibration
    print("--> Calibrating Probabilities (Isotonic Regression)...")
    calibrated_cb = CalibratedClassifierCV(cb_model, cv='prefit', method='isotonic')
    calibrated_cb.fit(X_val, y_val)

    # 4. Evaluation on Test Set
    raw_preds = (cb_model.predict_proba(X_test) + lgb_model.predict_proba(X_test)) / 2.0
    cal_preds = calibrated_cb.predict_proba(X_test)

    acc = accuracy_score(y_test, np.argmax(cal_preds, axis=1))
    loss = log_loss(y_test, cal_preds)
    rps = rps_score(y_test.values, cal_preds)

    print("\n================ FINAL TEST RESULTS ================")
    print(f"🎯 Accuracy: {acc * 100:.2f}%")
    print(f"📉 Multi-class Log Loss: {loss:.4f}")
    print(f"📊 Ranked Probability Score (RPS): {rps:.4f}")
    print("====================================================")

    # 5. Save Model Artifacts & SHAP Explainer
    os.makedirs("models", exist_ok=True)
    joblib.dump(calibrated_cb, "models/catboost_calibrated.joblib")
    joblib.dump(feature_cols, "models/feature_cols.joblib")

    print("--> Generating SHAP Explainer...")
    explainer = shap.TreeExplainer(cb_model)
    joblib.dump(explainer, "models/shap_explainer.joblib")
    print("✅ Model Training & SHAP Artifacts Saved to models/")

if __name__ == "__main__":
    train_and_evaluate()
'''

for filepath, content in files.items():
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("📁 ML modules written successfully.")
