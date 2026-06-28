import os
import json
import logging
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except ImportError:
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except ImportError:
    CatBoostClassifier = None

log = logging.getLogger(__name__)

def benchmark_models(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_split=10, class_weight="balanced", random_state=42),
        "Extra Trees": ExtraTreesClassifier(n_estimators=300, class_weight="balanced", random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    
    if XGBClassifier: 
        models["XGBoost"] = XGBClassifier(eval_metric="logloss", random_state=42)
    if LGBMClassifier: 
        models["LightGBM"] = LGBMClassifier(random_state=42, verbose=-1)
    if CatBoostClassifier: 
        models["CatBoost"] = CatBoostClassifier(verbose=False, random_state=42)
        
    scoring = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "roc_auc": "roc_auc"
    }
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    best_model = None
    best_name = None
    best_f1 = -1
    
    log.info("Starting model benchmarking across %d algorithms...", len(models))
    
    for name, model in models.items():
        log.info(" Evaluating %s...", name)
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        result = {
            "Model": name,
            "Accuracy": scores["test_accuracy"].mean(),
            "Precision": scores["test_precision"].mean(),
            "Recall": scores["test_recall"].mean(),
            "F1": scores["test_f1"].mean(),
            "ROC_AUC": scores["test_roc_auc"].mean()
        }
        rows.append(result)
        
        if result["F1"] > best_f1: 
            best_f1 = result["F1"]
            best_model = model
            best_name = name
            
    df = pd.DataFrame(rows).sort_values("F1", ascending=False)
    df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)
    
    with open(os.path.join(output_dir, "model_comparison.json"), "w") as f:
        json.dump(df.to_dict("records"), f, indent=2)
        
    return best_model, best_name, df