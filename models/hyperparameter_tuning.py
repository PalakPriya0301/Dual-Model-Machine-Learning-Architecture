import os
import json
from sklearn.model_selection import GridSearchCV

def tune_model(model_name, model, X, y, output_dir):
    # Define hyperparameter grids for different models
    grids = {
        "Random Forest": {'n_estimators': [100, 300], 'max_depth': [5, 8, 12], 'min_samples_split': [5, 10]},
        "Extra Trees": {'n_estimators': [100, 300], 'max_depth': [5, 8, 12]},
        "Gradient Boosting": {'learning_rate': [0.01, 0.1], 'n_estimators': [100, 200]},
        "XGBoost": {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]},
        "Decision Tree": {'max_depth': [3, 5, 10]}
    }
    
    if model_name in grids:
        grid = GridSearchCV(model, grids[model_name], cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X, y)
        
        # Save tuning results
        results_path = os.path.join(output_dir, f"{model_name.replace(' ', '_')}_tuning.json")
        with open(results_path, "w") as f:
            json.dump(grid.best_params_, f, indent=2)
            
        return grid.best_estimator_, grid.best_params_, grid.best_score_
    
    # If no grid defined, just fit the baseline model
    model.fit(X, y)
    return model, {"info": "No tuning grid defined, used defaults"}, 0.0