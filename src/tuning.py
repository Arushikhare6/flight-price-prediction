def get_param_distributions(model_name):
    param_grids = {
        "Random Forest": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None],
        },

        "Extra Trees": {
            "model__n_estimators": [100, 200, 300, 500],
            "model__max_depth": [None, 10, 20, 30, 40],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        },

        "Gradient Boosting": {
            "model__n_estimators": [100, 200, 300],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 5, 7, 9],
            "model__subsample": [0.7, 0.85, 1.0],
        },

        "XGBoost": {
            "model__n_estimators": [200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__max_depth": [3, 5, 7, 9],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
        },

        "LightGBM": {
            "model__n_estimators": [200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__num_leaves": [20, 31, 50, 70],
            "model__max_depth": [-1, 5, 10, 15],
            "model__subsample": [0.7, 0.85, 1.0],
        },

        "CatBoost": {
            "model__iterations": [200, 300, 500],
            "model__learning_rate": [0.01, 0.05, 0.1],
            "model__depth": [4, 6, 8, 10],
            "model__l2_leaf_reg": [1, 3, 5, 7],
        },
    }

    if model_name not in param_grids:
        raise ValueError(
            f"No hyperparameter grid defined for '{model_name}'. "
            f"Available: {list(param_grids.keys())}"
        )

    return param_grids[model_name]