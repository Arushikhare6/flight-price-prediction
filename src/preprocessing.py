from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def create_preprocessor(X, scale_numeric: bool = False):
    """
    Create a scikit-learn ColumnTransformer preprocessor.

    Parameters:
        X (pd.DataFrame): Input feature DataFrame.
        scale_numeric (bool): If True, applies StandardScaler to numeric features.
                              If False, passes numeric features through as-is.

    Returns:
        ColumnTransformer: Configured preprocessor.
    """
    categorical_features = X.select_dtypes(include="object").columns.tolist()
    numeric_features = X.select_dtypes(exclude="object").columns.tolist()

    if scale_numeric:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                ),
                (
                    "numeric",
                    StandardScaler(),
                    numeric_features,
                ),
            ],
            remainder="drop",
        )
    else:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore"),
                    categorical_features,
                )
            ],
            remainder="passthrough",
        )

    return preprocessor


def get_preprocessor_for_model(X, model_name: str):
    """
    Helper function to select appropriate preprocessing based on model requirements.
    Applies StandardScaler for Linear Regression, while leaving numeric features unscaled
    for tree-based ensembles (Random Forest, XGBoost, LightGBM, CatBoost, etc.).
    """
    if model_name == "Linear Regression":
        return create_preprocessor(X, scale_numeric=True)
    return create_preprocessor(X, scale_numeric=False)