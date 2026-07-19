from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def create_preprocessor(X):
    

    categorical_features = X.select_dtypes(
        include="object"
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features,
            )
        ],
        remainder="passthrough",
    )

    return preprocessor