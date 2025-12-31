import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def create_preprocessor(num_cols, cat_cols, scaling="Standard", imputation="Median"):
    # Numerical pipeline
    if scaling == "Standard":
        scaler = StandardScaler()
    elif scaling == "MinMax":
        scaler = MinMaxScaler()
    else:
        scaler = "passthrough"


    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy=imputation.lower())),
        ("scaler", scaler)
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    

    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_cols),
        ("cat", cat_pipeline, cat_cols)
    ])

    return preprocessor



def process_data(df, preprocessor, fit=True):
    X = preprocessor.fit_transform(df) if fit else preprocessor.transform(df)

    try:
        cols = preprocessor.get_feature_names_out()
        return pd.DataFrame(X, columns=cols, index=df.index)
    except:
        return pd.DataFrame(X, index=df.index)