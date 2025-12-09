import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def create_preprocessor(numerical_cols, categorical_cols):
    """
    Creates a ColumnTransformer pipeline.
    """
    # Numerical: Impute -> Scale
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Categorical: Impute -> OneHotEncode
    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


    # Combine
    preprocessor = ColumnTransformer([
        ('num', num_pipeline, numerical_cols),
        ('cat', cat_pipeline, categorical_cols)
    ], verbose_feature_names_out=False)
    
    return preprocessor


################### This will clean the data with the names
def process_data(df, numerical_cols, categorical_cols):
    """
    Applies the preprocessor and returns a clean DataFrame with column names.
    """
    preprocessor = create_preprocessor(numerical_cols, categorical_cols)
    
    # Fit and Transform
    X_processed_array = preprocessor.fit_transform(df)
    
    # Attempt to recover column names for better visibility
    try:
        feature_names = preprocessor.get_feature_names_out()
        X_processed = pd.DataFrame(X_processed_array, columns=feature_names)
    except:
        X_processed = pd.DataFrame(X_processed_array)
        
    return X_processed

