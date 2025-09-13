if 'df' in locals() and df is not None:
    print("DataFrame 'df' found. Proceeding with preprocessing.")

    # Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if df[col].dtype == 'object':
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                print(f"Filled missing values in column '{col}' with mode: {mode_val}")
            else:
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
                print(f"Filled missing values in column '{col}' with mean: {mean_val}")
        else:
            print(f"Column '{col}' has no missing values.")

    # Identify categorical and numerical features
    categorical_features = df.select_dtypes(include=['object']).columns
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns
    print(f"Identified {len(categorical_features)} categorical features and {len(numerical_features)} numerical features.")
    print(f"Categorical features: {list(categorical_features)}")
    print(f"Numerical features: {list(numerical_features)}")

    # Separate target variable
    if 'Performance' in df.columns:
        X = df.drop('Performance', axis=1)
        y = df['Performance']
        print("Separated features (X) and target (y).")

        # Create a column transformer for preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)])

        # Create a preprocessing pipeline
        preprocess_pipeline = Pipeline(steps=[('preprocessor', preprocessor)])

        # Apply preprocessing
        X_processed = preprocess_pipeline.fit_transform(X)
        print("Applied one-hot encoding to categorical features.")

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
        print("Split data into training and testing sets (80/20 split).")
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"y_test shape: {y_test.shape}")

    else:
        print("Error: Target column 'Performance' not found in the DataFrame.")
        X_train, X_test, y_train, y_test = None, None, None, None

else:
    print("Error: DataFrame 'df' not found or is None. Data was not loaded in the previous step.")
    X_train, X_test, y_train, y_test = None, None, None, None
