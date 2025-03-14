import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

import joblib #exporting model

# Load dataset
data = pd.read_csv("Cleaned_Car_data.csv")
X = data[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = data['Price']

# Feature Preprocessing
categorical_features = ['name', 'company', 'fuel_type']
num_features = ['year', 'kms_driven']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scaler', StandardScaler(), num_features)
])

# Variables to store best results
best_r2_score = float('-inf')  # Initialize as the lowest possible value
best_std_dev = float('inf')

# Iterate over test_size from 1% to 100%
for tester in range(1, 3):  
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=399/1000, random_state=88)

    # Define model
    model = GradientBoostingRegressor(n_estimators=20, learning_rate=413/1000, max_depth=17, random_state=119)

    # Create Pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Train model
    pipeline.fit(X_train, y_train)

    # Perform cross-validation
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    mean_r2 = np.mean(cv_scores)
    std_r2 = np.std(cv_scores)

    # Check if the current mean R² score is the best
    if mean_r2 > best_r2_score :
        best_r2_score = mean_r2
        best_std_dev = std_r2
        print(f"✅ New Best R²: {best_r2_score:.3f} Std Dev: {std_r2:.3f} (Tester: {tester:.2f} ) | Cross-validation R² scores: {cv_scores}")

        # set tester in range(1, 3)
        #Grad_boost_regressor R2 Score : 0.775
        #joblib.dump(pipeline, "car_price_grad_boost_model.pkl") 
        #print("✅ Model saved as car_price_grad_boost_model.pkl")
    elif mean_r2*1.02 > best_r2_score and std_r2 < 1.08*best_std_dev :
        print(f"👀 New Wow! R²: {best_r2_score:.3f} Std Dev: {std_r2:.3f} (Tester: {tester:.2f} ) | Cross-validation R² scores: {cv_scores}")
    elif tester % 2 == 0:
        print(f"R²: {best_r2_score:.3f} Std Dev: {std_r2:.3f} (Tester: {tester:.2f} ) | Cross-validation R² scores: {cv_scores}")
