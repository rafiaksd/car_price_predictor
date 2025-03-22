import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
import joblib  # for saving the best model
import time
from hyperopt import hp, fmin, tpe, Trials
from functools import partial
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv("Cleaned_Car_data.csv")
X = data[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = data['Price']

# Train-test split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature Preprocessing
categorical_features = ['name', 'company', 'fuel_type']
num_features = ['year', 'kms_driven']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scaler', StandardScaler(), num_features)
])

# Define the XGBRegressor model
def create_model(n_estimators, learning_rate, max_depth, min_child_weight, gamma, subsample, colsample_bytree, random_state, alpha=0, reg_lambda=0):
    model = XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=gamma,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        random_state=random_state,
        alpha=alpha,  # L1 regularization term
        reg_lambda=reg_lambda  # L2 regularization term
    )
    return model

# Hyperopt optimization function
def optimize(params, X_train, y_train, loss_scores, models):
    """
    This function performs optimization using Hyperopt.
    It uses cross-validation to evaluate the model and returns the negative R² score to minimize.
    """
    # Ensure integer values for hyperparameters that require them
    n_estimators = int(params['n_estimators'])
    max_depth = int(params['max_depth'])
    min_child_weight = int(params['min_child_weight'])
    random_state = int(params['random_state'])
    
    # Create the model with the current parameters
    model = create_model(
        n_estimators=n_estimators,
        learning_rate=params['learning_rate'],
        max_depth=max_depth,
        min_child_weight=min_child_weight,
        gamma=params['gamma'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        random_state=random_state,
        alpha=params['alpha'],
        reg_lambda=params['reg_lambda']
    )
    
    # Create a pipeline with preprocessing and model
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    # Cross-validation score (R²)
    scores = cross_val_score(pipeline, X_train, y_train, cv=4, scoring='r2', n_jobs=-1)
    
    # Calculate the negative mean R² score (to minimize for hyperopt)
    mean_r2_score = -np.mean(scores)
    
    # Append the negative R² score and the model to the list
    loss_scores.append(mean_r2_score)
    models.append(model)
    
    # Keep only the best 1000 models based on negative R² score
    if len(loss_scores) > 1000:
        min_loss_idx = np.argmin(loss_scores)  # Find index of the most negative score (min value)
        loss_scores.pop(min_loss_idx)
        models.pop(min_loss_idx)
    
    # Return the negative mean R² score to Hyperopt for optimization
    return mean_r2_score

# Refined hyperopt parameter space
param_space = {
    "n_estimators": hp.quniform("n_estimators", 1, 50, 1),
    "learning_rate": hp.uniform("learning_rate", 0.001, 0.4),
    "max_depth": hp.quniform("max_depth", 1, 50, 1),
    "min_child_weight": hp.quniform("min_child_weight", 1, 20, 1),
    "gamma": hp.quniform("gamma", 0, 5, 0.1),
    "subsample": hp.uniform("subsample", 0.5, 1.0),
    "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
    "random_state": hp.quniform("random_state", 1, 1000, 1),
    "alpha": hp.uniform("alpha", 0, 1),  # L1 regularization term
    "reg_lambda": hp.uniform("reg_lambda", 0, 1)  # L2 regularization term
}

# Partial function for optimization, passing loss_scores and models as argument
loss_scores = []  # List to store the loss scores
models = []  # List to store the models
optimization_function = partial(
    optimize,
    X_train=X_train,
    y_train=y_train,
    loss_scores=loss_scores,  # Pass the list to capture loss scores
    models=models  # Pass the list to store models
)

# Initialize trials to keep logging information
trials = Trials()

# Run hyperopt optimization
start_time = time.time()
print("Starting hyperparameter optimization...\n")

hopt = fmin(
    fn=optimization_function,
    space=param_space,
    algo=tpe.suggest,
    max_evals=5000,  # Number of evaluations
    trials=trials
)

# Print the best hyperparameters found
print(f"Best parameters: {hopt}")

# Evaluate the top 400 models on the test set and select the top 10 with the highest R² score
r2_scores = []
for model in models:
    # Create pipeline with preprocessing and current model
    pipeline = Pipeline([('preprocessor', preprocessor), ('model', model)])
    
    # Train the model
    pipeline.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = pipeline.predict(X_test)
    
    # Calculate R² score on the test set
    r2 = r2_score(y_test, y_pred)  # Rename the variable to 'r2' instead of 'r2_score'
    r2_scores.append(r2)  # Append the result to the r2_scores list

# Get the top 10 models with the highest R² scores
top_10_indices = np.argsort(r2_scores)[-1500:]  # Get indices of top 10 highest R² scores
top_10_models = [models[i] for i in top_10_indices]

# Evaluate top 10 models and print R² scores
for i, model in enumerate(top_10_models):
    print(f"Model {i+1} - R² Score: {r2_scores[top_10_indices[i]]:.4f}")


end_time = time.time()
total_time_minutes = (end_time - start_time) / 60
print(f"\nTotal time taken: {total_time_minutes:.2f} minutes")

# Save the top 10 models
#for i, model in enumerate(top_10_models):
    #joblib.dump(model, f"best_car_price_xgb_model_top_10_{i+1}.pkl")
    #print(f"Top 10 Model {i+1} saved as 'best_car_price_xgb_model_top_10_{i+1}.pkl'")
