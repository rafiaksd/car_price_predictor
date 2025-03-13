import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import r2_score


data = pd.read_csv("Cleaned_Car_data.csv") 

X = data[['name', 'company', 'year', 'kms_driven', 'fuel_type']]
y = data['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=600)

# Feature Preprocessing:
# - OneHotEncoding for categorical features
# - StandardScaler for numerical features
categorical_features = ['name', 'company', 'fuel_type']
num_features = ['year', 'kms_driven']

preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(handle_unknown='ignore'), categorical_features),
    ('scaler', StandardScaler(), num_features)
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=160/1000, random_state=124)

# Model: Gradient Boosting Regressor
model = GradientBoostingRegressor(n_estimators=15, learning_rate=259/1000, max_depth=17, random_state=300)

# Create Pipeline
pipeline = Pipeline([
     ('preprocessor', preprocessor),
     ('model', model)
])

# Train Model
pipeline.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
print('r2 score: ' + str(r2))


# Save the trained model
import joblib #exporting model
#joblib.dump(pipeline, "car_price_model.pkl")
#print("✅ Model saved as car_price_model.pkl")