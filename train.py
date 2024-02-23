import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
  r2_score, 
  mean_absolute_error, 
  root_mean_squared_error
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor

def train():
    """Trains a Gradient Boosting model on the dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = [
       "nbr_bedrooms", "total_area_sqm", "surface_land_sqm", 
       "nbr_frontages", "latitude", "terrace_sqm", "garden_sqm"
       ]
    fl_features = ["fl_swimming_pool", "fl_floodzone"]
    cat_features = ["locality", "subproperty_type", "region"]

    # Split the data into features and target
    X = data[num_features + fl_features + cat_features]
    y = data["price"]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=505
    )

    # Impute missing values using SimpleImputer
    imputer = SimpleImputer(strategy="mean")
    imputer.fit(X_train[num_features])
    X_train[num_features] = imputer.transform(X_train[num_features])
    X_test[num_features] = imputer.transform(X_test[num_features])

    # Convert categorical columns with one-hot encoding using OneHotEncoder
    enc = OneHotEncoder()
    enc.fit(X_train[cat_features])
    X_train_cat = enc.transform(X_train[cat_features]).toarray()
    X_test_cat = enc.transform(X_test[cat_features]).toarray()

    # Combine the numerical and one-hot encoded categorical columns
    X_train = pd.concat(
       [X_train[num_features + fl_features].reset_index(drop=True), 
        pd.DataFrame(X_train_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
        )

    X_test = pd.concat(
       [X_test[num_features + fl_features].reset_index(drop=True),
        pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out()),
        ],
        axis=1,
        )

    # Print list of features used in the model
    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the Gradient Boosting model
    gradient_boosting_model = GradientBoostingRegressor(
       n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
       )
    gradient_boosting_model.fit(X_train, y_train)

    # Evaluate the model
    gradient_boosting_score = r2_score(
       y_train, gradient_boosting_model.predict(X_train)
       )
    gradient_boosting_test_score = r2_score(
       y_test, gradient_boosting_model.predict(X_test)
       )

    # Calculate MAE and RMSE
    gradient_boosting_train_mae = mean_absolute_error(
       y_train, gradient_boosting_model.predict(X_train)
       )
    gradient_boosting_test_mae = mean_absolute_error(
       y_test, gradient_boosting_model.predict(X_test)
       )
    gradient_boosting_train_rmse = root_mean_squared_error(
       y_train, gradient_boosting_model.predict(X_train)
       )
    gradient_boosting_test_rmse = root_mean_squared_error(
       y_test, gradient_boosting_model.predict(X_test)
       )
    
    # Print model performance
    print("-" * 40)
    print("Gradient Boosting Model:")
    print("-" * 40)
    print(f"{'Train R²':10} {gradient_boosting_score:10.4f}")
    print(f"{'Test R²':10} {gradient_boosting_test_score:10.4f}")
    print(f"{'Train MAE':10} {gradient_boosting_train_mae:10.4f}")
    print(f"{'Test MAE':10} {gradient_boosting_test_mae:10.4f}")
    print(f"{'Train RMSE':10} {gradient_boosting_train_rmse:10.4f}")
    print(f"{'Test RMSE':10} {gradient_boosting_test_rmse:10.4f}")

    # Save the model
    artifacts = {
       "features": {
          "num_features": num_features,
          "fl_features": fl_features,
          "cat_features": cat_features,
          },
       "imputer": imputer,
       "enc": enc,
       "model": gradient_boosting_model,
       }

    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
  train()
