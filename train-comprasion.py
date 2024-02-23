import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

def train():
    """Trains a linear and three non-linear regression models on the full dataset and stores output."""
    # Load the data
    data = pd.read_csv("data/properties.csv")

    # Define features to use
    num_features = ["nbr_bedrooms", "total_area_sqm", "surface_land_sqm", "nbr_frontages", "latitude", "terrace_sqm", "garden_sqm"]
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

    print(f"Features: \n {X_train.columns.tolist()}")

    # Train the models
    linear_model = LinearRegression()
    linear_model.fit(X_train, y_train)

    decision_tree_model = DecisionTreeRegressor(max_depth=5, random_state=42)
    decision_tree_model.fit(X_train, y_train)

    random_forest_model = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
    random_forest_model.fit(X_train, y_train)
    
    gradient_boosting_model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
    gradient_boosting_model.fit(X_train, y_train)

    # Evaluate the models
    linear_score = r2_score(y_train, linear_model.predict(X_train))
    linear_test_score = r2_score(y_test, linear_model.predict(X_test))

    decision_tree_score = r2_score(y_train, decision_tree_model.predict(X_train))
    decision_tree_test_score = r2_score(y_test, decision_tree_model.predict(X_test))

    random_forest_score = r2_score(y_train, random_forest_model.predict(X_train))
    random_forest_test_score = r2_score(y_test, random_forest_model.predict(X_test))

    gradient_boosting_score = r2_score(y_train, gradient_boosting_model.predict(X_train))
    gradient_boosting_test_score = r2_score(y_test, gradient_boosting_model.predict(X_test))

    # Calculate MAE and RMSE for all models
    linear_train_mae = mean_absolute_error(y_train, linear_model.predict(X_train))
    linear_test_mae = mean_absolute_error(y_test, linear_model.predict(X_test))
    linear_train_rmse = root_mean_squared_error(y_train, linear_model.predict(X_train))
    linear_test_rmse = root_mean_squared_error(y_test, linear_model.predict(X_test))

    decision_tree_train_mae = mean_absolute_error(y_train, decision_tree_model.predict(X_train))
    decision_tree_test_mae = mean_absolute_error(y_test, decision_tree_model.predict(X_test))
    decision_tree_train_rmse = root_mean_squared_error(y_train, decision_tree_model.predict(X_train))
    decision_tree_test_rmse = root_mean_squared_error(y_test, decision_tree_model.predict(X_test))

    random_forest_train_mae = mean_absolute_error(y_train, random_forest_model.predict(X_train))
    random_forest_test_mae = mean_absolute_error(y_test, random_forest_model.predict(X_test))
    random_forest_train_rmse = root_mean_squared_error(y_train, random_forest_model.predict(X_train))
    random_forest_test_rmse = root_mean_squared_error(y_test, random_forest_model.predict(X_test))

    gradient_boosting_train_mae = mean_absolute_error(y_train, gradient_boosting_model.predict(X_train))
    gradient_boosting_test_mae = mean_absolute_error(y_test, gradient_boosting_model.predict(X_test))
    gradient_boosting_train_rmse = root_mean_squared_error(y_train, gradient_boosting_model.predict(X_train))
    gradient_boosting_test_rmse = root_mean_squared_error(y_test, gradient_boosting_model.predict(X_test))
    
    print("-" * 40)
    print("Comprasion of models:")
    print("-" * 40)
    print(f"{'Model':15} {'Train R²':10} {'Test R²':10} {'Train MAE':10} {'Test MAE':10} {'Train RMSE':10} {'Test RMSE':10}")
    print("-" * 40)
    print(f"{'Linear':15} {linear_score:10.4f} {linear_test_score:10.4f} {linear_train_mae:10.4f} {linear_test_mae:10.4f} {linear_train_rmse:10.4f} {linear_test_rmse:10.4f}")
    print(f"{'Decision Tree':15} {decision_tree_score:10.4f} {decision_tree_test_score:10.4f} {decision_tree_train_mae:10.4f} {decision_tree_test_mae:10.4f} {decision_tree_train_rmse:10.4f} {decision_tree_test_rmse:10.4f}")
    print(f"{'Random Forest':15} {random_forest_score:10.4f} {random_forest_test_score:10.4f} {random_forest_train_mae:10.4f} {random_forest_test_mae:10.4f} {random_forest_train_rmse:10.4f} {random_forest_test_rmse:10.4f}")
    print(f"{'Gradient Boosting':15} {gradient_boosting_score:10.4f} {gradient_boosting_test_score:10.4f} {gradient_boosting_train_mae:10.4f} {gradient_boosting_test_mae:10.4f} {gradient_boosting_train_rmse:10.4f} {gradient_boosting_test_rmse:10.4f}")

    # Save the models
    artifacts = {
        "features": {
            "num_features": num_features,
            "fl_features": fl_features,
            "cat_features": cat_features,
            },
        "imputer": imputer,
        "enc": enc,
        "linear_model": linear_model,
        "decision_tree_model": decision_tree_model,
        "random_forest_model": random_forest_model,
        "gradient_boosting_model": gradient_boosting_model,
        }

    joblib.dump(artifacts, "models/artifacts.joblib")


if __name__ == "__main__":
   train()
