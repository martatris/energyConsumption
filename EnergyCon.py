# energy_consumption_advanced_models.py
import pandas as pd
import numpy as np
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

warnings.filterwarnings("ignore")

# -----------------------------
# 1️⃣ LOAD DATA
# -----------------------------
df = pd.read_csv("AEP_hourly.csv", parse_dates=["Datetime"])
df = df.rename(columns={"AEP_MW": "Energy_MW"})
df = df.dropna()

# Feature Engineering
df["hour"] = df["Datetime"].dt.hour
df["day"] = df["Datetime"].dt.day
df["month"] = df["Datetime"].dt.month
df["year"] = df["Datetime"].dt.year
df["dayofweek"] = df["Datetime"].dt.dayofweek

# -----------------------------
# 2️⃣ SPLIT DATA
# -----------------------------
X = df[["hour", "day", "month", "year", "dayofweek"]]
y = df["Energy_MW"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, shuffle=False
)

# -----------------------------
# 3️⃣ DEFINE MODELS (Tuned)
# -----------------------------
models = {
    # Linear Regression (no hyperparameters, but included for baseline)
    "Linear Regression": LinearRegression(),

    # Decision Tree
    "Decision Tree": DecisionTreeRegressor(
        criterion="squared_error",
        splitter="best",
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42
    ),

    # Random Forest
    "Random Forest": RandomForestRegressor(
        n_estimators=400,
        max_depth=15,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        random_state=42,
        n_jobs=-1
    ),

    # Gradient Boosting
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=500,
        learning_rate=0.03,
        subsample=0.9,
        max_depth=6,
        min_samples_split=4,
        min_samples_leaf=2,
        loss="squared_error",
        random_state=42
    ),

    # XGBoost
    "XGBoost": XGBRegressor(
        n_estimators=700,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_lambda=1.2,
        reg_alpha=0.4,
        min_child_weight=3,
        random_state=42,
        n_jobs=1,  # safer for macOS
        tree_method="hist",
        objective="reg:squarederror"
    ),

    # LightGBM
    "LightGBM": LGBMRegressor(
        n_estimators=700,
        learning_rate=0.03,
        num_leaves=60,
        max_depth=-1,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.2,
        reg_alpha=0.4,
        random_state=42,
        n_jobs=1,
        force_col_wise=True
    ),

    # Support Vector Regressor
    "SVR (RBF)": SVR(
        kernel="rbf",
        C=500,
        gamma=0.05,
        epsilon=0.1
    ),

    # Neural Network (MLP)
    "Neural Network (MLP)": MLPRegressor(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.0005,
        learning_rate_init=0.001,
        batch_size=128,
        max_iter=400,
        early_stopping=True,
        random_state=42
    ),
}

# -----------------------------
# 4️⃣ TRAIN & EVALUATE
# -----------------------------
results = []

for name, model in models.items():
    print(f"\n🔹 Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"{name} → MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

    results.append(
        {
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
        }
    )

# -----------------------------
# 5️⃣ SAVE & DISPLAY RESULTS
# -----------------------------
results_df = pd.DataFrame(results).sort_values(by="RMSE", ascending=True)
results_df.reset_index(drop=True, inplace=True)

results_df.to_csv("model_results.csv", index=False)

print("\n✅ All models trained successfully!")
print("\n📊 Summary of Results:")
print(results_df)