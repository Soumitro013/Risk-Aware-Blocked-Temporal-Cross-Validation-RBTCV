import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from RBTCV import rbtc_cv


# Load data
data = pd.read_csv("SCADA.csv")

# IMPORTANT:
# If a timestamp column exists, sort the data by it
# before extracting X and y.
#
# Example:
# data["Timestamp"] = pd.to_datetime(data["Timestamp"])
# data = data.sort_values("Timestamp").reset_index(drop=True)

features = [
    "Generator temperature",
    "Bearing temperature",
    "Vibration (g)",
    "Power (kw)"
]

X = data[features].values
y = data["Fault Type"].values

classes = [
    "Normal",
    "Bearing Fault",
    "Overheating Fault",
    "Electrical Fault"
]

# Rows = true class; columns = predicted class
cost_matrix = [
    [0, 0.5, 0.7, 0.7],
    [2, 0, 1, 1.5],
    [6, 4, 0, 2],
    [10, 7, 5, 0]
]

# Predictive model
model = DecisionTreeClassifier(
    max_depth=4,
    min_samples_leaf=2,
    random_state=42
)

# Run RBTCV
results = rbtc_cv(
    X=X,
    y=y,
    estimator=model,
    classes=classes,
    cost_matrix=cost_matrix,
    n_splits=3,
    gap=2,
    test_size=None,
    initial_train_fraction=0.5
)

print("\n--- RBTCV Fold Results ---")
print(results["fold_results"].to_string(index=False))

print("\nPooled ordinary classification error:")
print(round(results["ordinary_error"], 4))

print("\nPooled risk-aware error:")
print(round(results["risk_error"], 4))

print("\nPooled macro-F1:")
print(round(results["macro_f1"], 4))

print("\nPer-class recall:")
for label, value in results["class_recall"].items():
    print(f"{label}: {value:.4f}")

print("\nConfusion matrix:")
print(results["confusion_matrix"])
