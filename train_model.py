import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score, classification_report


# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv("data/Crop_recommendation.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ==========================================
# 2. SEPARATE INPUTS AND TARGET
# ==========================================

X = df[
    [
        "N",
        "P",
        "K",
        "temperature",
        "humidity",
        "ph",
        "rainfall"
    ]
]

y = df["label"]


# ==========================================
# 3. SPLIT DATA
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 4. SCALE DATA
# ==========================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================================
# 5. CREATE MODELS
# ==========================================

models = {
    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    ),

    "KNN": KNeighborsClassifier(n_neighbors=5),

    "SVM": SVC(kernel="rbf", random_state=42)
}


# ==========================================
# 6. TRAIN AND COMPARE MODELS
# ==========================================

results = {}

print("\n========== MODEL RESULTS ==========")

for name, model in models.items():

    # Tree models can use original data
    if name == "Decision Tree" or name == "Random Forest":

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

    else:

        model.fit(X_train_scaled, y_train)

        predictions = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, predictions)

    results[name] = accuracy

    print(f"\n{name}: {accuracy:.4f}")


# ==========================================
# 7. DISPLAY COMPARISON
# ==========================================

print("\n========== MODEL COMPARISON ==========")

for name, accuracy in results.items():
    print(f"{name}: {accuracy * 100:.2f}%")


# ==========================================
# 8. TRAIN FINAL MODEL
# ==========================================

# We use Random Forest as our final model.
# The model comparison above lets us verify
# how it performs against other algorithms.

final_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

final_model.fit(X, y)


# ==========================================
# 9. SAVE MODEL
# ==========================================

joblib.dump(final_model, "model/crop_model.pkl")

print("\nFinal model saved successfully!")
print("Location: model/crop_model.pkl")