# %%
import matplotlib.pyplot as plt
import pandas as pd

# import seaborn as sns
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
from ucimlrepo import fetch_ucirepo

# %%

# Load dataset
dataset = fetch_ucirepo(id=544)
if dataset is None:
    print("Dataset not found")
    exit()

# Split features and target
X = dataset.data.features
y = dataset.data.targets

# %%
# Rename columns to more descriptive and Pythonic names
feature_mapping = {
    "Gender": "gender",
    "Age": "age",
    "Height": "height",
    "Weight": "weight",
    "family_history_with_overweight": "family_history_overweight",
    "FAVC": "high_caloric_food",
    "FCVC": "freqof_vegetables",
    "NCP": "number_of_main_meals",
    "CAEC": "food_between_meals",
    "SMOKE": "smoking",
    "CH2O": "water_intake",
    "SCC": "calories_monitoring",
    "FAF": "freq_physical_activity",
    "TUE": "time_using_technology",
    "CALC": "alcohol_consumption",
    "MTRANS": "transportation_type",
}
target_mapping = {
    "NObeyesdad": "obesity_level",
}

X = X.rename(columns=feature_mapping)
y = y.rename(columns=target_mapping)

print("Renamed feature and target columns.")

# %%
# Check for missing values
print("Checking for NaN/null values:")
print("Features (X):")
print(X.isna().sum())
print(f"Total missing in X: {X.isna().sum().sum()}")

print("\nTarget (y):")
print(y.isna().sum())
print(f"Total missing in y: {y.isna().sum().sum()}")

# Basic dataset information
print("\nDataset shape:")
print(f"X: {X.shape}")
print(f"y: {y.shape}")
print("\nFeature columns:", X.columns.tolist())
print("Target column:", y.columns.tolist())

# %%
# Target class distribution
print("\nTarget class distribution:")
print(
    y["obesity_level"].value_counts(normalize=True).mul(100).round(2).astype(str) + "%"
)

# %%
# Check distributions of categorical features
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
print("\nCategorical features in X:")
for col in categorical_columns:
    print(f"\n{col} distribution:")
    print(X[col].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# Check distributions of numerical features
numerical_columns = X.select_dtypes(include=["int", "float"]).columns
numerical_cat_columns = []

print("\nNumerical features in X:")
for col in numerical_columns:
    unique_values = X[col].nunique()
    if col in ["age", "height", "weight", "number_of_main_meals"]:
        # These are clearly continuous variables, so no need to treat as categorical
        print(f"{col} is a numerical column.")
    else:
        # Variables with few unique numeric values might represent categorical choices
        numerical_cat_columns.append(col)
        print(f"\n{col} distribution ({unique_values} unique values):")
        print(X[col].value_counts(normalize=True).mul(100).round(2).astype(str) + "%")

# %%
# Preprocessing step: round age to nearest integer and convert to float
# Helps model generalize better by focusing on completed years rather than decimal values
X["age"] = X["age"].round().astype(float)
print("Rounded 'age' to 0 decimal places and converted to float.")

# Round continuous variables (height, weight) to 2 decimal points
# Reduces noise and creates more consistent buckets
for col in ["height", "weight"]:
    X[col] = X[col].round(2).astype(float)
    print(f"Rounded '{col}' to 2 decimal points and converted to float.")

# %%

# Round number_of_main_meals to nearest integer
# This is a discrete variable, so rounding to nearest integer makes sense
# Helps model generalize better by focusing on completed meals rather than decimal values
X["number_of_main_meals"] = X["number_of_main_meals"].round().astype(float)

# %%
# Numerical features that are actually categorical (like Likert scales)
# First round to nearest class, then convert to category dtype
for col in numerical_cat_columns:
    X[col] = pd.Categorical(X[col].round().astype(int))
    print(f"Converted '{col}' to category dtype (rounded integer values).")

# %%
# Convert all object-typed columns to category dtype
# Helps reduce memory usage and ensures consistency for categorical encoding
for col in categorical_columns:
    X[col] = X[col].astype("category")
    print(f"Converted '{col}' to category dtype.")

# %%
# Categorizing features for modeling

# Ordinal categories (order matters)
ordinal_cat = [
    "freqof_vegetables",  # frequency of vegetable consumption
    "food_between_meals",  # snacking habits
    "water_intake",  # daily water consumption
    "freq_physical_activity",  # frequency of physical activity
    "time_using_technology",  # sedentary time
    "alcohol_consumption",  # frequency of alcohol use
]

# Nominal categories (no inherent order)
nominal_cat = [
    "gender",
    "family_history_overweight",
    "high_caloric_food",
    "smoking",
    "calories_monitoring",
    "transportation_type",
]

# Continuous numerical features
ratio_cat = ["number_of_main_meals", "height", "weight"]

# Discrete numeric features
discrete_cat = ["age"]

# %%
# Target label is ordinal — represents increasing severity of obesity
# Mapping used for ordinal encoding
obesity_level_mapping = {
    "Insufficient_Weight": -1,  # Less than normal weight
    "Normal_Weight": 0,
    "Overweight_Level_I": 1,
    "Overweight_Level_II": 2,
    "Obesity_Type_I": 3,
    "Obesity_Type_II": 4,
    "Obesity_Type_III": 5,
}


# encoded by hand
y["obesity_level"] = y["obesity_level"].map(obesity_level_mapping)
print("Mapped target labels to ordinal values.")
# %%
# get all possible values for each ordinal category
categories_values = []
for col in ordinal_cat:
    order = X[col].cat.categories.tolist()
    try:
        order = [int(i) for i in order]
        order = sorted(set(order))
    except ValueError:
        order = sorted(set(order), reverse=True)
    categories_values.append(order)
    print(f"Gathered possible values for '{col}': {order}")

print("Gathered all possible values for ordinal categories")

# %%

# setup encoders for categoricals
# use one hot encoder for nominal
nominal_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
# ordinal encoder for ordinal
ordinal_encoder = OrdinalEncoder(
    categories=categories_values, handle_unknown="use_encoded_value", unknown_value=-1
)

# %%
print("\nPreprocessing completed. Dataset is ready for modeling or further steps.")


# %%

# even though we made the analysis before and there were no null values
# better safe than sorry
numerical_imputer = SimpleImputer(strategy="mean")
categorical_imputer = SimpleImputer(strategy="most_frequent")


# Pipeline for ordinal features
ordinal_pipeline = Pipeline(
    steps=[("imputer", categorical_imputer), ("ordinal_encoder", ordinal_encoder)]
)

# Pipeline for nominal features
nominal_pipeline = Pipeline(
    steps=[("imputer", categorical_imputer), ("onehot_encoder", nominal_encoder)]
)

# Pipeline for ratio (continuous) numerical features
ratio_pipeline = Pipeline(
    steps=[("imputer", numerical_imputer), ("scaler", StandardScaler())]
)

# Pipeline for discrete numerical features (e.g., age)
discrete_pipeline = Pipeline(
    steps=[("imputer", numerical_imputer), ("scaler", StandardScaler())]
)

# combine all with ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ("ord", ordinal_pipeline, ordinal_cat),
        ("nom", nominal_pipeline, nominal_cat),
        ("rat", ratio_pipeline, ratio_cat),
        ("dis", discrete_pipeline, discrete_cat),
    ]
)

# %%


# First split: 60% train, 40% temp (validation + test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)
# Confirm the shapes
print("Train set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)

# Optionally check target distribution in each
for label, split in zip(["Train", "Test"], [y_train, y_test]):
    print(f"\n{label} target distribution:")
    print(split["obesity_level"].value_counts(normalize=True).round(3))
# %%
## KNN


# Create a pipeline with preprocessing and KNN classifier
knn_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", KNeighborsClassifier())]
)
# make grid search

knn_grid = {
    "classifier__n_neighbors": [3, 5, 7, 9],
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan"],
}


# since classes are distributed:
# f1_macro =
# Classes are balanced and you want equal treatment per class


# Create a GridSearchCV object
knn_grid_search = GridSearchCV(
    knn_pipeline,
    param_grid=knn_grid,
    scoring="f1_macro",
    cv=10,  # simulate LOOCV
    verbose=1,
    n_jobs=-1,
)

knn_grid_search.fit(X_train, y_train.values.ravel())
print("Best parameters:", knn_grid_search.best_params_)
print("Best cross-validated training score (F1 macro):", knn_grid_search.best_score_)

best_knn_model = knn_grid_search.best_estimator_
# Get the best parameters

# use cross validation


# Perform cross-validation to get scores
cv_scores = cross_val_score(
    best_knn_model,
    X_test,
    y_test.values.ravel(),
    cv=10,  # 10 fold to simualte LOOCV
    scoring="f1_macro",
    n_jobs=-1,
)

# Print cross-validation results
print(f"Cross-validation F1 scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")

# %%


# Create an inverse mapping to convert numeric predictions back to string labels.
inverse_mapping = {
    -1: "Insufficient_Weight",
    0: "Normal_Weight",
    1: "Overweight_Level_I",
    2: "Overweight_Level_II",
    3: "Obesity_Type_I",
    4: "Obesity_Type_II",
    5: "Obesity_Type_III",
}

# Convert y_test to string labels
y_test_str = y_test["obesity_level"].map(inverse_mapping)

# Make predictions on the test set
y_pred = best_knn_model.predict(X_test)
y_pred_str = pd.Series(y_pred).map(inverse_mapping)

# Define the string labels for display
y_labels = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]

# === Confusion Matrix ===
cm = confusion_matrix(y_test_str, y_pred_str, labels=y_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# Classification report for precision/recall/F1 per class
print("\nClassification Report:")
print(classification_report(y_test_str, y_pred_str))

# === SHAP EXPLAINER ===
# Extract the model and the preprocessed data
X_test_transformed = best_knn_model.named_steps["preprocessor"].transform(X_test)
model_knn = best_knn_model.named_steps["classifier"]

# SHAP only supports KNN via KernelExplainer
# Use 100 samples for performance reasons
X_sample = shap.sample(X_test_transformed, 100, random_state=42)

explainer = shap.KernelExplainer(model_knn.predict_proba, X_sample)
shap_values = explainer.shap_values(X_sample)

# Plot SHAP summary
shap.summary_plot(shap_values, X_sample, show=True)


# %%
# TODO
# add gridsearch for parameters?
# train with each model
# add expaliner
