# %%
# === IMPORTS ===
# Standard libraries
import matplotlib.pyplot as plt
import pandas as pd
import shap

# Sklearn imports for preprocessing, modeling, evaluation
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB

# UCI dataset loader
from ucimlrepo import fetch_ucirepo

# %%
# === LOAD DATASET ===
print("Fetching dataset...")
dataset = fetch_ucirepo(id=544)
if dataset is None:
    print("Dataset not found")
    exit()

# Split features and target
X = dataset.data.features
y = dataset.data.targets

# %%
# === RENAME COLUMNS FOR CLARITY ===
print("Renaming columns...")
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

# %%
# === CHECK MISSING VALUES ===
print("Checking for missing values...")
print(X.isna().sum())
print(y.isna().sum())

# %%
# === DATA TYPE CLEANING AND ROUNDING ===
print("Cleaning data types and rounding values...")
X["age"] = X["age"].round().astype(float)
X["number_of_main_meals"] = X["number_of_main_meals"].round().astype(float)
for col in ["height", "weight"]:
    X[col] = X[col].round(2).astype(float)

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(include=["object", "category"]).columns
numerical_columns = X.select_dtypes(include=["int", "float"]).columns
numerical_cat_columns = []

# Detect which numerical features are likely categorical
for col in numerical_columns:
    if col not in ["age", "height", "weight", "number_of_main_meals"]:
        numerical_cat_columns.append(col)

# Convert numerical categorical features
for col in numerical_cat_columns:
    X[col] = pd.Categorical(X[col].round().astype(int))

# Convert object columns to category
for col in categorical_columns:
    X[col] = X[col].astype("category")

# %%
# === CATEGORIZE FEATURES ===
ordinal_cat = [
    "freqof_vegetables",
    "food_between_meals",
    "water_intake",
    "freq_physical_activity",
    "time_using_technology",
    "alcohol_consumption",
]
nominal_cat = [
    "gender",
    "family_history_overweight",
    "high_caloric_food",
    "smoking",
    "calories_monitoring",
    "transportation_type",
]
ratio_cat = ["number_of_main_meals", "height"]
discrete_cat = ["age"]

# %%
# === MAP TARGET LABELS TO INTEGERS ===
obesity_level_mapping = {
    "Insufficient_Weight": -1,
    "Normal_Weight": 0,
    "Overweight_Level_I": 1,
    "Overweight_Level_II": 2,
    "Obesity_Type_I": 3,
    "Obesity_Type_II": 4,
    "Obesity_Type_III": 5,
}
y["obesity_level"] = y["obesity_level"].map(obesity_level_mapping)

# %%
# === GATHER ORDERED CATEGORIES FOR ENCODING ===
categories_values = []
for col in ordinal_cat:
    order = X[col].cat.categories.tolist()
    try:
        order = sorted(set([int(i) for i in order]))
    except ValueError:
        order = sorted(set(order), reverse=True)
    categories_values.append(order)

# %%
# === BUILD PREPROCESSING PIPELINES ===
print("Setting up preprocessing pipelines...")

numerical_imputer = SimpleImputer(strategy="mean")
categorical_imputer = SimpleImputer(strategy="most_frequent")

ordinal_encoder = OrdinalEncoder(
    categories=categories_values, handle_unknown="use_encoded_value", unknown_value=-1
)
nominal_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)

ordinal_pipeline = Pipeline(
    steps=[("imputer", categorical_imputer), ("ordinal_encoder", ordinal_encoder)]
)
nominal_pipeline = Pipeline(
    steps=[("imputer", categorical_imputer), ("onehot_encoder", nominal_encoder)]
)
ratio_pipeline = Pipeline(
    steps=[("imputer", numerical_imputer), ("scaler", StandardScaler())]
)
discrete_pipeline = Pipeline(
    steps=[("imputer", numerical_imputer), ("scaler", StandardScaler())]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("ord", ordinal_pipeline, ordinal_cat),
        ("nom", nominal_pipeline, nominal_cat),
        ("rat", ratio_pipeline, ratio_cat),
        ("dis", discrete_pipeline, discrete_cat),
    ]
)

# %%
# === SPLIT DATASET ===
print("Splitting data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# %%
# === KNN CLASSIFIER ===
print("Training KNN classifier...")

knn_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", KNeighborsClassifier())]
)
knn_grid = {
    "classifier__n_neighbors": list(range(3, 21, 2)),
    "classifier__weights": ["uniform", "distance"],
    "classifier__metric": ["euclidean", "manhattan", "minkowski"],
}
knn_grid_search = GridSearchCV(
    knn_pipeline, knn_grid, scoring="f1_macro", cv=10, verbose=1, n_jobs=-1
)
knn_grid_search.fit(X_train, y_train.values.ravel())

best_knn_model = knn_grid_search.best_estimator_
print("KNN training complete. Best parameters:", knn_grid_search.best_params_)

cv_scores = cross_val_score(
    best_knn_model, X_test, y_test.values.ravel(), cv=10, scoring="f1_macro", n_jobs=-1
)
print(f"KNN Mean F1 score: {cv_scores.mean()}")

# Confusion matrix and report
inverse_mapping = {v: k for k, v in obesity_level_mapping.items()}
y_test_str = y_test["obesity_level"].map(inverse_mapping)
y_pred = best_knn_model.predict(X_test)
y_pred_str = pd.Series(y_pred).map(inverse_mapping)

y_labels = list(obesity_level_mapping.keys())
cm = confusion_matrix(y_test_str, y_pred_str, labels=y_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_labels)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("KNN - Confusion Matrix")
plt.savefig("results/knn/confusion_matrix_knn.png", dpi=300, bbox_inches="tight")

with open("results/knn/classification_report_knn.txt", "w") as f:
    f.write(str(classification_report(y_test_str, y_pred_str)))

# SHAP explainability
print("Generating SHAP explanation for KNN...")
X_test_transformed = best_knn_model.named_steps["preprocessor"].transform(X_test)
model_knn = best_knn_model.named_steps["classifier"]
X_sample = shap.sample(X_test_transformed, 100, random_state=42)
explainer = shap.KernelExplainer(model_knn.predict_proba, X_sample)
shap_values = explainer.shap_values(X_sample)
feature_names = best_knn_model.named_steps["preprocessor"].get_feature_names_out()
plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_sample,
    plot_type="bar",
    feature_names=feature_names,
    class_names=y_labels,
    show=False,
)
plt.title("SHAP - KNN")
plt.tight_layout()
plt.savefig("results/knn/shap_summary_knn.png", dpi=300, bbox_inches="tight")

# %%
# === DECISION TREE CLASSIFIER ===
print("Training Decision Tree classifier...")

dt_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", DecisionTreeClassifier())]
)
dt_grid = {
    "classifier__criterion": ["gini", "entropy", "log_loss"],
    "classifier__splitter": ["best"],
    "classifier__max_depth": [None, 3, 5, 7, 9, 15],
    "classifier__min_samples_split": [2, 5, 10],
}
dt_grid_search = GridSearchCV(
    dt_pipeline, dt_grid, scoring="f1_macro", cv=10, verbose=1, n_jobs=-1
)
dt_grid_search.fit(X_train, y_train)
best_dt_model = dt_grid_search.best_estimator_
print("Decision Tree training complete. Best parameters:", dt_grid_search.best_params_)

cv_scores = cross_val_score(
    best_dt_model, X_test, y_test.values.ravel(), cv=10, scoring="f1_macro", n_jobs=-1
)
print(f"Decision Tree Mean F1 score: {cv_scores.mean()}")

# Evaluation
y_pred_dt = best_dt_model.predict(X_test)
y_pred_dt_str = pd.Series(y_pred_dt).map(inverse_mapping)

cm_dt = confusion_matrix(y_test_str, y_pred_dt_str, labels=y_labels)
disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt, display_labels=y_labels)
disp_dt.plot(cmap="Oranges", xticks_rotation=45)
plt.title("Decision Tree - Confusion Matrix")
plt.savefig(
    "results/decision_tree/confusion_matrix_decision_tree.png",
    dpi=300,
    bbox_inches="tight",
)

with open("results/decision_tree/classification_report_decision_tree.txt", "w") as f:
    f.write(str(classification_report(y_test_str, y_pred_dt_str)))

# SHAP for Decision Tree
print("Generating SHAP explanation for Decision Tree...")
X_test_transformed = best_dt_model.named_steps["preprocessor"].transform(X_test)
model_dt = best_dt_model.named_steps["classifier"]
explainer = shap.TreeExplainer(model_dt)
shap_values = explainer.shap_values(X_test_transformed)

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test_transformed,
    plot_type="bar",
    feature_names=feature_names,
    class_names=y_labels,
    show=False,
)
plt.title("SHAP - Decision Tree")
plt.tight_layout()
plt.savefig(
    "results/decision_tree/shap_summary_decision_tree_bar.png",
    dpi=300,
    bbox_inches="tight",
)

# %%
# === NAIVE BAYES CLASSIFIER ===
print("Training Naive Bayes classifier...")

nb_pipeline = Pipeline(
    steps=[("preprocessor", preprocessor), ("classifier", GaussianNB())]
)
nb_grid = {
    "classifier__var_smoothing": [0.001, 0.01, 0.1, 1.0, 10.0],
}
nb_grid_search = GridSearchCV(
    nb_pipeline, nb_grid, scoring="f1_macro", cv=10, verbose=1, n_jobs=-1
)
nb_grid_search.fit(X_train, y_train.values.ravel())
best_nb_model = nb_grid_search.best_estimator_
print("Naive Bayes training complete. Best parameters:", nb_grid_search.best_params_)

cv_scores = cross_val_score(
    best_nb_model, X_test, y_test.values.ravel(), cv=10, scoring="f1_macro", n_jobs=-1
)
print(f"Naive Bayes Mean F1 score: {cv_scores.mean()}")


# Evaluation
y_pred_nb = best_nb_model.predict(X_test)
y_pred_nb_str = pd.Series(y_pred_nb).map(inverse_mapping)

cm_nb = confusion_matrix(y_test_str, y_pred_nb_str, labels=y_labels)
disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb, display_labels=y_labels)
disp_nb.plot(cmap="Greens", xticks_rotation=45)
plt.title("Naive Bayes - Confusion Matrix")
plt.savefig(
    "results/naive_bayes/confusion_matrix_naive_bayes.png", dpi=300, bbox_inches="tight"
)

with open("results/naive_bayes/classification_report_naive_bayes.txt", "w") as f:
    f.write(str(classification_report(y_test_str, y_pred_nb_str)))

# SHAP for Naive Bayes
print("Generating SHAP explanation for Naive Bayes...")
X_test_transformed = best_nb_model.named_steps["preprocessor"].transform(X_test)
explainer = shap.KernelExplainer(
    best_nb_model.named_steps["classifier"].predict_proba, X_test_transformed[:100]
)
shap_values = explainer.shap_values(X_test_transformed[:100])

plt.figure(figsize=(12, 8))
shap.summary_plot(
    shap_values,
    X_test_transformed[:100],
    plot_type="bar",
    feature_names=feature_names,
    class_names=y_labels,
    show=False,
)
plt.title("SHAP - Naive Bayes")
plt.tight_layout()
plt.savefig(
    "results/naive_bayes/shap_summary_naive_bayes_bar.png", dpi=300, bbox_inches="tight"
)

print("Script complete ✅")
# %%
