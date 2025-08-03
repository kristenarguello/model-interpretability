# 🧠 Interpretable Machine Learning Models

This project explores the interpretability of classic machine learning models by training and analyzing three well-known classifiers: **K-Nearest Neighbors (KNN)**, **Naïve Bayes**, and **Decision Trees**. It was developed as part of an academic assignment for the Machine Learning course at PUCRS.

## 📌 Objective

Interpretability is a key aspect of trustworthy and responsible machine learning — especially in sensitive domains like healthcare and law. The goal of this work was to:

- Train and evaluate different models on a real-world classification task.
- Explore and compare interpretability techniques tailored to each model.
- Use tools such as SHAP and LIME to explain model predictions.

## 📊 Dataset

We used a public classification dataset (not among the common ones like Iris or Titanic) containing:
- At least 5 features (both categorical and numerical)
- A categorical target variable
- Over 100 instances

> The dataset was preprocessed to handle missing values, normalize features, and encode categorical variables. An 80/20 train-test split was used.

## ⚙️ Models Trained

- **K-Nearest Neighbors (KNN)**
- **Naïve Bayes**
- **Decision Tree**

Each model was evaluated using:
- Accuracy
- Precision
- Recall
- F1-score

## 🔍 Interpretability Techniques

We applied different interpretability methods tailored to each model:

- **Decision Tree**: Visual inspection and feature importance analysis.
- **Naïve Bayes**: Interpretation based on conditional probability distributions.
- **KNN**: Local interpretability using **SHAP** and **LIME**, since KNN is inherently harder to interpret.

Additionally, model-agnostic methods like **Permutation Feature Importance** and SHAP summary plots were used across models to gain global insights.

## 📈 Results & Discussion

- Comparative analysis was done between models in terms of **interpretability**, **performance**, and **limitations**.
- Special focus was given to explaining how each model arrives at its predictions.
- Reflections were made on the trade-off between accuracy and explainability in practical scenarios.

## 🎥 Presentation

A recorded presentation (10–15 min) summarizing the project and findings is available [here](#). *(Insert YouTube or video link in a real submission)*

## 🛠️ Tools & Libraries

- Python
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- SHAP
- LIME

## 🧾 Structure
