"""
Principal Component Analysis (PCA)
Dataset: Breast Cancer Wisconsin Dataset (sklearn)
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


# Load and prepare the cancer dataset
def load_and_prepare_data():
    cancer = load_breast_cancer()
    X = cancer.data
    y = cancer.target

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, cancer


# Apply PCA to reduce dimensions
def apply_pca(X_scaled, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)

    print("Explained Variance Ratio:", pca.explained_variance_ratio_)
    print("Total Explained Variance:", pca.explained_variance_ratio_.sum())

    return X_pca

# Visualize PCA results
def plot_pca(X_pca, y):
    plt.figure(figsize=(8, 6))
    plt.scatter(X_pca[y == 0, 0], X_pca[y == 0, 1],
                label="Malignant", alpha=0.7)
    plt.scatter(X_pca[y == 1, 0], X_pca[y == 1, 1],
                label="Benign", alpha=0.7)

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("PCA of Breast Cancer Dataset")
    plt.legend()
    plt.grid(True)
    plt.show()


# Logistic Regression on PCA components
def logistic_regression_model(X_pca, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X_pca, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    print("\nLogistic Regression Results")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("\nClassification Report:\n",
          classification_report(y_test, y_pred))


def main():
    X_scaled, y, cancer = load_and_prepare_data()
    X_pca = apply_pca(X_scaled, n_components=2)
    plot_pca(X_pca, y)
    logistic_regression_model(X_pca, y)


if __name__ == "__main__":
    main()
