# Principal Component Analysis (PCA) – Breast Cancer Dataset

## Project Overview
This project demonstrates the use of **Principal Component Analysis (PCA)** to reduce the dimensionality of a high-dimensional medical dataset while preserving essential information. The analysis uses the **Breast Cancer Wisconsin dataset** available from `sklearn.datasets`.

In addition, a **Logistic Regression model** is implemented on the reduced dataset to evaluate predictive performance.

---

## Objectives
- Load and preprocess the breast cancer dataset
- Standardize features to ensure equal contribution
- Apply PCA to reduce the dataset to **2 principal components**
- Visualize the PCA results
- Apply **Logistic Regression** on the PCA-transformed data for prediction

---

## Dataset
- **Source:** `sklearn.datasets.load_breast_cancer`
- **Features:** 30 numeric medical attributes
- **Target Classes:**
  - `0` → Malignant
  - `1` → Benign

---

## Methodology

### 1. Data Preprocessing
- The dataset is standardized using `StandardScaler` to normalize feature values.
- Standardization is required for PCA to function correctly.

### 2. Principal Component Analysis (PCA)
- PCA is applied to reduce the dataset to **2 principal components**.
- The explained variance ratio is printed to show how much information is retained.
- Total variance explained is approximately **63%**, indicating effective dimensionality reduction.

### 3. Visualization
- A 2D scatter plot visualizes the two principal components.
- Points are colored based on cancer diagnosis (Malignant vs Benign).

### 4. Logistic Regression
- Logistic regression is trained using the PCA-transformed features.
- Model performance is evaluated using:
  - Accuracy score
  - Precision, recall, and F1-score (classification report)

---

## How to Run the Program

### 1. Install Required Libraries
```bash
pip install pandas matplotlib scikit-learn 
```
## Output
- Explained variance ratio of PCA components
- PCA visualization plot
- Logistic regression accuracy score
- Logistic regression classification report (precision, recall, F1-score)

---

## Technologies Used
- Python 3.x
- pandas
- matplotlib
- scikit-learn

---

## Conclusion
This project demonstrates that Principal Component Analysis (PCA) can
significantly reduce dataset dimensionality while preserving essential
information. The high accuracy achieved by the logistic regression model
confirms that the PCA-reduced feature space remains effective for
classification tasks, highlighting PCA as a valuable technique in
medical data analysis.




