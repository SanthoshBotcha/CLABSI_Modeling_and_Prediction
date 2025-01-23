# CLABSI Prediction Project

## Overview
This project focuses on predicting **Central Line-Associated Bloodstream Infections (CLABSI)** using machine learning models. The goal was to analyze patterns in clinical data, address challenges like class imbalance, and develop robust predictive models to support healthcare decision-making.

---

## Key Features

### 1. **Data Preprocessing**
- **Imputation**: Handled missing values using mean and median imputation.
- **Scaling**: Standardized numerical features to enhance model performance.
- **Feature Selection**: Applied `SelectKBest` to identify top predictors.
- **Encoding**: Used one-hot encoding for categorical variables.

### 2. **Handling Class Imbalance**
- Implemented **SMOTE (Synthetic Minority Oversampling Technique)** to balance the dataset.
- Evaluated performance improvements with balanced training data.

### 3. **Model Development**
Developed and compared the following models:
- **Logistic Regression**: Baseline model, optimized with hyperparameter tuning.
- **K-Nearest Neighbors (KNN)**: Improved performance with SMOTE and hyperparameter tuning.
- **Decision Trees**: Enhanced with bootstrapping to improve reliability.
- **Neural Networks**: Achieved high recall for detecting CLABSI cases.
- **XGBoost**: Applied hyperparameter tuning but struggled with class imbalance.

### 4. **Evaluation Metrics**
- Used metrics such as **accuracy**, **precision**, **recall**, **F1-score**, and **AUC** to assess model performance.
- Prioritized **recall** to minimize false negatives and avoid missed diagnoses.

---

## Results
| Model               | Accuracy  | FPR (Type I) | FNR (Type II) | TPR    | TNR    | AUC   |
|---------------------|-----------|--------------|---------------|--------|--------|-------|
| KNN                 | 95.54%    | 4.4%         | 8.3%          | 91.7%  | 95.6%  | 0.94  |
| Decision Trees      | 99.98%    | 0%           | 0.1%          | 100%   | 99.9%  | 0.99  |
| XGBoost             | 97.27%    | 2.4%         | 90%           | 10%    | 97.6%  | 0.66  |
| Neural Networks     | 97.55%    | 4.87%        | 0%            | 100%   | 95.1%  | 0.97  |

---

## Technologies Used
- **Programming Language**: Python
- **Libraries**:
  - Data Preprocessing: Pandas, NumPy, Scikit-learn
  - Oversampling: Imbalanced-learn (SMOTE)
  - Machine Learning: TensorFlow, Keras, XGBoost
  - Visualization: Matplotlib, Seaborn

---

## How to Run
1. Clone the repository:
   ```bash
   git clone <[repository-link](https://github.com/SanthoshBotcha/CLABSI_Modeling_and_Prediction.git)>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook clabsi_prediction.ipynb
   ```

---

## Key Learnings
- **Addressing Class Imbalance**: SMOTE significantly improved model sensitivity for minority class detection.
- **Model Selection**: Neural Networks and KNN (with SMOTE) performed best for this dataset.
- **Evaluation Trade-offs**: Prioritized recall to reduce false negatives, critical in medical diagnostics.

---

## Next Steps
- Use a holdout dataset to validate generalization performance.
- Experiment with cost-sensitive learning to reduce false positives.
- Incorporate explainability tools like SHAP or LIME.

---

## Author
**Santhosh Botcha**  
Academic - Business Analysis Project
