# CreditWise
# 🏦 CreditWise: Loan Approval Predictor

CreditWise is an interactive web application built with **Streamlit** that predicts the likelihood of loan approval based on applicant data. It uses a Machine Learning backend trained on historical loan data to provide real-time predictions.

## 🚀 Live Demo
Check out the live app here: (https://creditwise-wqwvctemzjxchyvqpuqfwf.streamlit.app/)

## 🛠️ Features
- **Real-time Prediction:** Get instant feedback on loan eligibility.
- **Interactive UI:** Input fields for income, credit score, debt-to-income ratio, and more.
- **Data-Driven Logic:** Uses Scikit-Learn for data preprocessing (imputation, encoding, and scaling) and classification.

- ## 📊 Model Performance & Evaluation

After experimenting with multiple algorithms, the **Logistic Regression** model was chosen for the final application due to its high balanced performance.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | **88.0%** | **78.5%** | **83.6%** | **81.0%** |
| Naive Bayes | 86.0% | 81.1% | 70.5% | 75.4% |
| KNN (n=15) | 77.5% | 70.0% | 45.9% | 55.4% |

### Key Technical Pointers:
- **Handling Missing Values:** Used `SimpleImputer` with a "mean" strategy for numerical gaps (like Income) and "most_frequent" for categorical gaps (like Gender).
- **Feature Engineering:** Implemented `OneHotEncoder` for multi-category features and `LabelEncoder` for binary/ordinal features to ensure the model could process the non-numeric data.
- **Scaling:** Applied `StandardScaler` to normalize feature ranges, which was critical for the performance of the Logistic Regression model.

## 📁 Project Structure
- `app.py`: The main Streamlit application script containing both the ML logic and the frontend.
- `loan_approval_data.csv`: The dataset used to train the model.
- `requirements.txt`: List of dependencies required to run the app.
- `credit_wise.ipynb`: The original exploratory data analysis and model training notebook.


📊 Model Information
The model currently uses a Logistic Regression classifier (which you can update to KNN or Naive Bayes based on your notebook experiments).

Preprocessing: Handles missing values using Mean/Most-Frequent imputation.

Scaling: Uses StandardScaler to ensure all numerical features contribute equally.

