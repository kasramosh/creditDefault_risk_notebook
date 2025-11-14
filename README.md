# Credit Risk Prediction Project

A comprehensive machine learning project for predicting credit card defaults using various classification algorithms.

## Overview

This project applies machine learning techniques to predict credit card defaults using the [Default of Credit Card Clients Dataset](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset). The goal is to build models that can identify customers at risk of defaulting on their credit card payments, which is crucial for financial institutions to manage risk effectively.

## Dataset

The dataset contains 30,000 examples with 24 features, including:
- **Demographic features**: Age, sex, education, marriage status
- **Credit information**: Credit limit (LIMIT_BAL)
- **Payment history**: Repayment status for the past 6 months (PAY_0 to PAY_6)
- **Bill amounts**: Bill statement amounts for the past 6 months (BILL_AMT1 to BILL_AMT6)
- **Payment amounts**: Amount of previous payments (PAY_AMT1 to PAY_AMT6)

## Project Structure

The project follows a complete machine learning workflow:

1. **Data Understanding**: Exploratory data analysis to understand the dataset structure and identify patterns
2. **Feature Engineering**: Creation of new features such as aggregated payment delays and utilization ratios
3. **Model Development**: Experimentation with multiple algorithms:
   - Logistic Regression
   - XGBoost
   - Support Vector Machines (SVC)
   - Decision Trees
4. **Model Optimization**: Hyperparameter tuning using RandomizedSearchCV
5. **Model Interpretation**: SHAP values analysis to understand feature importance
6. **Evaluation**: Performance evaluation on held-out test set

## Key Results

- **Best Model**: XGBoost
- **Final Test F1 Score**: 0.4662
- **Evaluation Metric**: F1 Score (chosen due to class imbalance)

The dataset exhibits class imbalance (approximately 78% non-defaults, 22% defaults), making F1 score an appropriate metric that balances precision and recall.

## Technologies Used

- Python
- pandas
- scikit-learn
- XGBoost
- SHAP (for model interpretation)
- matplotlib & seaborn (for visualization)

## Getting Started

1. Clone the repository
2. Install required dependencies:
   ```bash
   pip install pandas scikit-learn xgboost shap matplotlib seaborn numpy
   ```
3. Download the dataset from [Kaggle](https://www.kaggle.com/uciml/default-of-credit-card-clients-dataset) and place it in the `data/` directory
4. Open and run the Jupyter notebook `hw5.ipynb`

## Features

- Comprehensive exploratory data analysis
- Feature engineering with domain knowledge
- Multiple model comparison
- Hyperparameter optimization
- Model interpretation using SHAP values
- Detailed performance evaluation and analysis

## Future Improvements

Potential areas for improvement:
- Handling class imbalance with techniques like SMOTE or class weighting
- Exploring additional models (LightGBM, CatBoost, Neural Networks)
- Advanced feature engineering
- Ensemble methods (stacking, blending)
- Bayesian optimization for hyperparameter tuning

## License

This project is for educational and portfolio purposes.
