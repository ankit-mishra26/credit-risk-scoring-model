"""
Credit Risk Scoring Model using Machine Learning
================================================

This program builds a comprehensive credit risk scoring model using multiple ML algorithms.
It includes data preprocessing, model training, evaluation, and a credit scoring system.

Author: AI Assistant
Date: August 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve)
import warnings
warnings.filterwarnings('ignore')

# For XGBoost - install with: pip install xgboost
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not available. Using GradientBoostingClassifier instead.")

import joblib
import os

class CreditRiskModel:
    """
    Credit Risk Scoring Model class that handles the entire ML pipeline
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        self.results = {}
        
    def load_sample_data(self):
        """
        Create sample credit risk dataset for demonstration
        In practice, load from: 
        - German Credit Risk dataset
        - Kaggle's "Give Me Some Credit" dataset
        """
        np.random.seed(42)
        n_samples = 2000
        
        # Generate synthetic credit data
        data = {
            'Age': np.random.randint(18, 80, n_samples),
            'Income': np.random.normal(50000, 20000, n_samples),
            'LoanAmount': np.random.normal(15000, 8000, n_samples),
            'CreditHistory': np.random.randint(0, 60, n_samples),  # months
            'Employment': np.random.choice(['Employed', 'Self-Employed', 'Unemployed'], n_samples, p=[0.7, 0.2, 0.1]),
            'Housing': np.random.choice(['Own', 'Rent', 'Mortgage'], n_samples, p=[0.4, 0.3, 0.3]),
            'LoanPurpose': np.random.choice(['Auto', 'Home', 'Personal', 'Business'], n_samples, p=[0.3, 0.25, 0.3, 0.15]),
            'DebtToIncome': np.random.uniform(0.1, 0.8, n_samples),
            'NumCreditLines': np.random.randint(1, 10, n_samples),
            'MonthlyPayment': np.random.normal(500, 200, n_samples)
        }
        
        # Ensure positive values for certain features
        data['Income'] = np.abs(data['Income'])
        data['LoanAmount'] = np.abs(data['LoanAmount'])
        data['MonthlyPayment'] = np.abs(data['MonthlyPayment'])
        
        df = pd.DataFrame(data)
        
        # Create target variable based on risk factors
        risk_score = (
            (df['Age'] < 25).astype(int) * 0.2 +
            (df['Income'] < 30000).astype(int) * 0.3 +
            (df['DebtToIncome'] > 0.5).astype(int) * 0.4 +
            (df['CreditHistory'] < 12).astype(int) * 0.3 +
            (df['Employment'] == 'Unemployed').astype(int) * 0.5 +
            np.random.normal(0, 0.1, len(df))
        )
        
        # Convert to binary classification (1 = Bad Credit, 0 = Good Credit)
        df['CreditRisk'] = (risk_score > 0.4).astype(int)
        
        # Add some missing values for demonstration
        missing_indices = np.random.choice(df.index, size=int(0.05 * len(df)), replace=False)
        df.loc[missing_indices, 'Income'] = np.nan
        
        return df
    
    def preprocess_data(self, df, is_training=True):
        """
        Preprocess the credit risk dataset
        """
        print("Preprocessing data...")
        
        # Handle missing values
        # Numerical features - impute with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = numerical_cols.drop('CreditRisk', errors='ignore')
        
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical features - impute with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        # Encode categorical variables
        if is_training:
            # Fit label encoders on training data
            for col in categorical_cols:
                le = LabelEncoder()
                df[col + '_encoded'] = le.fit_transform(df[col])
                self.label_encoders[col] = le
        else:
            # Transform using existing encoders
            for col in categorical_cols:
                if col in self.label_encoders:
                    df[col + '_encoded'] = self.label_encoders[col].transform(df[col])
        
        # Create dummy variables for categorical features
        categorical_encoded_cols = [col + '_encoded' for col in categorical_cols]
        df_encoded = pd.get_dummies(df, columns=categorical_encoded_cols, prefix=categorical_cols)
        
        # Remove original categorical columns
        df_encoded = df_encoded.drop(columns=categorical_cols)
        
        # Separate features and target
        if 'CreditRisk' in df_encoded.columns:
            X = df_encoded.drop('CreditRisk', axis=1)
            y = df_encoded['CreditRisk']
        else:
            X = df_encoded
            y = None
        
        # Scale numerical features
        if is_training:
            X_scaled = self.scaler.fit_transform(X)
            self.feature_names = X.columns.tolist()
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.15, val_size=0.15):
        """
        Split data into train, validation, and test sets
        """
        print("Splitting data...")
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Second split: separate train and validation
        val_size_adjusted = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=42, stratify=y_temp
        )
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, y_train, X_val, y_val):
        """
        Train multiple models with hyperparameter tuning
        """
        print("Training models...")
        
        # Model configurations
        models_config = {
            'LogisticRegression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1, 10, 100],
                    'penalty': ['l1', 'l2'],
                    'solver': ['liblinear']
                }
            },
            'RandomForest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                }
            },
            'GradientBoosting': {
                'model': GradientBoostingClassifier(random_state=42),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        }
        
        # Add XGBoost if available
        if XGBOOST_AVAILABLE:
            models_config['XGBoost'] = {
                'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                'params': {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'max_depth': [3, 5, 7]
                }
            }
        
        # Train models with hyperparameter tuning
        for name, config in models_config.items():
            print(f"Training {name}...")
            
            # Use RandomizedSearchCV for faster training
            search = RandomizedSearchCV(
                config['model'], 
                config['params'], 
                n_iter=20, 
                cv=5, 
                scoring='roc_auc', 
                random_state=42,
                n_jobs=-1
            )
            
            search.fit(X_train, y_train)
            self.models[name] = search.best_estimator_
            
            print(f"{name} - Best score: {search.best_score_:.4f}")
    
    def evaluate_models(self, X_val, y_val, X_test, y_test):
        """
        Evaluate all trained models
        """
        print("Evaluating models...")
        
        best_auc = 0
        
        for name, model in self.models.items():
            # Predictions
            y_val_pred = model.predict(X_val)
            y_val_prob = model.predict_proba(X_val)[:, 1]
            y_test_pred = model.predict(X_test)
            y_test_prob = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            val_metrics = {
                'accuracy': accuracy_score(y_val, y_val_pred),
                'precision': precision_score(y_val, y_val_pred),
                'recall': recall_score(y_val, y_val_pred),
                'f1': f1_score(y_val, y_val_pred),
                'auc': roc_auc_score(y_val, y_val_prob)
            }
            
            test_metrics = {
                'accuracy': accuracy_score(y_test, y_test_pred),
                'precision': precision_score(y_test, y_test_pred),
                'recall': recall_score(y_test, y_test_pred),
                'f1': f1_score(y_test, y_test_pred),
                'auc': roc_auc_score(y_test, y_test_prob)
            }
            
            self.results[name] = {
                'validation': val_metrics,
                'test': test_metrics,
                'val_prob': y_val_prob,
                'test_prob': y_test_prob
            }
            
            # Track best model based on validation AUC
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                self.best_model = model
                self.best_model_name = name
            
            print(f"\n{name} Results:")
            print(f"Validation - AUC: {val_metrics['auc']:.4f}, Accuracy: {val_metrics['accuracy']:.4f}")
            print(f"Test - AUC: {test_metrics['auc']:.4f}, Accuracy: {test_metrics['accuracy']:.4f}")
        
        print(f"\nBest model: {self.best_model_name} (Validation AUC: {best_auc:.4f})")
    
    def plot_curves(self, y_val, y_test):
        """
        Plot ROC and Precision-Recall curves
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # ROC Curves
        axes[0, 0].set_title('ROC Curves - Validation Set')
        axes[0, 1].set_title('ROC Curves - Test Set')
        axes[1, 0].set_title('Precision-Recall Curves - Validation Set')
        axes[1, 1].set_title('Precision-Recall Curves - Test Set')
        
        for name in self.models.keys():
            val_prob = self.results[name]['val_prob']
            test_prob = self.results[name]['test_prob']
            
            # ROC curves
            fpr_val, tpr_val, _ = roc_curve(y_val, val_prob)
            fpr_test, tpr_test, _ = roc_curve(y_test, test_prob)
            
            axes[0, 0].plot(fpr_val, tpr_val, label=f'{name} (AUC={self.results[name]["validation"]["auc"]:.3f})')
            axes[0, 1].plot(fpr_test, tpr_test, label=f'{name} (AUC={self.results[name]["test"]["auc"]:.3f})')
            
            # Precision-Recall curves
            prec_val, rec_val, _ = precision_recall_curve(y_val, val_prob)
            prec_test, rec_test, _ = precision_recall_curve(y_test, test_prob)
            
            axes[1, 0].plot(rec_val, prec_val, label=name)
            axes[1, 1].plot(rec_test, prec_test, label=name)
        
        # Format plots
        for i in range(2):
            for j in range(2):
                axes[i, j].legend()
                axes[i, j].grid(True)
        
        # ROC diagonal line
        axes[0, 0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
        axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.6)
        
        plt.tight_layout()
        plt.show()
    
    def probability_to_credit_score(self, probability):
        """
        Convert predicted probability to credit score (300-850 scale)
        """
        # Invert probability (lower probability of default = higher credit score)
        inverted_prob = 1 - probability
        
        # Scale to 300-850 range
        credit_score = 300 + (inverted_prob * 550)
        
        return np.round(credit_score).astype(int)
    
    def get_risk_category(self, credit_score):
        """
        Categorize credit score into risk levels
        """
        if credit_score >= 740:
            return "Very Low Risk"
        elif credit_score >= 670:
            return "Low Risk"
        elif credit_score >= 580:
            return "Medium Risk"
        else:
            return "High Risk"
    
    def predict_customer_risk(self, customer_data):
        """
        Predict credit risk for a new customer
        """
        if self.best_model is None:
            raise ValueError("No trained model available. Please train the model first.")
        
        # Create DataFrame
        customer_df = pd.DataFrame([customer_data])
        
        # Preprocess
        X_customer, _ = self.preprocess_data(customer_df, is_training=False)
        
        # Predict
        probability = self.best_model.predict_proba(X_customer)[0, 1]
        credit_score = self.probability_to_credit_score(probability)
        risk_category = self.get_risk_category(credit_score)
        
        return {
            'probability_of_default': probability,
            'credit_score': credit_score[0] if isinstance(credit_score, np.ndarray) else credit_score,
            'risk_category': risk_category
        }
    
    def save_model(self, filename='credit_risk_model.pkl'):
        """
        Save the trained model and preprocessors
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filename)
        print(f"Model saved as {filename}")
    
    def load_model(self, filename='credit_risk_model.pkl'):
        """
        Load a pre-trained model
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"Model file {filename} not found.")
        
        model_data = joblib.load(filename)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.label_encoders = model_data['label_encoders']
        self.feature_names = model_data['feature_names']
        
        print(f"Model loaded from {filename}")


def main():
    """
    Main function to demonstrate the Credit Risk Scoring Model
    """
    print("=" * 60)
    print("CREDIT RISK SCORING MODEL")
    print("=" * 60)
    
    # Initialize model
    credit_model = CreditRiskModel()
    
    # Load data (in practice, load from file)
    print("Loading dataset...")
    df = credit_model.load_sample_data()
    print(f"Dataset shape: {df.shape}")
    print(f"Credit Risk distribution:")
    print(df['CreditRisk'].value_counts())
    
    # Preprocess data
    X, y = credit_model.preprocess_data(df, is_training=True)
    
    # Split data
    X_train, X_val, X_test, y_train, y_val, y_test = credit_model.split_data(X, y)
    
    # Train models
    credit_model.train_models(X_train, y_train, X_val, y_val)
    
    # Evaluate models
    credit_model.evaluate_models(X_val, y_val, X_test, y_test)
    
    # Plot evaluation curves
    credit_model.plot_curves(y_val, y_test)
    
    # Save best model
    credit_model.save_model('credit_risk_model.pkl')
    
    # Demonstrate prediction for new customer
    print("\n" + "=" * 60)
    print("CUSTOMER RISK PREDICTION DEMO")
    print("=" * 60)
    
    # Sample customer data
    new_customer = {
        'Age': 35,
        'Income': 65000,
        'LoanAmount': 20000,
        'CreditHistory': 36,
        'Employment': 'Employed',
        'Housing': 'Own',
        'LoanPurpose': 'Auto',
        'DebtToIncome': 0.3,
        'NumCreditLines': 4,
        'MonthlyPayment': 450
    }
    
    print("Customer Profile:")
    for key, value in new_customer.items():
        print(f"  {key}: {value}")
    
    # Predict risk
    result = credit_model.predict_customer_risk(new_customer)
    
    print(f"\nCredit Risk Assessment:")
    print(f"  Probability of Default: {result['probability_of_default']:.3f}")
    print(f"  Credit Score: {result['credit_score']}")
    print(f"  Risk Category: {result['risk_category']}")
    
    # Interactive prediction function
    def interactive_prediction():
        """
        Allow user to input customer data for prediction
        """
        print("\n" + "=" * 60)
        print("INTERACTIVE CUSTOMER ASSESSMENT")
        print("=" * 60)
        
        try:
            customer_input = {}
            customer_input['Age'] = int(input("Enter customer age: "))
            customer_input['Income'] = float(input("Enter annual income ($): "))
            customer_input['LoanAmount'] = float(input("Enter loan amount ($): "))
            customer_input['CreditHistory'] = int(input("Enter credit history (months): "))
            customer_input['Employment'] = input("Enter employment status (Employed/Self-Employed/Unemployed): ")
            customer_input['Housing'] = input("Enter housing status (Own/Rent/Mortgage): ")
            customer_input['LoanPurpose'] = input("Enter loan purpose (Auto/Home/Personal/Business): ")
            customer_input['DebtToIncome'] = float(input("Enter debt-to-income ratio (0.0-1.0): "))
            customer_input['NumCreditLines'] = int(input("Enter number of credit lines: "))
            customer_input['MonthlyPayment'] = float(input("Enter monthly payment capacity ($): "))
            
            # Predict
            result = credit_model.predict_customer_risk(customer_input)
            
            print(f"\nCredit Risk Assessment:")
            print(f"  Probability of Default: {result['probability_of_default']:.3f}")
            print(f"  Credit Score: {result['credit_score']}")
            print(f"  Risk Category: {result['risk_category']}")
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
    
    # Uncomment the line below to enable interactive mode
    # interactive_prediction()


if __name__ == "__main__":
    main()