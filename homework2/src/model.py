"""
Model stencil for Homework 2: Ensemble Methods with Gradient Boosting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Optional, Union

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import plot_tree

# Set plotting style
sns.set_style("whitegrid")


class GradientBoostingModel:
    """Gradient Boosting model implementation with comprehensive evaluation and analysis tools"""

    def __init__(
        self,
        task: str = "classification",
        max_depth: int = 3,
        learning_rate: float = 0.1,
        n_estimators: int = 50,
        subsample: float = 1.0,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[int] = None,
        random_state: int = 42,
        use_scaler: bool = False,
    ):
        """
        Initialize Gradient Boosting model with customizable parameters

        Args:
            task: 'classification' or 'regression'
            max_depth: Maximum depth of a tree (controls pruning)
            learning_rate: Step size shrinkage to prevent overfitting
            n_estimators: Number of boosting rounds/trees
            subsample: Subsample ratio of training instances
            min_samples_split: Minimum samples required to split an internal node
            min_samples_leaf: Minimum samples required to be at a leaf node
            max_features: Number of features to consider when looking for the best split
            random_state: Random seed for reproducibility
            use_scaler: Whether to apply StandardScaler before training/prediction
        """
        self.params = {
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "n_estimators": n_estimators,
            "subsample": subsample,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
            "max_features": max_features,
            "random_state": random_state,
        }

        if task not in {"classification", "regression"}:
            raise ValueError("task must be 'classification' or 'regression'.")

        self.model = None
        self.feature_names = None
        self.task = task
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None

    def train_test_split(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        Split data into training and testing sets

        Args:
            X: Feature DataFrame
            y: Target Series
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            X_train, X_test, y_train, y_test: Split datasets
        """
        # TODO: Implement train/test split and track feature names
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        return X_train, X_test, y_train, y_test

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, verbose: bool = True):
        """
        Train the Gradient Boosting model

        Args:
            X_train: Training features
            y_train: Training targets
            verbose: Whether to print training progress

        Returns:
            self: Trained model instance
        """
        # TODO: Create classifier/regressor based on task and fit it
        
        if self.use_scaler:
            X_train = self.scaler.fit_transform(X_train)
        
        if self.task == "classification":
            self.model = GradientBoostingClassifier(max_depth=self.params["max_depth"], learning_rate=self.params["learning_rate"], n_estimators=self.params["n_estimators"], subsample=self.params["subsample"], min_samples_split=self.params["min_samples_split"], min_samples_leaf=self.params["min_samples_leaf"], max_features=self.params["max_features"], random_state=self.params["random_state"], verbose=int(verbose))
            self.model.fit(X_train, y_train)
        else:
            self.model = GradientBoostingRegressor(max_depth=self.params["max_depth"], learning_rate=self.params["learning_rate"], n_estimators=self.params["n_estimators"], subsample=self.params["subsample"], min_samples_split=self.params["min_samples_split"], min_samples_leaf=self.params["min_samples_leaf"], max_features=self.params["max_features"], random_state=self.params["random_state"], verbose=int(verbose))
            self.model.fit(X_train, y_train)

    def predict(
        self, X: pd.DataFrame, return_proba: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """
        Make predictions with the trained model

        Args:
            X: Feature data for prediction
            return_proba: If True and model is a classifier, return probability estimates

        Returns:
            Predictions or probability estimates
        """
        # TODO: Apply scaler when enabled, then predict
        if self.use_scaler:
            X = self.scaler.transform(X)
        
        if return_proba and self.task == "classification":
            prediction = self.model.predict_proba(X)
        else:
            prediction = self.model.predict(X)
        
        return prediction


    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance on test data

        Args:
            X_test: Test features
            y_test: Test targets

        Returns:
            Dictionary of evaluation metrics
        """

        # TODO: Compute metrics (classification vs regression)
        prediction = self.predict(X_test)
        probabilities = self.predict(X_test, return_proba=True)
        print(f"Prediction is {prediction}")
        if self.task == "classification":
            metrics = {
                "accuracy": accuracy_score(y_test, prediction),
                "precision": precision_score(y_test, prediction, average="macro"),
                "recall": recall_score(y_test, prediction, average="macro"),
                "f1": f1_score(y_test, prediction, average="macro"),
                "roc_auc": roc_auc_score(y_test, probabilities[:, 1], average="macro", multi_class="ovo"),
                "prediction": prediction,
                "probabilities": probabilities
            }
        else:
            metrics = {"rmse": mean_squared_error(y_test, prediction), "mae": mean_absolute_error(y_test, prediction), "r2": r2_score(y_test, prediction)}

        return metrics

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        cv: int = 5,
    ) -> Dict:
        """
        Perform cross-validation

        Args:
            X: Feature data
            y: Target data
            cv: Number of cross-validation folds

        Returns:
            Dictionary of cross-validation results using sklearn cross_val_score
        """
        # TODO: Use Pipeline when scaling, and choose classifier/regressor based on task
        model = self.model
        if self.use_scaler:
            if self.task == "classification":
                pipe = Pipeline([('scaler', self.scaler), ('classifier', self.model)])
            else:
                pipe = Pipeline([('scaler', self.scaler), ('regressor', self.model)])
            model = pipe
    
            
        

        # TODO: Choose scoring metrics based on classification vs regression
        if self.task == "classification":
            scoring = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovo"]
        else:
            scoring = ["neg_mean_squared_error", "neg_mean_absolute_error", "r2"]
        
        results = {}
        # TODO: Get mean, stdev of cross_val_score for each metric
        
        for scoringtype in scoring:
            cvscores = cross_val_score(model, X, y, scoring=scoringtype, cv=cv, error_score="raise")
            results[scoringtype + " Mean"] = np.mean(cvscores)
            results[scoringtype + " Std"] = np.std(cvscores)
        
        
        return results

    def get_feature_importance(
        self, plot: bool = False, top_n: int = 20
    ) -> pd.DataFrame:
        """
        Get feature importances

        Returns:
            DataFrame with feature importances
        """
        featureimportances = self.model.feature_importances_
        # TODO: Optionally plot a bar chart of top_n feature importances
        if not plot:
            #print(f"Feature Importances: {featureimportances}")
            return self.model.feature_importances_
        else:
            zippedArrs = zip(self.feature_names, featureimportances)
            topfeatureimportances = sorted(zippedArrs, key=lambda x: -x[1])
            topnfeaturenames = topfeatureimportances[0][top_n:]
            topnfeatureimportances = topfeatureimportances[1][top_n:]
            plt.bar(topnfeaturenames, topnfeatureimportances)
            plt.xlabel("Feature")
            plt.ylabel("Importance")
            plt.show()
            
            return self.model.feature_importances_

    def tune_hyperparameters(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        param_grid: Dict,
        cv: int = 3,
        scoring: str = "roc_auc",
    ) -> Dict:
        """
        Perform grid search for hyperparameter tuning

        Args:
            X: Feature data
            y: Target data
            param_grid: Dictionary of parameters to search
            cv: Number of cross-validation folds
            scoring: Scoring metric to evaluate

        Returns:
            Dictionary with best parameters and results
        """
        # TODO: Choose classifier or regressor based on task
        model = self.model

        # TODO: Initialize GridSearchCV
        grid_search = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring=scoring)

        # TODO: Perform grid search for hyperparameter tuning
        
        grid_search.fit(X, y)
        results = {"best_params": grid_search.best_params_, "best_score": grid_search.best_score_, "all_results": grid_search.cv_results_}
        return results

    def plot_tree(
        self, tree_index: int = 0, figsize: Tuple[int, int] = (20, 15)
    ) -> None:
        """
        Plot a specific tree from the ensemble

        Args:
            tree_index: Index of the tree to plot
            figsize: Figure size for the plot
        """
        tree = self.model.estimators_[tree_index]
        plt.figure(figsize)
        plot_tree(tree, feature_names=self.feature_names)
        plt.show()
