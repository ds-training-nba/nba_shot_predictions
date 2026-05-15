import pandas as pd
import matplotlib.pyplot as plt
import shap

from evaluation.shap import shap_explainer, shap_values


def feature_importance_bar_plot(columns, importances, title, importance_col_name="importance"):
    importance_df = pd.DataFrame({
        "feature": columns,
        importance_col_name: importances
    }).sort_values(importance_col_name, ascending=True)

    plt.figure(figsize=(8, 10))
    plt.barh(importance_df["feature"], importance_df[importance_col_name])
    plt.title(title)


def create_shap_plotter_from_model_and_data(model, X_test, X_train):
    explainer = shap_explainer(model, X_train)
    values = shap_values(model, X_test, X_train)
    return ShapPlotter(explainer, values, X_test)

class ShapPlotter:
    def __init__(self, explainer: shap.Explainer, shap_values, X_test):
        self.explainer = explainer
        self.shap_values = shap_values
        self.X_test = X_test

    def plot_summary(self):
        shap.summary_plot(
            self.shap_values,
            self.X_test,
            plot_type="bar"
        )
    def plot_interaction_summary(self):
        sample = self.X_test.sample(1000, random_state=42)


        sample_renamed = sample.copy()

        sample_renamed.columns = [
            c.replace("cat__", "").replace("ACTION_TYPE_", "").replace("SHOT_TYPE_", "").replace("num__").replace("PLAYER_ID", "ply")
            for c in sample.columns
        ]
        interaction_values = self.explainer.shap_interaction_values(sample_renamed)
        shap.summary_plot(
            interaction_values,
            sample_renamed,
            plot_size=(35, 12)
        )



    def plot_force(self, index):
        shap.force_plot(
            self.explainer.expected_value,
            self.shap_values[index],
            self.X_test.iloc[index],
            matplotlib=True
        )

    def plot_waterfall(self, index):
        shap.plots.waterfall(self.explainer(self.X_test)[index])

    def plot_dependence(self, column_name):
        shap.dependence_plot(
            column_name,
            self.shap_values,
            self.X_test
        )