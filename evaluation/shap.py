import shap
from sklearn.linear_model import LinearRegression


def shap_explainer(model, X_train):
    if isinstance(model,LinearRegression):
        return shap.LinearExplainer(model, X_train)
    else:
        return shap.TreeExplainer(model)

def shap_values(model, X_test, X_train):
    explainer = shap_explainer(model, X_train)
    return explainer.shap_values(X_test)

