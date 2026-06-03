import shap
from sklearn.linear_model import LinearRegression


def shap_explainer(model, X_train):
    """
    Factory for SHAP explainer according to model
    :param model:
    :param X_train:
    :return:
    """
    if isinstance(model,LinearRegression):
        return shap.LinearExplainer(model, X_train)
    else:
        return shap.TreeExplainer(model)

def shap_values(model, X_test, X_train):
    """
    Convenience function to get shap values for model and input data
    :param model:
    :param X_test:
    :param X_train:
    :return:
    """
    explainer = shap_explainer(model, X_train)
    return explainer.shap_values(X_test)

