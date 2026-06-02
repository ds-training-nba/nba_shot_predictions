from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
import tensorflow.keras
import tensorflow as tf

import app.conf.run
from app.conf.run import RunConfig, ModelConfig, build_best_run_config
from app.data_providers import ready_split_dataset
from app.model_persistence import model_path, persist_model, load_model


def model_prediction(config: RunConfig):

    """
    whole processing pipeline, including prediction on the test and train sets
    (train included to be able to estimate overfitting)
    :return: y_pred, y_test, y_pred_train, y_train
    """
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    model = build_model(config.model_config)
    model.fit(X_train, y_train)

    y_pred = predict(model, X_test) if not config.return_probabilities else predict_probabilities(model, X_test)
    y_pred_train = predict(model, X_train) if not config.return_probabilities else predict_probabilities(model, X_train)
    return y_pred, y_test, y_pred_train, y_train



def evaluate_predictions(y_test, y_pred):
    """
    Convenience function to bundle evaluation via confusion matrix and classification report
    :param y_test:
    :param y_pred:
    :return: cm, cr
    """
    cm = pd.crosstab(y_test, y_pred, rownames=['Real Class'], colnames=['Predicted Class'])
    cr = classification_report(y_test, y_pred)
    return cm, cr

def predict(model, X):
    """
    Abstraction for models that do not always have sklearn interface (in the future)
    :param model:
    :param X:
    :return:
    """
    return model.predict(X)

def predict_probabilities(model, X):
    """
    Abstraction for models that do not always have sklearn interface (in the future)
    :param model:
    :param X:
    :return:
    """
    return model.predict_proba(X)


def build_model(model_config: ModelConfig):
    """
    Factory function to build the model object according to a configuration
    Hard coded parameters are results of the RandomizedSearchCV experiments (except for the DeepLearning model)
    :param model_config:
    :return:
    """
    model = None
    match model_config.model_id:
        case app.conf.run.MODEL_ID_RANDOM_FOREST:
            model = RandomForestClassifier(
                n_estimators=100,
                min_samples_split=5,
                min_samples_leaf=5,
                max_features="sqrt",
                max_depth=5,
                class_weight="balanced",
                bootstrap=True
            )
        case app.conf.run.MODEL_ID_LOGISTIC_REGRESSION:
            # params according to RandomSearchCV
            model = LogisticRegression(solver="liblinear", l1_ratio=0, max_iter=1000, class_weight=None, C=1)
        case app.conf.run.MODEL_ID_DECISION_TREE:
            model = DecisionTreeClassifier(
                min_samples_split=10,
                min_samples_leaf=2,
                max_depth=8,
                criterion="gini",
                class_weight=None
            )
        case app.conf.run.MODEL_ID_LIGHT_GBM:
            model = LGBMClassifier(
                subsample=0.8,
                num_leaves=31,
                n_estimators=100,
                min_child_samples=20,
                max_depth=10,
                learning_rate=0.05,
                colsample_bytree=0.8
            )
        case app.conf.run.MODEL_ID_SIMPLE_LOOKUP:
            model = SimpleLookupClassifier(["MAIN_ACTION_TYPE", "PLAYER_NAME"])
        case app.conf.run.MODEL_ID_DEEP_LEARNING:
            model = DeepLearningClassifier(
                [

                    {
                        "n_neurons": 60,
                        "activation": "gelu",
                    },
                    {
                        "n_neurons": 15,
                        "activation": "gelu",
                    },
                ]
            )
    if not model_config.wrap_calibrated:
        return model
    else:
        return CalibratedClassifierCV(
            model,
            method="isotonic",
            cv=3
        )

def build_fitted_model(config: RunConfig):
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    model = build_model(config.model_config)
    model.fit(X_train, y_train)
    return model

def build_persisted_model(model_id: str):
    """
    Build a model and persist it. To not mix up build time and runtime configurations,
    the only parameter is model id. The config is fixed for it.
    :param model_id:
    :return:
    """
    config = build_best_run_config()
    config.model_config.model_id = model_id
    model = build_fitted_model(config)
    persist_model(model, model_path(config.model_config.model_id))
    return model

def load_persisted_model(model_id: str):
    """
    Convenience function to retrieve a persisted model
    :param model_id:
    :return:
    """
    return load_model(model_path(model_id))

def build_param_grid(model_config: ModelConfig):
    """
    Input parameter grids for Grid/RandomizedSearchCV
    :param model_config:
    :return:
    """
    match model_config.model_id:
        case app.conf.run.MODEL_ID_RANDOM_FOREST:
            return {
                "n_estimators": [100, 300],
                "max_depth": [5, 10, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "max_features": ["sqrt", "log2"],
                "class_weight": [None, "balanced"],
                "bootstrap": [True]
            }

        case app.conf.run.MODEL_ID_LOGISTIC_REGRESSION:
            return {
                "C": [0.01, 0.1, 1, 10],
                "penalty": ["l2"],
                "solver": ["lbfgs", "liblinear"],
                "class_weight": [None, "balanced"],
                "max_iter": [1000]
            }
        case app.conf.run.MODEL_ID_LIGHT_GBM:
            return {
                "n_estimators": [100, 300],
                "learning_rate": [0.01, 0.05, 0.1],
                "num_leaves": [15, 31, 63],
                "max_depth": [-1, 5, 10],
                "min_child_samples": [10, 20, 50],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0]
            }
        case app.conf.run.MODEL_ID_DECISION_TREE:
            return {
                "max_depth": [3, 5, 8, 12, None],
                "min_samples_split": [2, 5, 10, 20],
                "min_samples_leaf": [1, 2, 5, 10],
                "criterion": ["gini", "entropy"],
                "class_weight": [None, "balanced"]
            }
def run_grid_search(config: RunConfig, cv):
    """
    Actually runs a randomizedSearchCV bc of slow traiining and big hyperparameter space
    :param config: The run config defining model and variables
    :param cv: no of cross validation runs
    :return: y_pred, y_test, {"best_params": search.best_params_, "best_score": search.best_score_ }
    """
    model = build_model(config.model_config)
    param_grid = build_param_grid(config.model_config)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        cv=cv,
        scoring=config.metric_string,
        n_jobs=-1,
        n_iter=10
    )
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    search.fit(X_train, y_train)

    y_pred = predict(search, X_test) if not config.return_probabilities else predict_probabilities(search, X_test)
    return y_pred, y_test, {"best_params": search.best_params_, "best_score": search.best_score_ }

def run_feature_selection(config: RunConfig):
    """
    Similar to run_grid_search, we run a feature selection according to a run config
    :param config:
    :return:
    """
    model = build_model(config.model_config)
    X_train, y_train, X_test, y_test, X_train_orig, X_test_orig = ready_split_dataset(config)
    rfecv = RFECV(
        estimator=model,
        step=1,
        cv=3,
        scoring="neg_brier_score",
        n_jobs=-1,
        verbose=2
    )

    X_train_sel = rfecv.fit_transform(X_train, y_train)
    ranking = pd.Series(rfecv.ranking_, index=X_train.columns)
    y_pred = rfecv.predict(X_test)


    ranking = ranking.sort_values()
    selected_features = X_train.columns[rfecv.support_]
    rejected_features = X_train.columns[~rfecv.support_]
    return  ranking, y_pred, y_test

class SimpleLookupClassifier:
    """
    Classifier class that implements training and predict()/predict_proba() interfaces of sklearn models.
    The idea of its algorithm is to just take very few of the most important variables and get mean
    predictions for every combination, which are stored in a LookUp-Table dataFrame for prediction
    """
    def __init__(self, cols):
        self.cols = cols
        self.lookup_dict = {}
        self.global_mean = 0.5

    def fit(self, X_train: pd.DataFrame, y_train):
        train_lookup = X_train.copy()

        train_lookup["y"] = y_train.values
        self.global_mean = y_train.mean()
        lookup_table = (
            train_lookup
            .groupby(
                self.cols
            )["y"]
            .mean()
            .reset_index()
            .rename(columns={"y": "p_make"})
        )
        self.lookup_dict = {
            self.row_values_as_tuple(row): row["p_make"]

            for _, row in lookup_table.iterrows()
        }
    def row_values_as_tuple(self, row):
        return tuple(
            row[col] for col in self.cols
        )
    def predict(self, X):

        probs = self.predict_proba(X)
        predictions = (probs[:,1] >= 0.5).astype(int)

        return predictions

    def predict_proba(self, X):
        probs = []

        for _, row in X.iterrows():
            key = self.row_values_as_tuple(row)

            # fallback to global mean if unseen
            p = self.lookup_dict.get(key, self.global_mean)

            probs.append([(1-p),p]) # sklearn compatible probs: [0-prob 1-prob]

        return np.array(probs)

class DeepLearningClassifier:
    """
    Classifier class that implements training and predict()/predict_proba() interfaces of sklearn models.
    Wraps a keras neural network that it builds according to the data during the fit() function.
    """
    def __init__(self, layer_config):
        self.layer_config = layer_config
        self.keras_model = None
        self.training_history = None

    def fit(self, X_train, y_train):
        """
        Build and train the wrapped neural network
        :param X_train: explanatory variables
        :param y_train: true values for training
        :return: None
        """
        from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

        early_stopping = EarlyStopping(
            patience=10,  # Wait for 5 epochs before applying
            min_delta=0.0005,  # If the loss function doesn't change by 1% after 5 epochs, either up or down, we stop
            verbose=1,  # Display the epoch at which training stops
            mode='min',
            monitor='val_loss')
        cat_inputs, num_inputs, embeddings, X_train_split = self.build_inputs_and_embeddings(X_train)
        model = self.compose_model(cat_inputs,num_inputs, embeddings)

        epochs = 50
        batch_size = 128
        steps_per_epoch = len(X_train) // batch_size
        total_steps = steps_per_epoch * epochs
        lr_schedule = tensorflow.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=1e-3,
            decay_steps=total_steps,
            alpha=1e-2
        )

        model.compile(
                       # loss='binary_crossentropy',  # Loss function
                        loss="mse", # we are dealing with noisy data. BCE punishes confident wrong predictions (90% for a missed dunk)
                                    # too hard, although statistically, the prediction was plausible. In this context,
                                    # "mse" results in brier_score_loss.
                      optimizer=tensorflow.keras.optimizers.AdamW(learning_rate=lr_schedule,weight_decay=1e-6),  # Optimization algorithm
                      metrics=['AUC', 'accuracy'])  # Evaluation metric

        self.training_history = model.fit(X_train_split, y_train,  # Training data
                                     epochs=epochs,  # Number of epochs
                                     batch_size=batch_size,  # Batch size
                                     validation_split=0.2,   # Proportion of the validation set
                                     callbacks=[]
                                          )

        self.keras_model = model
    def compose_model(self,all_inputs, num_inputs, embeddings):
        """
        compose and instantiate the actual model
        Following the idea that the network could profit from the combination of a deep MLP part and a wide linear part,
        here the two parts are combined
        :param all_inputs: all input layers to be fed into actual model
        :param num_inputs: numerical inputs only. As input for the wide layers part of the network
        :param embeddings: category embeddings as input for the deep layers part of the network
        :return: tensorflow.keras.Model
        """
        deep = self.build_deep_layers(embeddings)
        wide =  self.build_wide_layers(num_inputs) # currently unused, because we try deep only
        logits = tensorflow.keras.layers.Add()([wide, deep]) # currently unused, because we try deep only
        output = tensorflow.keras.layers.Activation("sigmoid")(logits) # currently unused, because we try deep only
        return tensorflow.keras.Model(
            inputs=all_inputs,
            outputs=deep
        )
    def build_wide_layers(self, inputs):
        """
        Build wide layers part of the network
        :param inputs:
        :return: KerasTensor
        """
        wide = tensorflow.keras.layers.Concatenate()(inputs)

        wide = tensorflow.keras.layers.Dense(1, use_bias=True)(wide)
        return wide
    def build_deep_layers(self, embeddings):
        """
        Build wide layers part of the network
        :param embeddings:
        :return: KerasTensor
        """
        x = tensorflow.keras.layers.Concatenate()(

            embeddings
        )
        mlp_layers = []
        for i, config in enumerate(self.layer_config):
            mlp_layers.append(Dense(units=config['n_neurons'],
                            activation=config['activation'],
                            kernel_initializer='normal'))
            mlp_layers.append(Dropout(0.2))

        mlp_layers.append(Dense(units=1))
        mlp = Sequential(mlp_layers)
        return mlp(x)


    def separate_dataframes(self, X: pd.DataFrame):
        """
        Convenience function to separate numerical and categorical data
        :param X:
        :return:
        """
        X_num = X.select_dtypes(include=['int', 'float'])
        X_cat = X.select_dtypes(include=['category'])
        return X_num, X_cat
    def build_inputs_and_embeddings(self, X_train: pd.DataFrame):
        """
        analyze the input structure and generate inputs, embeddings and restructured data accordingly.
        :param X_train:
        :return:
        """
        data = {}
        X_num, X_cat = self.separate_dataframes(X_train)
        all_inputs = []
        num_inputs = []
        embeddings = []

        def bin_feature(x, bins):
            x = tf.clip_by_value(x, 0.0, 0.9999)

            bucket_ids = tf.floor(x * bins)

            return tf.cast(bucket_ids, tf.int32)

        for col in X_cat.columns:
            name = col + "_input"
            cat_input = Input(
                shape=(1,),
                name=name
            )
            all_inputs.append(cat_input)
            data[name] = pd.DataFrame(X_cat[col].cat.codes)
            embedding = tensorflow.keras.layers.Embedding(
                input_dim=X_cat[col].nunique(),
                output_dim=int(np.floor(np.sqrt(X_cat[col].nunique())/2))
            )(cat_input)
            embedding = tensorflow.keras.layers.Flatten()(embedding)
            embeddings.append(embedding)

        for col in X_num.columns:
            name = col + "_input"
            num_input = Input(
                shape=(1,),
                name=name
            )
            all_inputs.append(num_input)
            num_inputs.append(num_input)
            data[name] = X_num[[col]]
            num_bins = 20
            layer =  tensorflow.keras.layers.Lambda(lambda x: bin_feature(x, num_bins))(num_input)
            embedding = tensorflow.keras.layers.Embedding(
                input_dim=num_bins + 2,
                output_dim=4
            )(layer)
            embedding = tensorflow.keras.layers.Flatten()(embedding)
            embeddings.append(embedding)

        return all_inputs, num_inputs, embeddings, data

    def predict_proba(self, X):
        """
        Implement the sklearn predict_proba interface
        :param X: input DataFrame
        :return: numpy array with an array of probabilities for each class as a result for each prediction
        """
        cat_inputs,num_inputs, embeddings, X_split = self.build_inputs_and_embeddings(X)
        prob_1 = self.keras_model.predict(X_split, verbose=0).reshape(-1)
        prob_0 = 1.0 - prob_1

        return np.column_stack([prob_0, prob_1])

    def predict(self, X):
        """
        Implement the sklearn predict interface
        :param X: input DataFrame
        :return: numpy array of predicted classes
        """
        probs = self.predict_proba(X)
        predictions = (probs[:,1] >= 0.5).astype(int)
        return predictions