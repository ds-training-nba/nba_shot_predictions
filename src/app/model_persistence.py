from pathlib import Path
import pickle
from app.config import MODELS_PATH


def model_path(model_id: str, user_suffix = 'sp'):
    """
    returns a consistent path for model persistence
    :param model_id:
    :param user_suffix:
    :return: Path
    """
    file_suffix = ".pkl"
    path = Path(MODELS_PATH)
    return path / (model_id + "-" + user_suffix + file_suffix)

def persist_model(model, model_path: Path):
    """
    Convenience function to persist a model
    :param model:
    :param model_path:
    :return:
    """
    with open(model_path.as_posix(), 'wb') as file:
        pickle.dump(model, file)


def load_model(model_path: Path):
    """
    convenience function to retrieve a persisted model
    :param model_path:
    :return:
    """
    with open(model_path.as_posix(), 'rb') as file:
        model = pickle.load(file)
    return model
