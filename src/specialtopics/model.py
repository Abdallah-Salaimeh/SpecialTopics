
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .features import build_preprocessor, prepare_features


def build_model(config):
    preprocessor = build_preprocessor(config)
    classifier = LogisticRegression(max_iter=1000, random_state=42)
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])


def train_model(data, config):
    label_column = config["label"]
    X = prepare_features(data, config)
    y = data[label_column]
    model = build_model(config)
    model.fit(X, y)
    return model


def predict_severity(model, data, config):
    X = prepare_features(data, config)
    return model.predict(X)


def predict_probabilities(model, data, config):
    X = prepare_features(data, config)
    probabilities = model.predict_proba(X)
    classes = model.classes_
    return probabilities, classes


def predict_critical_probability(model, data, config):
    probabilities, classes = predict_probabilities(model, data, config)

    if "CRITICAL" not in classes:
        raise ValueError("The trained model does not contain the CRITICAL class.")

    critical_index = list(classes).index("CRITICAL")
    return probabilities[:, critical_index]
