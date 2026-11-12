
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def get_feature_columns(config):
    features = config["features"]

    text_columns = features.get("text", [])
    categorical_columns = features.get("categorical", [])
    numeric_columns = features.get("numeric", [])

    return text_columns, categorical_columns, numeric_columns


def build_preprocessor(config):
    text_columns, categorical_columns, numeric_columns = (get_feature_columns(config))
    transformers = []

    for column in text_columns:
        text_pipeline = Pipeline([("tfidf", TfidfVectorizer())])
        transformers.append((f"text_{column}", text_pipeline, column))

    if categorical_columns:
        categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])

        transformers.append(("categorical", categorical_pipeline, categorical_columns))

    if numeric_columns:
        numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])

        transformers.append(("numeric", numeric_pipeline, numeric_columns))

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return preprocessor


def prepare_features(data, config):
    text_columns, categorical_columns, numeric_columns = (get_feature_columns(config))

    feature_columns = (text_columns + categorical_columns + numeric_columns)
    features = data[feature_columns].copy()

    for column in text_columns:
        features[column] = features[column].fillna("")
        
    return features
