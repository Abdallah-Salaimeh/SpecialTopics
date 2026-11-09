
import pandas as pd
import yaml


with open("features.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)


df = pd.read_csv("data/cves_history.csv")


roles = {
    "label": [config["label"]],
    "id": [config["id"]],
    "time": [config["time"]],
    "features": (
        config["features"]["text"]
        + config["features"]["categorical"]
        + config["features"]["numeric"]
    ),
    "not_features": list(config["not_features"].keys()),
}


assigned_columns = [column for columns in roles.values() for column in columns]

duplicates = sorted({column for column in assigned_columns if assigned_columns.count(column) > 1})


dataset_columns = set(df.columns)
configured_columns = set(assigned_columns)

missing = sorted(dataset_columns - configured_columns)
unexpected = sorted(configured_columns - dataset_columns)


label_is_feature = config["label"] in roles["features"]


invalid_groups = sorted(set(config["groups"]) - dataset_columns)


print("Dataset columns: ", len(dataset_columns))
print("Configured columns: ", len(configured_columns))
print("Duplicate assignments: ", duplicates)
print("Missing columns: ", missing)
print("Unexpected columns: ", unexpected)
print("Label is a feature: ", label_is_feature)
print("Invalid groups: ", invalid_groups)


if (duplicates or missing or unexpected or label_is_feature or invalid_groups):
    raise SystemExit("Val has issues")

print("Val perfect")
