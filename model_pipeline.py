import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def build_isolation_forest_pipeline(
    contamination: float = 0.02, random_state: int = 42
) -> Pipeline:
    """
    create a simple pipeline that:

    - scales numeric columns  
    - one hot encodes text columns  
    - runs an isolation forest model on the final features
    """
    numeric_scaler = StandardScaler()
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)

    preprocess = ColumnTransformer(
        transformers=[
            ("num", numeric_scaler, lambda df: df.select_dtypes(include=["number"]).columns),
            ("cat", categorical_encoder, lambda df: df.select_dtypes(exclude=["number"]).columns),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("model", isolation_forest),
        ]
    )
    return pipeline


def train_model_and_score_anomalies(
    model: Pipeline, data: pd.DataFrame
) -> pd.DataFrame:
    """
    fit the pipeline and return a table with anomaly scores and labels.

    isolation forest gives higher scores for normal rows. for easier reading
    we flip the sign so that higher numbers now mean "more anomalous".
    """
    model.fit(data)
    decision_scores = model.decision_function(data)

    anomaly_score = -decision_scores

    predicted_labels = model.predict(data)
    is_anomaly = (predicted_labels == -1).astype(int)

    scored_data = data.copy()
    scored_data["anomaly_score"] = anomaly_score
    scored_data["is_anomaly"] = is_anomaly
    return scored_data


def explain_anomalies_in_plain_language(
    scored_data: pd.DataFrame, top_n_features: int = 5
) -> str:
    """
    very small rule based explanation that compares anomalies to normal rows.
    it only looks at how much the average value per feature is different.
    """
    if "is_anomaly" not in scored_data.columns:
        return "No anomaly information available."

    anomalies = scored_data[scored_data["is_anomaly"] == 1]
    normal_points = scored_data[scored_data["is_anomaly"] == 0]

    if len(anomalies) == 0 or len(normal_points) == 0:
        return "The model did not clearly separate anomalies from normal data, so no explanation can be derived."

    numeric_feature_cols = [
        c
        for c in scored_data.select_dtypes(include=["number"]).columns
        if c not in ["anomaly_score", "is_anomaly"]
    ]

    if not numeric_feature_cols:
        return (
            "The model produced anomaly scores, but there are no numeric features "
            "available to compare anomalies against normal points."
        )

    mean_anom = anomalies[numeric_feature_cols].mean()
    mean_norm = normal_points[numeric_feature_cols].mean()

    diff = (mean_anom - mean_norm).abs().sort_values(ascending=False)
    top_features = diff.head(top_n_features).index.tolist()

    explanation_lines = []
    explanation_lines.append(
        f"In this dataset, the model marked {len(anomalies)} out of {len(scored_data)} rows as anomalous."
    )
    explanation_lines.append(
        "When comparing anomalies to normal points, the following features differ the most on average:"
    )
    for feat in top_features:
        direction = "higher" if mean_anom[feat] > mean_norm[feat] else "lower"
        explanation_lines.append(
            f"- Feature `{feat}` tends to be {direction} for anomalous points compared to normal ones."
        )

    explanation_lines.append(
        "These patterns suggest that extreme values in these features are strong indicators of unusual behavior in the system."
    )

    return "\n".join(explanation_lines)

