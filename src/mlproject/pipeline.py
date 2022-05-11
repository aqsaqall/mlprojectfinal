from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline_LogReg(
    use_scaler: bool, max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(
                random_state=random_state, max_iter=max_iter, C=logreg_C
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_pipeline_RandomForest(
    use_scaler: bool, n_estimators: int, criterion: str, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            RandomForestClassifier(
                random_state=random_state, n_estimators=n_estimators, criterion=criterion
            ),
        )
    )
    return Pipeline(steps=pipeline_steps)