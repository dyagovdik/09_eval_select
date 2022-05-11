from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier


def create_pipeline_LogReg(
    use_scaler: bool, max_iter: int, logregc: float, random_state: int
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))
    pipeline_steps.append(
        (
            "classifier",
            LogisticRegression(random_state=random_state, max_iter=max_iter, C=logregc),
        )
    )
    return Pipeline(steps=pipeline_steps)


def create_pipeline_RFC(
    use_scaler: bool,
    random_state: int,
    criterion: str,
    max_depth: int,
    bootstrap: bool,
    n_estimators: int,
) -> Pipeline:
    pipeline_steps = []
    if use_scaler:
        pipeline_steps.append(("scaler", StandardScaler()))

        pipeline_steps.append(
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=n_estimators,
                    random_state=random_state,
                    criterion=criterion,
                    max_depth=max_depth,
                    bootstrap=bootstrap,
                ),
            )
        )
    return Pipeline(steps=pipeline_steps)
