from pathlib import Path
from joblib import dump

import click
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.model_selection import  cross_validate
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectFromModel
from sklearn.tree import DecisionTreeClassifier

from .data import get_dataset
from .pipeline import create_pipeline_LogReg
from .pipeline import create_pipeline_RFC

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True,
)
@click.option(
    "--random-state",
    default=42,
    type=int,
    show_default=True,
)
@click.option(
    "--test-split-ratio",
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True,
)
@click.option(
    "--use-model",
    default='LogReg',
    type=str,
    show_default=True,
)
@click.option(
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--use-feature-selection", 
    default=0, 
    type=int,
    show_default=True,
)
@click.option(
    "--max_iter",
    default=1000,
    type=int,
    show_default=True,
)
@click.option(
    "--logregc",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--n_estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="entropy",
    type=str,
    show_default=True,
)
@click.option(
    "--max_depth",
    default=10,
    type=int,
    show_default=True,
)
@click.option(
    "--bootstrap",
    default=True,
    type=bool,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    use_feature_selection: int,
    logregc: float, 
    max_iter: int,
    use_model: str,
    n_estimators: int,
    criterion: str,
    max_depth: int,
    bootstrap: bool,
) -> None:
    features_train, features_val, target_train, target_val = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        if use_model == 'LogReg':
            pipeline = create_pipeline_LogReg(use_scaler, max_iter, logregc, random_state)
            print("Use_model == 'LogReg'")
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logregc", logregc)
            mlflow.log_param("random_state", random_state)
        elif use_model == 'RFC':
            pipeline = create_pipeline_RFC(use_scaler, random_state, criterion, max_depth, bootstrap, n_estimators)
            print("Use_model == 'RFC'")
            mlflow.log_param("criterion", criterion)
            mlflow.log_param("max_depth", max_depth)
            mlflow.log_param("bootstrap", bootstrap)
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("random_state", random_state)
        if use_feature_selection == 0:
            cross_val = cross_validate(pipeline, features_train, target_train, cv=5, scoring=('accuracy', 'f1_weighted', 'precision_weighted'))
            print("No use_feature_selection")
        elif use_feature_selection == 1:
            train_tranc = PCA(n_components='mle', svd_solver='full', random_state=random_state).fit_transform(features_train)
            click.echo(f"tranc shape: {train_tranc.shape}.")
            cross_val = cross_validate(pipeline, train_tranc, target_train, cv=5, scoring=('accuracy', 'f1_weighted', 'precision_weighted'))        
            print("Use_feature_selection == 'PCA'")
        elif use_feature_selection == 2:
            selection_model = DecisionTreeClassifier(random_state=random_state)
            pipe_selection = make_pipeline(SelectFromModel(selection_model), pipeline)
            cross_val = cross_validate(pipe_selection, features_train, target_train, cv=5, scoring=('accuracy', 'f1_weighted', 'precision_weighted'))
            print("Use_feature_selection == 'SelectFromModel'")


        
        mlflow.log_param("use_model", use_model)
        mlflow.log_param("use_scaler", use_scaler)
        mlflow.log_param("use_feature_selection", use_feature_selection)

        accuracy = np.mean(cross_val['test_accuracy'])
        precision = np.mean(cross_val['test_precision_weighted'])
        f1score = np.mean(cross_val['test_f1_weighted'])
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("f1score", f1score)
        click.echo(f"Accuracy: {accuracy}.")
        click.echo(f"Precision: {precision}.")
        click.echo(f"F1score: {f1score}.")
        dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")