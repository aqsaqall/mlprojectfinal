from pathlib import Path
import joblib
import click
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, log_loss
from sklearn.model_selection import StratifiedKFold

from .data import get_dataset
from .pipeline import create_pipeline_LogReg, create_pipeline_RandomForest

import numpy as np

@click.command()
@click.option(
    "-d",
    "--dataset-path",
    default="src/mlproject/data/train.csv",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    show_default=True,
)
@click.option(
    "-s",
    "--save-model-path",
    default="src/mlproject/data/model.joblib",
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
    "--use-scaler",
    default=True,
    type=bool,
    show_default=True,
)
@click.option(
    "--max-iter",
    default=500,
    type=int,
    show_default=True,
)
@click.option(
    "--logreg-c",
    default=1.0,
    type=float,
    show_default=True,
)
@click.option(
    "--n-estimators",
    default=100,
    type=int,
    show_default=True,
)
@click.option(
    "--criterion",
    default="gini",
    type=str,
    show_default=True,
)
@click.option(
    "--model-type",
    default="LogisticRegression",
    type=str,
    show_default=True,
)
def train(
    dataset_path: Path,
    save_model_path: Path,
    random_state: int,
    test_split_ratio: float,
    use_scaler: bool,
    max_iter: int,
    logreg_c: float,
    n_estimators: int,
    criterion: str,
    model_type: str
) -> None:
    data_features, data_target = get_dataset(
        dataset_path,
        random_state,
        test_split_ratio,
    )
    with mlflow.start_run():
        if model_type=="LogisticRegression": pipeline = create_pipeline_LogReg(use_scaler, max_iter, logreg_c, random_state)
        else: pipeline = create_pipeline_RandomForest(use_scaler, n_estimators, criterion, random_state)
        cv = StratifiedKFold(n_splits=5, random_state=random_state, shuffle=True)
        accuracy_list, f1_list, rocauc_list = [], [], []
        for (train, test), i in zip(cv.split(data_features, data_target), range(5)):
            pipeline.fit(data_features.iloc[train], data_target.iloc[train])
            accuracy_list.append(accuracy_score(data_target.iloc[test], pipeline.predict(data_features.iloc[test])))
            f1_list.append(f1_score(data_target.iloc[test], pipeline.predict(data_features.iloc[test]), average='macro'))
            #accuracy_list.append(accuracy_score(data_target.iloc[test], pipeline.predict(data_features.iloc[test])))
        pipeline.fit(data_features, data_target)
        accuracy = np.mean(accuracy_list)
        f1 = np.mean(f1_list)
        #loss = log_loss(target_val, pipeline.predict(features_val))
        mlflow.log_param("model_type", model_type)
        mlflow.log_param("use_scaler", use_scaler)
        if model_type=="LogisticRegression":
            mlflow.log_param("max_iter", max_iter)
            mlflow.log_param("logreg_c", logreg_c)
        else:
            mlflow.log_param("n_estimators", n_estimators)
            mlflow.log_param("criterion", criterion)
        mlflow.log_metric("accuracy", accuracy)
        click.echo(f"Accuracy: {accuracy}.")
        mlflow.log_metric("f1_score", f1)
        click.echo(f"F1-score: {f1}")
        #mlflow.log_metric("Log_loss", loss)
        #click.echo(f"Log_loss: {loss}")
        joblib.dump(pipeline, save_model_path)
        click.echo(f"Model is saved to {save_model_path}.")