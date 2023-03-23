import mlflow
from typing import Any, Dict


class Logger:
    def __init__(self, config: Dict) -> None:
        self.history = dict()
        mlflow.set_tracking_uri(config["mlflow_uri"])
        mlflow.set_experiment(experiment_name=config["mlflow_experiment_name"])

    def __call__(self, key: str, value: Any) -> None:
        if key in self.history:
            self.history[key] += 1
        else:
            self.history[key] = 0
        mlflow.log_metric(key=key, value=value, step=self.history[key])
