import mlflow


def start_run(run_label):

    mlflow.set_experiment("portfolio_optimization")

    return mlflow.start_run(run_name=run_label)


def log_params(config):

    for k, v in config.items():
        mlflow.log_param(k, v)


def log_metrics(metrics):

    for k, v in metrics.items():
        mlflow.log_metric(k, v)