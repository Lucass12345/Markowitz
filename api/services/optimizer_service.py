from src.tracking.mlflow_tracker import start_run, log_params, log_metrics

with start_run(run_label):

    log_params(config)

    results = run_optimizer(...)

    log_metrics({
        "return_annual": results["return_annual"],
        "vol_annual": results["vol_annual"],
        "sharpe_annual": results["sharpe_annual"]
    })