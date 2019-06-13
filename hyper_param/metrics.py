import tensorflow as tf

from gradient_statsd import Client

HYPER_PARAMS = "hyperParams"
HYPER_BEST = "hyperBest"

MetricKeys = {
    "CoarseBestAccuracy": "coarse_best_accuracy",
    "CoarseBestLoss": "coarse_best_loss",
    "CoarseEndAccuracy": "coarse_end_accuracy",
    "Coarse_end_loss": "coarse_end_loss",
    "FineBestAccuracy": "fine_best_accuracy",
    "FineBestLoss": "fine_best_loss",
    "FineEndAccuracy": "fine_end_accuracy",
    "FineEndLoss": "fine_end_loss",
    "Loss": "loss",
    "RealLoss": "real_loss",
}


def publish_metrics(result, is_best=False):
    tf.logging.info("Publish hyperopt metrics:")
    tf.logging.info("Is it best params: %s", is_best)
    client = Client()
    for key, value in MetricKeys:
        if is_best:
            metric = "{}{}".format(HYPER_BEST, key)
        else:
            metric = "{}{}".format(HYPER_PARAMS, key)
        client.gauge(metric, result.get(value))

    tf.logging.info("Publish metrics completed")
