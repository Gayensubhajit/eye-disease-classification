from src.metrics import classification_metrics


def test_classification_metrics_contains_headline_scores():
    result = classification_metrics([0, 1, 1], [0, 1, 0])
    assert set(result) == {"accuracy", "macro_f1", "kappa"}
