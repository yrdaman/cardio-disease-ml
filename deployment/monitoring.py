"""
Monitoring utilities (conceptual).

This module outlines metrics that would be monitored
post-deployment in a production environment.
"""

def monitoring_plan():
    metrics = [
        "Recall",
        "Precision",
        "Prediction probability distribution",
        "False Negative rate",
        "Data drift indicators"
    ]

    return {
        "frequency": "Periodic (e.g., weekly)",
        "metrics": metrics,
        "notes": "Monitoring not automated in this prototype"
    }
