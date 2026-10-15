
from dataclasses import dataclass


@dataclass
class RiskInput:
    anomaly_score: float       # 0.0 = normal, 1.0 = highly anomalous
    classification_confidence: float
    asset_criticality: float   # 0.0 = low, 1.0 = critical
    trust_score: float         # 0.0 = untrusted, 1.0 = trusted


def calculate_risk(data: RiskInput) -> dict:
    """
    Calculate a prototype risk score between 0 and 100.

    This is a rule-based demonstration, not a production
    security policy. Scores and thresholds require calibration.
    """

    values = {
        "anomaly_score": data.anomaly_score,
        "classification_confidence": data.classification_confidence,
        "asset_criticality": data.asset_criticality,
        "trust_score": data.trust_score,
    }

    for name, value in values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0.0 and 1.0")

    # Weight the four risk factors.
    anomaly_risk = data.anomaly_score * 50
    confidence_risk = data.classification_confidence * 20
    criticality_risk = data.asset_criticality * 20
    trust_risk = (1.0 - data.trust_score) * 10

    risk_score = round(
        anomaly_risk
        + confidence_risk
        + criticality_risk
        + trust_risk,
        2,
    )

    if risk_score < 30:
        action = "ALLOW_AND_MONITOR"
    elif risk_score < 60:
        action = "REQUIRE_ADDITIONAL_VERIFICATION"
    else:
        action = "RESTRICT_AND_ALERT"

    return {
        "risk_score": risk_score,
        "risk_level": (
            "LOW" if risk_score < 30
            else "MEDIUM" if risk_score < 60
            else "HIGH"
        ),
        "recommended_action": action,
        "risk_factors": {
            "anomaly": round(anomaly_risk, 2),
            "classification_confidence": round(confidence_risk, 2),
            "asset_criticality": round(criticality_risk, 2),
            "lack_of_trust": round(trust_risk, 2),
        },
    }


if __name__ == "__main__":
    scenarios = [
        (
            "Normal activity on a low-criticality asset",
            RiskInput(0.05, 0.10, 0.20, 0.95),
        ),
        (
            "Suspicious activity on an important server",
            RiskInput(0.70, 0.80, 0.80, 0.40),
        ),
        (
            "Highly suspicious activity on a critical asset",
            RiskInput(0.95, 0.95, 1.00, 0.10),
        ),
    ]

    for name, data in scenarios:
        result = calculate_risk(data)
        print(f"\nScenario: {name}")
        print(f"Risk score: {result['risk_score']}/100")
        print(f"Risk level: {result['risk_level']}")
        print(f"Recommended action: {result['recommended_action']}")
        print(f"Risk factors: {result['risk_factors']}")
