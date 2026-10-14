# ResiNet-IDAM

## Adaptive Deep Learning and Zero-Trust Framework for Detecting Zero-Day Intrusions in IT-OT Hybrid Critical Infrastructures

## 1. Project Overview

ResiNet-IDAM is a research project focused on developing an adaptive, intelligent security framework for detecting suspicious activities and potential zero-day intrusions in hybrid Information Technology (IT) and Operational Technology (OT) environments.

Modern critical infrastructures depend on interconnected IT systems and industrial OT environments. Traditional security solutions may struggle to identify previously unseen attack patterns, particularly when threats cross the boundary between enterprise networks and industrial control systems.

ResiNet-IDAM aims to address this challenge by combining deep-learning-based anomaly detection, IT and OT telemetry analysis, and Zero Trust security principles in a unified framework.

**Core concept:** Monitor system behavior, detect anomalies, assess risk using contextual security signals, and support informed security responses.

## 2. Problem Statement

IT and OT environments have different operational characteristics, but their increasing interconnection creates additional security risks. Conventional detection mechanisms that depend heavily on known attack signatures may fail to identify unfamiliar or evolving threats.

A unified framework is needed to analyze behavior across IT and OT environments, identify potentially malicious deviations, and incorporate identity, device, and contextual information into security decisions.

## 3. Project Objectives

- Develop anomaly-detection mechanisms using deep learning and machine learning.
- Analyze industrial telemetry to identify deviations from normal OT behavior.
- Extend detection capabilities to IT security telemetry.
- Investigate adaptive detection approaches for previously unseen attack patterns.
- Incorporate Zero Trust principles into risk assessment and access-related decisions.
- Correlate IT and OT security findings to support unified monitoring.
- Develop a monitoring dashboard for presenting anomalies, alerts, and security insights.

## 4. Proposed System Architecture

The proposed framework consists of the following components:

1. **IT Data Collection:** Collect relevant enterprise telemetry, such as authentication events, server logs, and network activity.
2. **OT Data Collection:** Process industrial telemetry from sensors, programmable logic controllers (PLCs), supervisory control and data acquisition (SCADA) systems, or suitable benchmark datasets.
3. **Data Preprocessing:** Clean, transform, normalize, and prepare data for analysis.
4. **Anomaly Detection:** Apply machine-learning and deep-learning models to identify deviations from expected behavior.
5. **Adaptive Detection:** Investigate mechanisms for updating detection behavior as operating conditions and threat patterns change.
6. **Zero Trust Risk Assessment:** Evaluate available identity, device, behavior, and contextual signals to support risk-aware security decisions.
7. **IT-OT Correlation:** Combine relevant findings from IT and OT monitoring to identify potentially related security events.
8. **Monitoring and Response:** Present alerts, risk indicators, and detection results through a unified dashboard and support appropriate response workflows.

### Architecture Flow

IT Telemetry + OT Telemetry → Data Preprocessing → Anomaly Detection → Risk Assessment and IT-OT Correlation → Monitoring and Response

*This represents the proposed end-to-end architecture. Components are implemented progressively, and the complete integrated framework is not yet finished.*

## 5. Technology Stack

The technologies used or planned for the project include:

| Component | Technology |
|---|---|
| Programming language | Python |
| Data processing | pandas, NumPy |
| Baseline anomaly detection | scikit-learn |
| Deep learning | TensorFlow / Keras |
| Model persistence | joblib and Keras model files |
| OT benchmark dataset | HAI industrial telemetry dataset |
| IT detection | To be selected and integrated |
| Risk assessment and Zero Trust | Planned framework component |
| Unified dashboard | Planned integration component |

## 6. Current Implementation Status

### Implemented: Initial OT Detection Component

- HAI industrial telemetry data preparation.
- Feature selection and preprocessing.
- Feature scaling for model inputs.
- Isolation Forest baseline training and evaluation.
- Autoencoder training using a chronological training and validation split.
- Preliminary evaluation against labeled HAI test data.
- Experimental comparison of anomaly-detection performance.

### Planned or In Progress

- Integration of an appropriate IT intrusion-detection dataset.
- IT anomaly-detection pipeline.
- Unified IT-OT event correlation.
- Zero Trust risk-scoring and contextual decision mechanisms.
- Adaptive model updating and evaluation.
- Unified security monitoring dashboard.
- Evaluation of the integrated framework under suitable test conditions.

## 7. OT Component Documentation

Detailed instructions for setting up and running the current OT implementation are available in:

- **[OT Setup and Usage Guide](OT_SETUP.md)**

The guide covers environment setup, dataset requirements, preprocessing, model training, evaluation, and troubleshooting.

## 8. Research and Evaluation

The current OT experiments use the HAI industrial telemetry dataset to investigate anomaly detection with Isolation Forest and an autoencoder.

Evaluation includes attack precision, attack recall, attack F1-score, false positives, and false negatives.

The current results are preliminary and specific to the implemented experiments. They do not independently establish reliable zero-day detection or validate the complete IT-OT and Zero Trust framework.

## 9. Project Limitations

- The current implementation focuses on the OT anomaly-detection component.
- The full IT-OT integration and Zero Trust framework remain to be implemented and evaluated.
- Benchmark dataset performance may not directly represent real-world deployment conditions.
- Further controlled experiments are needed to evaluate generalization to previously unseen attacks.
- Detection thresholds, preprocessing, and evaluation procedures must be validated carefully to avoid test-data leakage.

## 10. Repository Structure

```text
ResiNet-IDAM/
├── README.md
├── OT_SETUP.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Original datasets, obtained separately
│   └── processed/        # Generated/prepared datasets
├── models/               # Locally generated model files
└── src/
    ├── prepare_training_data.py
    ├── prepare_test_data.py
    ├── scale_features.py
    ├── train_baseline.py
    ├── evaluate_baseline.py
    ├── train_autoencoder.py
    ├── evaluate_autoencoder.py
    ├── dataset_report.py
    └── validate_labels.py
```

The repository structure may evolve as additional components are developed. Large datasets and generated model binaries are not necessarily stored in Git.

## 11. Research Goal

The long-term goal of ResiNet-IDAM is to investigate how anomaly detection, contextual risk assessment, and Zero Trust principles can work together to improve security monitoring across interconnected IT and OT environments.

The framework will be evaluated incrementally, with the current OT implementation serving as an initial component of the larger system.
