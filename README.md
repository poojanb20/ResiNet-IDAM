

\# ResiNet-IDAM

\## OT Anomaly Detection Component



\*\*Project:\*\* Adaptive Deep Learning and Zero-Trust Framework for Detecting Zero-Day Intrusions in IT-OT Hybrid Critical Infrastructures



This repository contains the initial Operational Technology (OT) anomaly-detection component developed using the HAI industrial telemetry dataset.



\## 1. Contribution Overview



The current implementation prepares industrial telemetry data and evaluates two anomaly-detection approaches:



\- \*\*Isolation Forest:\*\* Unsupervised baseline for detecting unusual process behavior.

\- \*\*Autoencoder:\*\* Neural network that learns to reconstruct input features and flags records with high reconstruction error.



The current scope is OT anomaly detection. IT intrusion detection, complete Zero Trust integration, adaptive response, and the unified dashboard are future integration tasks.



\## 2. Technology Stack



\- Python 3.11

\- pandas and NumPy

\- scikit-learn

\- joblib

\- TensorFlow / Keras

\- HAI industrial telemetry dataset



\## 3. Project Structure



```text

ResiNet-IDAM/

├── data/

│   ├── raw/                 # Original HAI files; obtain separately

│   └── processed/           # Prepared CSV files

├── models/                  # Generated model files; not committed

├── src/

│   ├── prepare\_training\_data.py

│   ├── prepare\_test\_data.py

│   ├── scale\_features.py

│   ├── train\_baseline.py

│   ├── evaluate\_baseline.py

│   ├── train\_autoencoder.py

│   ├── evaluate\_autoencoder.py

│   ├── dataset\_report.py

│   └── validate\_labels.py

├── .gitignore

├── requirements.txt

└── README.md

```



The repository may contain additional scripts as the project develops.



\## 4. Dataset Requirements



The implementation uses HAI industrial telemetry.



Expected original files:



\- `data/raw/hai-train1.txt`

\- `data/raw/hai-test1.txt`

\- `data/raw/label-test1.txt`



Processed files used by the current experiments:



\- `data/processed/hai-train-features.csv`

\- `data/processed/hai-train-scaled.csv`

\- `data/processed/hai-test-labeled.csv`

\- `data/processed/hai-test-scaled.csv`



`feature\_distribution\_report.csv` is an optional analysis report.



Obtain the dataset and any shared processed files from the project team, subject to the dataset's applicable terms. Do not assume the raw files are included in this repository.



\## 5. Environment Setup



Use Python 3.11 on Windows.



Clone the repository:



```powershell

git clone <REPOSITORY\_URL>

cd ResiNet-IDAM

```



Create and activate a virtual environment:



```powershell

py -3.11 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

```



Install dependencies:



```powershell

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```



Obtain the required datasets and place them in the expected directories before running the scripts.



\## 6. Data Preparation



Run these commands from the project root, in order:



```powershell

python src/prepare\_training\_data.py

python src/prepare\_test\_data.py

python src/scale\_features.py

```



These scripts prepare the training and test data and produce scaled feature files. Check each script's configured input and output paths before running it.



\## 7. Train and Evaluate Isolation Forest



Train the baseline model:



```powershell

python src/train\_baseline.py

```



Evaluate the baseline:



```powershell

python src/evaluate\_baseline.py

```



The training script saves the trained model under `models/`. Evaluation requires the appropriate processed test data and the trained model.



\## 8. Train and Evaluate the Autoencoder



Train the Autoencoder:



```powershell

python src/train\_autoencoder.py

```



Evaluate it:



```powershell

python src/evaluate\_autoencoder.py

```



The training script saves:



\- `models/ot\_autoencoder.keras`

\- `models/autoencoder\_threshold.json`



The threshold is selected from validation reconstruction errors. Evaluation compares test reconstruction errors against this saved threshold.



\## 9. Initial Experimental Results



The following results were obtained in the current experiments on the labeled HAI test set.



| Metric | Isolation Forest baseline | Autoencoder |

|---|---:|---:|

| Attack precision | 0.1049 | 0.3492 |

| Attack recall | 0.6880 | 0.4639 |

| Attack F1-score | 0.1820 | 0.3984 |

| False positives | 17,506 | 2,578 |

| False negatives | 930 | 1,598 |



The Autoencoder produced a higher attack F1-score and fewer false positives in these reported experiments, while missing more attacks.



These are preliminary results, not proof of zero-day detection. The models used different training setups, and further controlled evaluation is needed before drawing final conclusions.



\## 10. Important Notes



\- Do not train models using the labeled test set.

\- Do not use test labels to select the anomaly threshold.

\- Fit preprocessing transformations on training data and reuse them for evaluation.

\- Confirm that the test feature columns and their order match the training inputs.

\- The current Autoencoder is a candidate detector, not a validated zero-day detector.

\- HAI represents OT telemetry; it does not by itself provide the IT identity and access information required for a complete Zero Trust framework.

\- Keep large datasets, local virtual environments, credentials, and generated model binaries out of ordinary Git commits.



\## 11. Current Scope and Future Integration



Completed initial work:

\- HAI data preparation and feature scaling

\- Isolation Forest baseline

\- Autoencoder training

\- Preliminary labeled-test evaluation



Planned work:

\- Integrate an IT intrusion-detection dataset

\- Combine IT and OT detection outputs

\- Implement Zero Trust risk scoring using available identity, device, behavior, and context signals

\- Develop the unified monitoring dashboard

\- Conduct further evaluation under appropriate test conditions



