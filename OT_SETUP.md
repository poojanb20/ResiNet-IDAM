\# ResiNet-IDAM: OT Anomaly Detection Setup



This document explains how to set up, prepare, train, and evaluate the Operational Technology (OT) anomaly-detection component of ResiNet-IDAM.



For the complete project overview and proposed IT-OT architecture, see \[README.md](README.md).



\## 1. Scope



The current implementation uses the HAI industrial telemetry dataset to investigate two anomaly-detection approaches:



\- \*\*Isolation Forest:\*\* An unsupervised machine-learning baseline.

\- \*\*Autoencoder:\*\* A neural network that learns to reconstruct input features and identifies records with high reconstruction error.



This guide covers the implemented OT experiments only. IT detection, full Zero Trust integration, IT-OT correlation, and the unified dashboard are outside the current implementation scope.



\## 2. Prerequisites



\- Windows operating system.

\- Python 3.11.

\- Git.

\- Sufficient disk space and memory for dataset processing and model training.

\- Access to the HAI dataset and its applicable usage terms.



\## 3. Clone the Repository



```powershell

git clone https://github.com/rakshitabh/ResiNet-IDAM.git

cd ResiNet-IDAM

```



\## 4. Create a Python Environment



Create a virtual environment using Python 3.11:



```powershell

py -3.11 -m venv .venv

```



Activate it:



```powershell

.\\.venv\\Scripts\\Activate.ps1

```



If PowerShell prevents activation, follow your organization's security policy or activate the environment using the appropriate Windows command prompt instead.



Verify the Python version:



```powershell

python --version

```



The expected major and minor version is Python 3.11.



\## 5. Install Dependencies



Upgrade pip:



```powershell

python -m pip install --upgrade pip

```



Install the project dependencies:



```powershell

python -m pip install -r requirements.txt

```



The current dependency list includes pandas, NumPy, scikit-learn, joblib, and TensorFlow.



\## 6. Dataset Requirements



The implementation expects the following original HAI files:



```text

data/

└── raw/

&#x20;   ├── hai-train1.txt

&#x20;   ├── hai-test1.txt

&#x20;   └── label-test1.txt

```



The original dataset is not included in the Git repository. Obtain it separately from an authorized source and follow its applicable licensing and redistribution conditions.



Create the directories if necessary:



```powershell

New-Item -ItemType Directory -Force data\\raw

New-Item -ItemType Directory -Force data\\processed

```



Place the original dataset files under `data/raw/`.



\*\*Important:\*\* The repository excludes raw and processed datasets from ordinary Git commits. A fresh clone therefore requires the original data before the preprocessing scripts can run.



\## 7. Prepare the Data



Run these commands from the project root, in order:



```powershell

python src/prepare\_training\_data.py

python src/prepare\_test\_data.py

python src/scale\_features.py

```



The scripts prepare the training and test data and generate processed feature files.



Expected data products used by the experiments include:



```text

data/processed/

├── hai-train-features.csv

├── hai-train-scaled.csv

├── hai-test-labeled.csv

├── hai-test-scaled.csv

└── feature\_distribution\_report.csv

```



The exact outputs depend on the paths configured in the scripts. Inspect each script if the files are not generated at the expected locations.



\## 8. Train and Evaluate Isolation Forest



Train the baseline:



```powershell

python src/train\_baseline.py

```



Evaluate it:



```powershell

python src/evaluate\_baseline.py

```



The training script generates a model under `models/`, and the evaluation script uses the trained model and prepared test data.



Check the configured model and dataset paths if either script reports that a file is missing.



\## 9. Train and Evaluate the Autoencoder



Train the autoencoder:



```powershell

python src/train\_autoencoder.py

```



Evaluate it:



```powershell

python src/evaluate\_autoencoder.py

```



The current training implementation saves the following artifacts:



```text

models/

├── ot\_autoencoder.keras

└── autoencoder\_threshold.json

```



The threshold is derived from reconstruction errors on a chronological validation split. The evaluation script compares reconstruction errors from the test data against the saved threshold.



Training may take time depending on the hardware and TensorFlow configuration.



\## 10. Preliminary Experimental Results



The following results were obtained in the current experiments on the labeled HAI test set.



| Metric | Isolation Forest | Autoencoder |

|---|---:|---:|

| Attack precision | 0.1049 | 0.3492 |

| Attack recall | 0.6880 | 0.4639 |

| Attack F1-score | 0.1820 | 0.3984 |

| False positives | 17,506 | 2,578 |

| False negatives | 930 | 1,598 |



In these experiments, the autoencoder achieved a higher attack F1-score and fewer false positives, but it missed more attacks.



These results are preliminary. The models used different training setups, so a controlled comparison is needed before drawing definitive conclusions.



\## 11. Important Evaluation Practices



\- Do not train models using the labeled test set.

\- Do not select detection thresholds using test labels.

\- Fit preprocessing transformations on training data and reuse them during evaluation.

\- Ensure that training and test feature columns have the same names and order.

\- Report both false positives and false negatives alongside precision, recall, and F1-score.

\- Avoid describing the detector as a validated zero-day detector without appropriate experiments using previously unseen attack scenarios.



\## 12. Troubleshooting



\### Python 3.11 is not found



Check available Python installations:



```powershell

py -0p

```



Install or select a compatible Python 3.11 interpreter if necessary.



\### A dataset file is missing



Verify that the required original files exist under `data/raw/` and that the script input paths match the actual filenames.



\### A processed CSV file is missing



Run the preprocessing scripts in the order given in Section 7. Check their configured output paths.



\### A trained model is missing



Run the relevant training script before running its evaluation script.



\### TensorFlow installation or import fails



Verify that the virtual environment is active and that the interpreter is Python 3.11:



```powershell

python --version

python -m pip show tensorflow

```



Consult the TensorFlow installation documentation if installation continues to fail.



\## 13. Current Limitations



This implementation is an initial OT anomaly-detection component. It does not yet implement the complete ResiNet-IDAM framework.



The following remain separate integration tasks:



\- IT intrusion detection.

\- Unified IT-OT event correlation.

\- Zero Trust identity and contextual risk assessment.

\- Adaptive response and model updating.

\- Unified monitoring dashboard.

\- Controlled validation against previously unseen attacks.



The HAI benchmark results should not be interpreted as proof of production readiness or generalization to all zero-day attacks.



