# Battery Aging Cross-Chemistry Benchmarking Framework

## 📌 Overview

This repository contains the implementation of a **cross-chemistry benchmarking framework** for battery aging analysis. The framework enables fair comparison of degradation behavior across different battery chemistries (Li-ion, Na-ion, Zn-ion) by normalizing cycling data onto a common effective throughput axis.

The framework was developed as part of a summer research project investigating aging mechanisms across multiple battery chemistries using the BatteryLife dataset.

### Key Features

- **Cross-Chemistry Comparison**: Compare aging behavior across Li-ion (LFP, NCA, NMC), Na-ion, and Zn-ion batteries
- **Physics-Based Model**: Semi-empirical degradation model with Arrhenius temperature acceleration and C-rate power law
- **Parameter Identifiability**: Bootstrap resampling with 95% confidence intervals
- **Comprehensive Evaluation**: MAE, RMSE, N80 (cycles to 80% SOH), and degradation rates
- **Visualization Tools**: Test coverage heatmaps, correlation matrices, and matched-condition plots

---

## 📊 Framework Equations

### State of Health (SOH)

$$SOH(N) = \frac{Q(N)}{Q_0}$$

### Normalized Throughput

$$Ah_{norm} = \Delta N \times DoD$$

### Temperature Acceleration (Arrhenius)

$$\Phi_T = \exp\left[\frac{E_a}{R}\left(\frac{1}{T_{ref}} - \frac{1}{T}\right)\right]$$

### C-rate Acceleration (Power Law)

$$\Phi_C = \left(\frac{C}{C_{ref}}\right)^{\beta}$$

### Effective Throughput

$$Ah_{eff} = Ah_{norm} \times \Phi_T \times \Phi_C$$

### Degradation Law

$$SOH = 1 - k \times Ah_{eff}^{b}$$

## 📊 Performance Metrics

The following metrics are used to evaluate model performance:

### Error Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **MAE** | $\frac{1}{n}\sum_{i=1}^{n} \|y_i - \hat{y}_i\|$ | Mean Absolute Error — average magnitude of prediction errors (treats all errors equally) |
| **RMSE** | $\sqrt{\frac{1}{n}\sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$ | Root Mean Square Error — heavily penalizes large errors (outlier-sensitive) |

### Battery-Specific Metrics

| Metric | Description |
|--------|-------------|
| **$N_{80}$** | Predicted number of cycles to reach 80% State of Health (end-of-life) |
| **$\frac{dSOH}{dAh_{eff}}$** | Intrinsic degradation rate — chemistry-level degradation per unit effective throughput |
| **$\frac{dSOH}{dN}$** | Application-facing degradation rate — condition-dependent degradation per cycle |

### Identifiability Assessment

| Metric | Description |
|--------|-------------|
| **95% Confidence Interval** | Range containing the true parameter value with 95% probability (from bootstrap resampling) |

### Interpretation Guide

| MAE / RMSE | Interpretation |
|------------|----------------|
| `< 0.02` | Excellent fit |
| `0.02 – 0.05` | Good fit |
| `0.05 – 0.10` | Moderate fit |
| `> 0.10` | Poor fit (investigate) |

# Installation & Reproducibility Guide

This guide provides instructions for setting up the environment and reproducing the results from the **Battery Aging Cross-Chemistry Benchmarking Framework**.

---

## 📋 Prerequisites

Before you begin, ensure you have the following installed:

- **Python** (Version 3.9 or higher)
- **Git** (to clone the repository)
- A code editor (like **VS Code**) is recommended for viewing and running the scripts.

---

## Step 1: Clone the Repository

Open your terminal or command prompt and run:

```bash
git clone https://github.com/jzh672/BatteryBenchmarkingFramework.git
cd BatteryBenchmarkingFramework
```
---

## Step 2: Install Dependencies

Install all required packages using the requirements.txt file provided in the repository.

```bash
pip install --upgrade pip
pip install -r requirements.txt
```
---

## Step 3: Prepare Your Dataset

Download the BatteryLife Dataset from Zenodo: https://zenodo.org/records/17958489

Note: It's recommended to use the preprocessed dataset provided in this repo as the preprocessing only adapts to BatteryLife Datasets and requires manual entries for temperature columns. But to run the preprocessing by yourself:

### Update File Paths
Before running the scripts, update the file paths in the code to point to your data location. Open the main Python scripts (e.g., preprocessing.py) and modify the input_folder and output_folder variables:

```python
# Example from preprocessing.py
input_folder = r'path/to/your/data/raw/'   # e.g., './data/raw/'
output_folder = r'path/to/your/data/processed/'
```

### Run the Preprocessing
Convert raw .pkl files into structured .csv files:
```bash
python Implementation Script/preprocessing.py
```

The script will process all files in your input folder and save the combined data as all_data_combined.csv in your specified output folder.

## Step 4: Run the Main Analysis
Execute the main script to perform Exploratory Data Analysis (EDA), model fitting, evaluation, and generate all visualizations:
```bash
python Implementation Script/combined_implementation.py
```
This will produce the figures and results discussed in the poster and the project report.
