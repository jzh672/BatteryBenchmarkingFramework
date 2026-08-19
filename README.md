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
| **Relative Width** | $\frac{CI_{upper} - CI_{lower}}{mean}$ — normalized measure of parameter uncertainty; values > 0.5 indicate poor identifiability |

### Interpretation Guide

| MAE / RMSE | Interpretation |
|------------|----------------|
| `< 0.02` | Excellent fit |
| `0.02 – 0.05` | Good fit |
| `0.05 – 0.10` | Moderate fit |
| `> 0.10` | Poor fit (investigate) |
