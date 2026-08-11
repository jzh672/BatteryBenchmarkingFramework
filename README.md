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
