# Distribution-Free Uncertainty Quantification in Machine Learning: A Theoretical and Empirical Comparison in R

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![R 4.6+](https://img.shields.io/badge/R-4.6%2B-blue.svg)](https://www.r-project.org/)
[![Reproducibility](https://img.shields.io/badge/Reproducibility-Verified-brightgreen.svg)]()

This repository contains the complete, reproducible simulation code, empirical benchmarks, and analysis scripts accompanying the research paper:

> **"Distribution-Free Uncertainty Quantification in Machine Learning: A Theoretical and Empirical Comparison of Conformal Prediction, Bayesian Credible Intervals, and Parametric Bootstrap in R"**  
> **Authors:** Yuvraj Chandak (Reg. No.: `25BCE2561`, School of Computer Science and Engineering) & Dr. Veeramani S (Department of Mathematics, School of Advanced Sciences)  
> **Affiliation:** Vellore Institute of Technology (VIT), Vellore, Tamil Nadu, India  
> **Repository:** [https://github.com/YuvrajC7/distribution-free-uncertainty-quantification](https://github.com/YuvrajC7/distribution-free-uncertainty-quantification)

---

## Overview

Modern machine learning models deliver accurate point forecasts but lack built-in guarantees on prediction variance, tail risk, and confidence bounds. Classical uncertainty quantification (UQ) techniques rely heavily on restrictive distributional assumptions:
1. **Ordinary Least Squares (OLS) Prediction Intervals:** Assumes homoscedastic, Gaussian disturbances and correct likelihood specification.
2. **Parametric Bootstrap Resampling:** Assumes valid model specification and parameterizes errors using estimated residual standard deviation.
3. **Bayesian Conjugate Normal-Inverse-Gamma (NIG):** Yields Student-$t$ posterior predictive intervals under normal likelihood priors.

Under distribution shift, heavy tails (Student-$t_3$), or heteroscedastic noise ($\sigma(X_1) = 1 + 2|X_1|$), classical parametric methods suffer from severe coverage bias ($92.6\%–93.0\%$ at nominal $90\%$, Student's $t$-test $p < 10^{-10}$), inflated by outlier-induced variance distortion.

In contrast, **Distribution-Free Conformal Prediction** guarantees finite-sample marginal coverage:
$$\mathbb{P}\big(Y_{n+1} \in \hat{C}_n(X_{n+1})\big) \ge 1 - \alpha$$
under the solitary, minimal assumption of **exchangeability**.

This repository evaluates:
- **Split Conformal Prediction** (Random Forests base learner via `ranger`)
- **Conformalized Quantile Regression (CQR)** (Quantile Random Forests base learner via `ranger`)
- **Classical OLS Prediction Intervals** (`stats::lm`)
- **Parametric Residual Bootstrap** (vectorized $B = 250$ resamples)
- **Bayesian Normal-Inverse-Gamma Credible Intervals** (analytical Student-$t$ predictive density)

---

## Repository Structure

```
distribution-free-uq-r/
├── README.md                      # Comprehensive project documentation
├── .gitignore                     # Git ignore rules for R environments
├── run_experiments.bat            # Windows one-click batch runner
├── simulation_experiments.R       # Complete standalone Monte Carlo simulation script
├── paper/                         # Academic manuscript and compiled IEEE PDF
│   ├── research_paper.md          # Full academic paper in Markdown format
│   └── Distribution_Free_UQ_Machine_Learning.pdf # Compiled IEEE 8-page publication PDF
├── figures/                       # High-resolution output figures (300 DPI)
│   ├── fig1_prediction_bands.png
│   ├── fig2_coverage_distribution.png
│   └── fig3_winkler_calibration_curve.png
└── results/                       # Processed summary CSV tables
    ├── summary_regime1_gaussian.csv
    ├── summary_regime2_student.csv
    ├── summary_regime3_hetero.csv
    └── summary_boston_benchmark.csv
```

---

## System Requirements & Prerequisites

### R Environment
- R version $\ge 4.0.0$ (Tested on R 4.6.1 on Windows 11 x64).
- Required CRAN packages:
  - `ranger` ($\ge 0.16.0$) — Fast Random Forests and Quantile Random Forests
  - `ggplot2` ($\ge 3.5.0$) — Visualization and figure generation
  - `MASS` — Benchmark datasets (`MASS::Boston`)
  - `splines` — Polynomial basis expansion utilities

To install all dependencies in R:
```r
install.packages(c("ranger", "ggplot2", "MASS", "splines"))
```

---

## Reproduction Instructions

### 1. Run via Windows Batch Script (Recommended)
Double-click `run_experiments.bat` or run from PowerShell / Command Prompt:
```cmd
run_experiments.bat
```

### 2. Run via Rscript Command Line
```cmd
Rscript simulation_experiments.R
```

### 3. Run Interactively in R / RStudio
Open `simulation_experiments.R` in RStudio or R console and run:
```r
source("simulation_experiments.R")
```

The script will:
1. Initialize RNG with fixed seed `set.seed(2026)`.
2. Execute $M = 100$ independent Monte Carlo replications across:
   - **Regime I:** Homoscedastic Gaussian ($\epsilon \sim \mathcal{N}(0, 1)$).
   - **Regime II:** Heavy-Tailed Student-$t$ ($\epsilon \sim t_{\nu=3}$, infinite kurtosis).
   - **Regime III:** Heteroscedastic Skewed ($\epsilon = (1 + 2|X_1|)(\text{Exp}(1) - 1)$).
3. Execute $B = 50$ random splits on `MASS::Boston`.
4. Perform replication-level two-sided Student's $t$-tests ($99$ and $49$ df) against nominal $90\%$ coverage.
5. Perform paired Wilcoxon signed-rank tests evaluating proper Winkler score efficiency.
6. Export processed summary tables to CSV and render high-resolution PNG figures to `figures/`.

---

## Key Empirical Findings

| Experiment / Regime | Method | Coverage (ECR) | Mean Width (MPIW) | Winkler ($S_\alpha$) | $t$-Statistic ($H_0: p=0.90$) | Coverage $p$-value |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Regime I** (Gaussian) | Parametric OLS | $90.04\% \pm 1.57\%$ | $3.327 \pm 0.070$ | $4.151 \pm 0.187$ | $+0.24$ | $0.8121$ |
| | Split Conformal | $90.18\% \pm 2.12\%$ | $3.606 \pm 0.171$ | $4.488 \pm 0.197$ | $+0.84$ | $0.4008$ |
| | CQR | $90.11\% \pm 2.08\%$ | $3.894 \pm 0.177$ | $4.830 \pm 0.210$ | $+0.55$ | $0.5832$ |
| **Regime II** (Student-$t_3$) | Parametric OLS | $92.94\% \pm 2.37\%$ | $5.729 \pm 1.011$ | $8.082 \pm 1.284$ | $+12.40$ | $< 10^{-15}$ |
| | Split Conformal | $89.87\% \pm 2.18\%$ | $5.131 \pm 0.361$ | $8.149 \pm 1.197$ | $-0.61$ | $0.5420$ |
| | CQR | $89.75\% \pm 2.10\%$ | $5.805 \pm 0.357$ | $8.760 \pm 1.212$ | $-1.19$ | $0.2360$ |
| **Regime III** (Heteroscedastic) | Parametric OLS | $92.95\% \pm 1.50\%$ | $6.827 \pm 0.349$ | $10.096 \pm 0.856$ | $+19.72$ | $< 10^{-15}$ |
| | Split Conformal | $89.87\% \pm 2.13\%$ | $6.315 \pm 0.409$ | $10.312 \pm 0.928$ | $-0.60$ | $0.5500$ |
| | CQR | $89.65\% \pm 2.34\%$ | $6.350 \pm 0.353$ | **$9.756 \pm 0.897$** | $-1.49$ | $0.1382$ |
| **Boston Housing** ($B=50$) | Parametric OLS | $93.24\% \pm 2.27\%$ | $15.900 \pm 0.700$ | $22.940 \pm 3.501$ | $+10.12$ | $1.36 \times 10^{-13}$ |
| | Split Conformal | $89.67\% \pm 3.59\%$ | **$10.686 \pm 1.796$** | $18.813 \pm 3.889$ | $-0.65$ | $0.5180$ |
| | CQR | $91.46\% \pm 3.78\%$ | $13.572 \pm 1.022$ | **$16.397 \pm 2.277$** | $+2.74$ | $0.0085$ |

- **Exact Finite-Sample Validity:** Conformal methods show no statistically significant divergence from nominal $90\%$ across all synthetic regimes ($p \ge 0.138$).
- **Winkler Efficiency:** Under heteroscedastic noise, CQR reduces proper Winkler scores to $9.756$ ($p = 2.22 \times 10^{-10}$ vs. OLS). On Boston housing, CQR yields a **$28.5\%$ reduction in Winkler score** ($p = 7.79 \times 10^{-10}$) and Split Conformal reduces width by **$32.8\%$** ($p = 1.19 \times 10^{-9}$).

---

## Citation

If you use this codebase or paper in your research, please cite:

```bibtex
@article{chandak2026distributionfree,
  title={Distribution-Free Uncertainty Quantification in Machine Learning: A Theoretical and Empirical Comparison of Conformal Prediction, Bayesian Credible Intervals, and Parametric Bootstrap in R},
  author={Chandak, Yuvraj and Veeramani, S.},
  journal={School of Computer Science and Engineering & School of Advanced Sciences, Vellore Institute of Technology},
  year={2026},
  url={https://github.com/YuvrajC7/distribution-free-uncertainty-quantification}
}
```

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.
