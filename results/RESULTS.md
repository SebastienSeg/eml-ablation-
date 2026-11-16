# Résultats (généré automatiquement par `main.py`)

Seeds : 15 · epochs : 3000 · hidden réf. : 24 · depth : 3 · device : cpu

Métrique : NMSE test = MSE / Var_train(y) sur données standardisées (1.0 ≈ prédire la moyenne). Médiane sur les seeds, runs divergents comptés comme +inf. **Gras** = meilleure médiane de la ligne.

## Verdict global : hypothèse **RÉFUTÉE**

Cibles favorables gagnées : **0/12** (0%) · contrôles négatifs gagnés : 1/4 (25%).

Règle fixée avant les runs : une cible est *gagnée* si au moins un membre de la famille EML (EML, EML-v2, EML→ReLU) est significativement meilleur (Mann-Whitney bilatéral, BH sur tous les tests) que **chacun** des comparateurs (EML-noexp, EML-nolog, exp-only, log-only, ReLU-MLP, KAN). Hypothèse réfutée si moins de 25% des cibles favorables sont gagnées, soutenue si au moins 50% le sont et plus que de contrôles, non concluante sinon.

| Cible | Famille | Meilleur EML | NMSE | Meilleur comparateur | NMSE | ratio | Gagnée |
|---|---|---|---:|---|---:|---:|---|
| `explog_product` | in-domain (favorable) | EML-v2 | 3.73e-05 | exp-only | 3.79e-05 | +0.01 | non |
| `power_function` | in-domain (favorable) | EML-v2 | 1.30e-05 | exp-only | 1.18e-05 | -0.04 | non |
| `logistic_log` | in-domain (favorable) | EML-v2 | 7.20e-04 | EML-nolog | 5.49e-04 | -0.12 | non |
| `cross_entropy` | in-domain (favorable) | EML-v2 | 2.54e-05 | exp-only | 2.95e-05 | +0.07 | non |
| `polynomial` | ood (contrôle) | EML-v2 | 5.38e-05 | exp-only | 5.99e-05 | +0.05 | non |
| `trigonometric` | ood (contrôle) | EML-v2 | 2.31e-04 | exp-only | 3.05e-04 | +0.12 | **oui** (EML-v2) |
| `piecewise` | ood (contrôle) | EML-v2 | 1.64e-03 | exp-only | 1.69e-03 | +0.01 | non |
| `log_ratio` | log-essential (favorable) | EML→ReLU | 4.92e-03 | ReLU-MLP | 3.14e-03 | -0.19 | non |
| `product_ratio` | log-essential (favorable) | EML-v2 | 6.86e-05 | exp-only | 5.26e-05 | -0.12 | non |
| `power_law` | log-essential (favorable) | EML-v2 | 2.38e-05 | EML-nolog | 2.31e-05 | -0.01 | non |
| `extrap_log_ratio` | extrapolation (favorable) | EML→ReLU | 1.66e-01 | ReLU-MLP | 1.19e-02 | -1.15 | non |
| `extrap_power` | extrapolation (favorable) | EML | 1.61e+02 | ReLU-MLP | 1.53e+02 | -0.02 | non |
| `extrap_product` | extrapolation (favorable) | EML-v2 | 6.33e-01 | ReLU-MLP | 5.39e-01 | -0.07 | non |
| `extrap_decay` | extrapolation (favorable) | EML-v2 | 1.14e-02 | EML-nolog | 1.05e-02 | -0.04 | non |
| `extrap_polynomial` | extrapolation-control (contrôle) | EML | 2.23e+02 | exp-only | 1.87e+02 | -0.08 | non |
| `radioactive_decay` | radioactive (favorable) | EML-v2 | 5.24e-04 | EML-nolog | 4.70e-04 | -0.05 | non |

ratio = log10(NMSE meilleur comparateur / NMSE meilleur EML) : > 0 ⇔ la famille EML a la meilleure médiane.

## in-domain

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `explog_product` | 1.76e-03 | 1.46e-02 | 7.33e-05 | 3.79e-05 | 2.83e-02 | 1.55e-04 | 1.91e-04 | **3.73e-05** | 2.46e-04 |
| `power_function` | 1.06e-03 | 8.39e-01 | 1.79e-05 | **1.18e-05** | 1.75e-02 | 7.38e-05 | 4.99e-05 | 1.30e-05 | 4.26e-05 |
| `logistic_log` | 2.38e-02 | 1.06e+00 | **5.49e-04** | 7.76e-04 | 1.69e-01 | 3.46e-03 | 3.79e-03 | 7.20e-04 | 1.19e-03 |
| `cross_entropy` | 2.00e-03 | 8.32e-03 | 5.11e-05 | 2.95e-05 | 4.08e-02 | 1.22e-04 | 1.65e-04 | **2.54e-05** | 1.36e-04 |

## log-essential

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `log_ratio` | 3.46e-02 | 2.08e-01 | 1.56e-02 | 1.75e-02 | 8.32e-02 | **3.14e-03** | 8.17e-03 | 1.94e-02 | 4.92e-03 |
| `product_ratio` | 3.70e-03 | 5.40e-01 | 5.72e-05 | **5.26e-05** | 1.12e-01 | 4.79e-04 | 2.33e-04 | 6.86e-05 | 1.97e-04 |
| `power_law` | 1.56e-03 | 1.37e-01 | **2.31e-05** | 2.73e-05 | 2.46e-02 | 8.87e-05 | 8.87e-05 | 2.38e-05 | 6.61e-05 |

## extrapolation

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `extrap_log_ratio` | 2.90e-01 | 7.12e-01 | 9.34e-02 | 9.91e-02 | 5.82e-01 | **1.19e-02** | 1.89e+00 | 1.87e-01 | 1.66e-01 |
| `extrap_power` | 1.61e+02 | 2.29e+02 | 1.59e+02 | 1.55e+02 | 2.08e+02 | **1.53e+02** | 2.06e+02 | 1.65e+02 | 1.63e+02 |
| `extrap_product` | 1.67e+00 | 5.07e+00 | 8.67e-01 | 6.45e-01 | 3.53e+00 | **5.39e-01** | 7.47e+00 | 6.33e-01 | 9.68e-01 |
| `extrap_decay` | 4.93e-02 | 3.26e+00 | **1.05e-02** | 1.19e-02 | 2.42e-01 | 1.98e-02 | 8.70e-01 | 1.14e-02 | 1.52e-02 |

## radioactive

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `radioactive_decay` | 3.29e-03 | 7.72e-01 | **4.70e-04** | 5.95e-04 | 3.28e-02 | 2.96e-03 | 5.16e-03 | 5.24e-04 | 9.93e-04 |

## ood

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `polynomial` | 1.90e-03 | 1.08e+00 | 7.24e-05 | 5.99e-05 | 2.24e-02 | 2.02e-04 | 2.41e-04 | **5.38e-05** | 3.44e-04 |
| `trigonometric` | 1.46e-02 | 1.10e+00 | 3.93e-04 | 3.05e-04 | 9.47e-02 | 5.18e-04 | 5.45e-04 | **2.31e-04** | 8.86e-04 |
| `piecewise` | 5.11e-03 | 8.09e-01 | 2.15e-03 | 1.69e-03 | 4.56e-02 | 3.21e-03 | 3.80e-03 | **1.64e-03** | 1.72e-03 |

## extrapolation-control

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `extrap_polynomial` | 2.23e+02 | 3.34e+02 | 1.89e+02 | **1.87e+02** | 2.92e+02 | — | — | — | — |

## Tests : EML vs chaque variant

Mann-Whitney bilatéral, correction BH. r > 0 ⇔ EML meilleur. ratio = log10(médiane autre / médiane EML).

| Cible | Variant | p (BH) | r | ratio | Verdict EML |
|---|---|---:|---:|---:|---|
| `explog_product` | EML-noexp | 5.26e-06 | +1.00 | +0.92 | better |
| `explog_product` | EML-nolog | 5.26e-06 | -1.00 | -1.38 | worse |
| `explog_product` | exp-only | 5.26e-06 | -1.00 | -1.67 | worse |
| `explog_product` | log-only | 5.26e-06 | +1.00 | +1.21 | better |
| `explog_product` | ReLU-MLP | 5.26e-06 | -1.00 | -1.06 | worse |
| `explog_product` | KAN | 5.26e-06 | -1.00 | -0.96 | worse |
| `explog_product` | EML-v2 | 5.26e-06 | -1.00 | -1.67 | worse |
| `explog_product` | EML→ReLU | 5.26e-06 | -1.00 | -0.86 | worse |
| `power_function` | EML-noexp | 5.26e-06 | +1.00 | +2.90 | better |
| `power_function` | EML-nolog | 5.26e-06 | -1.00 | -1.77 | worse |
| `power_function` | exp-only | 5.26e-06 | -1.00 | -1.95 | worse |
| `power_function` | log-only | 5.26e-06 | +1.00 | +1.22 | better |
| `power_function` | ReLU-MLP | 5.26e-06 | -1.00 | -1.16 | worse |
| `power_function` | KAN | 5.26e-06 | -1.00 | -1.33 | worse |
| `power_function` | EML-v2 | 5.26e-06 | -1.00 | -1.91 | worse |
| `power_function` | EML→ReLU | 5.26e-06 | -1.00 | -1.40 | worse |
| `logistic_log` | EML-noexp | 5.26e-06 | +1.00 | +1.65 | better |
| `logistic_log` | EML-nolog | 5.26e-06 | -1.00 | -1.64 | worse |
| `logistic_log` | exp-only | 5.26e-06 | -1.00 | -1.49 | worse |
| `logistic_log` | log-only | 5.26e-06 | +1.00 | +0.85 | better |
| `logistic_log` | ReLU-MLP | 1.07e-05 | -0.96 | -0.84 | worse |
| `logistic_log` | KAN | 3.77e-05 | -0.90 | -0.80 | worse |
| `logistic_log` | EML-v2 | 5.26e-06 | -1.00 | -1.52 | worse |
| `logistic_log` | EML→ReLU | 5.26e-06 | -1.00 | -1.30 | worse |
| `cross_entropy` | EML-noexp | 5.26e-06 | +1.00 | +0.62 | better |
| `cross_entropy` | EML-nolog | 5.26e-06 | -1.00 | -1.59 | worse |
| `cross_entropy` | exp-only | 5.26e-06 | -1.00 | -1.83 | worse |
| `cross_entropy` | log-only | 5.26e-06 | +1.00 | +1.31 | better |
| `cross_entropy` | ReLU-MLP | 5.26e-06 | -1.00 | -1.22 | worse |
| `cross_entropy` | KAN | 5.26e-06 | -1.00 | -1.08 | worse |
| `cross_entropy` | EML-v2 | 5.26e-06 | -1.00 | -1.90 | worse |
| `cross_entropy` | EML→ReLU | 5.26e-06 | -1.00 | -1.17 | worse |
| `polynomial` | EML-noexp | 5.26e-06 | +1.00 | +2.76 | better |
| `polynomial` | EML-nolog | 5.26e-06 | -1.00 | -1.42 | worse |
| `polynomial` | exp-only | 5.26e-06 | -1.00 | -1.50 | worse |
| `polynomial` | log-only | 5.26e-06 | +1.00 | +1.07 | better |
| `polynomial` | ReLU-MLP | 5.26e-06 | -1.00 | -0.97 | worse |
| `polynomial` | KAN | 5.26e-06 | -1.00 | -0.90 | worse |
| `polynomial` | EML-v2 | 5.26e-06 | -1.00 | -1.55 | worse |
| `polynomial` | EML→ReLU | 5.26e-06 | -1.00 | -0.74 | worse |
| `trigonometric` | EML-noexp | 5.26e-06 | +1.00 | +1.88 | better |
| `trigonometric` | EML-nolog | 5.26e-06 | -1.00 | -1.57 | worse |
| `trigonometric` | exp-only | 5.26e-06 | -1.00 | -1.68 | worse |
| `trigonometric` | log-only | 5.26e-06 | +1.00 | +0.81 | better |
| `trigonometric` | ReLU-MLP | 5.26e-06 | -1.00 | -1.45 | worse |
| `trigonometric` | KAN | 5.26e-06 | -1.00 | -1.43 | worse |
| `trigonometric` | EML-v2 | 5.26e-06 | -1.00 | -1.80 | worse |
| `trigonometric` | EML→ReLU | 5.26e-06 | -1.00 | -1.22 | worse |
| `piecewise` | EML-noexp | 5.26e-06 | +1.00 | +2.20 | better |
| `piecewise` | EML-nolog | 1.07e-05 | -0.96 | -0.38 | worse |
| `piecewise` | exp-only | 5.26e-06 | -1.00 | -0.48 | worse |
| `piecewise` | log-only | 5.26e-06 | +1.00 | +0.95 | better |
| `piecewise` | ReLU-MLP | 0.0275 | -0.48 | -0.20 | worse |
| `piecewise` | KAN | 0.105 | -0.36 | -0.13 | n.s. |
| `piecewise` | EML-v2 | 9.19e-06 | -0.97 | -0.49 | worse |
| `piecewise` | EML→ReLU | 6.34e-05 | -0.88 | -0.47 | worse |
| `log_ratio` | EML-noexp | 5.26e-06 | +1.00 | +0.78 | better |
| `log_ratio` | EML-nolog | 9.19e-06 | -0.97 | -0.35 | worse |
| `log_ratio` | exp-only | 5.26e-06 | -1.00 | -0.30 | worse |
| `log_ratio` | log-only | 5.26e-06 | +1.00 | +0.38 | better |
| `log_ratio` | ReLU-MLP | 5.26e-06 | -1.00 | -1.04 | worse |
| `log_ratio` | KAN | 5.26e-06 | -1.00 | -0.63 | worse |
| `log_ratio` | EML-v2 | 1.28e-05 | -0.96 | -0.25 | worse |
| `log_ratio` | EML→ReLU | 5.26e-06 | -1.00 | -0.85 | worse |
| `product_ratio` | EML-noexp | 5.26e-06 | +1.00 | +2.16 | better |
| `product_ratio` | EML-nolog | 5.26e-06 | -1.00 | -1.81 | worse |
| `product_ratio` | exp-only | 5.26e-06 | -1.00 | -1.85 | worse |
| `product_ratio` | log-only | 5.26e-06 | +1.00 | +1.48 | better |
| `product_ratio` | ReLU-MLP | 5.26e-06 | -1.00 | -0.89 | worse |
| `product_ratio` | KAN | 5.26e-06 | -1.00 | -1.20 | worse |
| `product_ratio` | EML-v2 | 5.26e-06 | -1.00 | -1.73 | worse |
| `product_ratio` | EML→ReLU | 5.26e-06 | -1.00 | -1.27 | worse |
| `power_law` | EML-noexp | 5.26e-06 | +1.00 | +1.94 | better |
| `power_law` | EML-nolog | 5.26e-06 | -1.00 | -1.83 | worse |
| `power_law` | exp-only | 5.26e-06 | -1.00 | -1.76 | worse |
| `power_law` | log-only | 5.26e-06 | +1.00 | +1.20 | better |
| `power_law` | ReLU-MLP | 5.26e-06 | -1.00 | -1.25 | worse |
| `power_law` | KAN | 5.26e-06 | -1.00 | -1.25 | worse |
| `power_law` | EML-v2 | 5.26e-06 | -1.00 | -1.82 | worse |
| `power_law` | EML→ReLU | 5.26e-06 | -1.00 | -1.37 | worse |
| `extrap_log_ratio` | EML-noexp | 0.0114 | +0.56 | +0.39 | better |
| `extrap_log_ratio` | EML-nolog | 0.00112 | -0.72 | -0.49 | worse |
| `extrap_log_ratio` | exp-only | 0.00291 | -0.65 | -0.47 | worse |
| `extrap_log_ratio` | log-only | 0.195 | +0.29 | +0.30 | n.s. |
| `extrap_log_ratio` | ReLU-MLP | 1.07e-05 | -0.96 | -1.39 | worse |
| `extrap_log_ratio` | KAN | 2.21e-05 | +0.93 | +0.81 | better |
| `extrap_log_ratio` | EML-v2 | 0.543 | -0.14 | -0.19 | n.s. |
| `extrap_log_ratio` | EML→ReLU | 0.145 | -0.32 | -0.24 | n.s. |
| `extrap_power` | EML-noexp | 6.34e-06 | +0.99 | +0.15 | better |
| `extrap_power` | EML-nolog | 0.42 | -0.18 | -0.01 | n.s. |
| `extrap_power` | exp-only | 0.934 | -0.02 | -0.02 | n.s. |
| `extrap_power` | log-only | 0.000244 | +0.80 | +0.11 | better |
| `extrap_power` | ReLU-MLP | 0.256 | -0.25 | -0.02 | n.s. |
| `extrap_power` | KAN | 3.77e-05 | +0.90 | +0.11 | better |
| `extrap_power` | EML-v2 | 0.543 | +0.14 | +0.01 | n.s. |
| `extrap_power` | EML→ReLU | 0.81 | +0.06 | +0.00 | n.s. |
| `extrap_product` | EML-noexp | 1.07e-05 | +0.96 | +0.48 | better |
| `extrap_product` | EML-nolog | 0.00903 | -0.57 | -0.29 | worse |
| `extrap_product` | exp-only | 0.00148 | -0.70 | -0.41 | worse |
| `extrap_product` | log-only | 0.00291 | +0.65 | +0.32 | better |
| `extrap_product` | ReLU-MLP | 0.00015 | -0.83 | -0.49 | worse |
| `extrap_product` | KAN | 5.26e-06 | +1.00 | +0.65 | better |
| `extrap_product` | EML-v2 | 0.000977 | -0.72 | -0.42 | worse |
| `extrap_product` | EML→ReLU | 0.0275 | -0.48 | -0.24 | worse |
| `extrap_decay` | EML-noexp | 5.26e-06 | +1.00 | +1.82 | better |
| `extrap_decay` | EML-nolog | 0.00054 | -0.76 | -0.67 | worse |
| `extrap_decay` | exp-only | 0.000977 | -0.72 | -0.62 | worse |
| `extrap_decay` | log-only | 0.00129 | +0.71 | +0.69 | better |
| `extrap_decay` | ReLU-MLP | 0.018 | -0.52 | -0.40 | worse |
| `extrap_decay` | KAN | 5.36e-05 | +0.88 | +1.25 | better |
| `extrap_decay` | EML-v2 | 0.00903 | -0.57 | -0.64 | worse |
| `extrap_decay` | EML→ReLU | 0.0304 | -0.48 | -0.51 | worse |
| `extrap_polynomial` | EML-noexp | 5.26e-06 | +1.00 | +0.18 | better |
| `extrap_polynomial` | EML-nolog | 0.00638 | -0.60 | -0.07 | worse |
| `extrap_polynomial` | exp-only | 0.00813 | -0.58 | -0.08 | worse |
| `extrap_polynomial` | log-only | 2.19e-05 | +0.99 | +0.12 | better |
| `radioactive_decay` | EML-noexp | 5.26e-06 | +1.00 | +2.37 | better |
| `radioactive_decay` | EML-nolog | 5.26e-06 | -1.00 | -0.85 | worse |
| `radioactive_decay` | exp-only | 5.26e-06 | -1.00 | -0.74 | worse |
| `radioactive_decay` | log-only | 5.26e-06 | +1.00 | +1.00 | better |
| `radioactive_decay` | ReLU-MLP | 0.241 | -0.26 | -0.05 | n.s. |
| `radioactive_decay` | KAN | 0.00291 | +0.65 | +0.20 | better |
| `radioactive_decay` | EML-v2 | 5.26e-06 | -1.00 | -0.80 | worse |
| `radioactive_decay` | EML→ReLU | 5.26e-06 | -1.00 | -0.52 | worse |

Bilan : EML significativement meilleur dans 36/124 comparaisons, significativement pire dans 78/124, non significatif dans 10/124.
