# Résultats (généré automatiquement par `main.py`)

Seeds : 5 · epochs : 3000 · hidden réf. : 24 · depth : 3 · device : cpu

Métrique : NMSE test = MSE / Var_train(y) sur données standardisées (1.0 ≈ prédire la moyenne). Médiane sur les seeds, runs divergents comptés comme +inf. **Gras** = meilleure médiane de la ligne.

## In-domain

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `explog_product` | 1.72e-03 | 1.21e-02 | 7.77e-05 | **3.12e-05** | 3.01e-02 | 1.79e-04 | 1.87e-04 | 3.43e-05 | 2.33e-04 |
| `power_function` | 1.10e-03 | 8.09e-01 | 1.35e-05 | 1.02e-05 | 2.66e-02 | 7.18e-05 | 4.88e-05 | **8.21e-06** | 2.75e-05 |
| `logistic_log` | 4.33e-02 | 1.15e+00 | 4.11e-02 | **3.85e-02** | 3.91e-01 | 7.98e-02 | 8.93e-02 | 7.82e-02 | 6.46e-02 |
| `cross_entropy` | 1.99e-03 | 8.56e-03 | 5.90e-05 | **2.59e-05** | 4.73e-02 | 1.05e-04 | 1.69e-04 | 2.86e-05 | 1.30e-04 |

## Out-of-domain

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `polynomial` | 1.88e-03 | 1.08e+00 | 5.68e-05 | **4.01e-05** | 3.75e-02 | 1.82e-04 | 2.29e-04 | 5.68e-05 | 3.20e-04 |
| `trigonometric` | 7.59e-03 | 1.10e+00 | 3.05e-04 | 2.84e-04 | 8.81e-02 | 4.67e-04 | 6.02e-04 | **2.13e-04** | 8.67e-04 |
| `piecewise` | 1.41e-02 | 8.26e-01 | 1.26e-02 | 1.36e-02 | 5.91e-02 | **9.77e-03** | 1.88e-02 | 1.10e-02 | 1.19e-02 |

## Décroissance radioactive

| Cible | EML | EML-noexp | EML-nolog | exp-only | log-only | ReLU-MLP | KAN | EML-v2 | EML→ReLU |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `radioactive_decay` | 2.90e-03 | 8.88e-01 | **4.70e-04** | 5.63e-04 | 3.19e-02 | 2.72e-03 | 4.50e-03 | 5.24e-04 | 9.82e-04 |

## Tests : EML vs chaque variant

Mann-Whitney bilatéral, correction BH. r > 0 ⇔ EML meilleur. ratio = log10(médiane autre / médiane EML).

| Cible | Variant | p (BH) | r | ratio | Verdict EML |
|---|---|---:|---:|---:|---|
| `explog_product` | EML-noexp | 0.00996 | +1.00 | +0.85 | better |
| `explog_product` | EML-nolog | 0.00996 | -1.00 | -1.34 | worse |
| `explog_product` | exp-only | 0.00996 | -1.00 | -1.74 | worse |
| `explog_product` | log-only | 0.00996 | +1.00 | +1.24 | better |
| `explog_product` | ReLU-MLP | 0.00996 | -1.00 | -0.98 | worse |
| `explog_product` | KAN | 0.00996 | -1.00 | -0.96 | worse |
| `explog_product` | EML-v2 | 0.00996 | -1.00 | -1.70 | worse |
| `explog_product` | EML→ReLU | 0.00996 | -1.00 | -0.87 | worse |
| `power_function` | EML-noexp | 0.00996 | +1.00 | +2.86 | better |
| `power_function` | EML-nolog | 0.00996 | -1.00 | -1.91 | worse |
| `power_function` | exp-only | 0.00996 | -1.00 | -2.03 | worse |
| `power_function` | log-only | 0.00996 | +1.00 | +1.38 | better |
| `power_function` | ReLU-MLP | 0.00996 | -1.00 | -1.19 | worse |
| `power_function` | KAN | 0.00996 | -1.00 | -1.36 | worse |
| `power_function` | EML-v2 | 0.00996 | -1.00 | -2.13 | worse |
| `power_function` | EML→ReLU | 0.00996 | -1.00 | -1.60 | worse |
| `logistic_log` | EML-noexp | 0.00996 | +1.00 | +1.43 | better |
| `logistic_log` | EML-nolog | 0.556 | -0.28 | -0.02 | n.s. |
| `logistic_log` | exp-only | 0.245 | -0.52 | -0.05 | n.s. |
| `logistic_log` | log-only | 0.00996 | +1.00 | +0.96 | better |
| `logistic_log` | ReLU-MLP | 0.0376 | +0.84 | +0.27 | better |
| `logistic_log` | KAN | 0.00996 | +1.00 | +0.31 | better |
| `logistic_log` | EML-v2 | 0.0376 | +0.84 | +0.26 | better |
| `logistic_log` | EML→ReLU | 0.109 | +0.68 | +0.17 | n.s. |
| `cross_entropy` | EML-noexp | 0.00996 | +1.00 | +0.63 | better |
| `cross_entropy` | EML-nolog | 0.00996 | -1.00 | -1.53 | worse |
| `cross_entropy` | exp-only | 0.00996 | -1.00 | -1.89 | worse |
| `cross_entropy` | log-only | 0.00996 | +1.00 | +1.38 | better |
| `cross_entropy` | ReLU-MLP | 0.00996 | -1.00 | -1.28 | worse |
| `cross_entropy` | KAN | 0.00996 | -1.00 | -1.07 | worse |
| `cross_entropy` | EML-v2 | 0.00996 | -1.00 | -1.84 | worse |
| `cross_entropy` | EML→ReLU | 0.00996 | -1.00 | -1.19 | worse |
| `polynomial` | EML-noexp | 0.00996 | +1.00 | +2.76 | better |
| `polynomial` | EML-nolog | 0.00996 | -1.00 | -1.52 | worse |
| `polynomial` | exp-only | 0.00996 | -1.00 | -1.67 | worse |
| `polynomial` | log-only | 0.00996 | +1.00 | +1.30 | better |
| `polynomial` | ReLU-MLP | 0.00996 | -1.00 | -1.01 | worse |
| `polynomial` | KAN | 0.00996 | -1.00 | -0.91 | worse |
| `polynomial` | EML-v2 | 0.00996 | -1.00 | -1.52 | worse |
| `polynomial` | EML→ReLU | 0.00996 | -1.00 | -0.77 | worse |
| `trigonometric` | EML-noexp | 0.00996 | +1.00 | +2.16 | better |
| `trigonometric` | EML-nolog | 0.00996 | -1.00 | -1.40 | worse |
| `trigonometric` | exp-only | 0.00996 | -1.00 | -1.43 | worse |
| `trigonometric` | log-only | 0.00996 | +1.00 | +1.07 | better |
| `trigonometric` | ReLU-MLP | 0.00996 | -1.00 | -1.21 | worse |
| `trigonometric` | KAN | 0.00996 | -1.00 | -1.10 | worse |
| `trigonometric` | EML-v2 | 0.00996 | -1.00 | -1.55 | worse |
| `trigonometric` | EML→ReLU | 0.00996 | -1.00 | -0.94 | worse |
| `piecewise` | EML-noexp | 0.00996 | +1.00 | +1.77 | better |
| `piecewise` | EML-nolog | 0.33 | -0.44 | -0.05 | n.s. |
| `piecewise` | exp-only | 0.841 | +0.12 | -0.01 | n.s. |
| `piecewise` | log-only | 0.00996 | +1.00 | +0.62 | better |
| `piecewise` | ReLU-MLP | 0.0376 | -0.84 | -0.16 | worse |
| `piecewise` | KAN | 0.434 | +0.36 | +0.13 | n.s. |
| `piecewise` | EML-v2 | 0.33 | -0.44 | -0.11 | n.s. |
| `piecewise` | EML→ReLU | 0.245 | -0.52 | -0.07 | n.s. |
| `radioactive_decay` | EML-noexp | 0.00996 | +1.00 | +2.49 | better |
| `radioactive_decay` | EML-nolog | 0.00996 | -1.00 | -0.79 | worse |
| `radioactive_decay` | exp-only | 0.00996 | -1.00 | -0.71 | worse |
| `radioactive_decay` | log-only | 0.00996 | +1.00 | +1.04 | better |
| `radioactive_decay` | ReLU-MLP | 0.434 | -0.36 | -0.03 | n.s. |
| `radioactive_decay` | KAN | 0.109 | +0.68 | +0.19 | n.s. |
| `radioactive_decay` | EML-v2 | 0.00996 | -1.00 | -0.74 | worse |
| `radioactive_decay` | EML→ReLU | 0.00996 | -1.00 | -0.47 | worse |

Bilan : EML significativement meilleur dans 19/64 comparaisons, significativement pire dans 35/64, non significatif dans 10/64.
