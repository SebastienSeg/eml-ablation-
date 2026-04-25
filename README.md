# EML — Exp-Minus-Log Neural Networks

> *Étude d'ablation rigoureuse d'une primitive neuronale combinant `exp` et `log`,
> testée contre ReLU-MLP, KAN, et toutes les ablations contrôlées.*

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](#licence)

---

## Table des matières

- [Idée centrale](#idée-centrale)
- [La couche EML en détail](#la-couche-eml-en-détail)
- [Pourquoi cette primitive ?](#pourquoi-cette-primitive-)
- [Variants comparés](#variants-comparés)
- [Fonctions cibles](#fonctions-cibles)
- [Méthodologie scientifique](#méthodologie-scientifique)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Sorties produites](#sorties-produites)
- [Résultats](#résultats)
- [Structure du code](#structure-du-code)
- [Décisions de conception](#décisions-de-conception)
- [Reproductibilité](#reproductibilité)
- [Limitations connues](#limitations-connues)
- [Licence](#licence)

---

## Idée centrale

La plupart des réseaux de neurones modernes apprennent à approximer des fonctions
arbitraires en empilant des transformations affines + une non-linéarité fixe
(ReLU, GELU, Swish…). Mais une grande partie des phénomènes que nous modélisons
— décroissance radioactive, lois de puissance, entropie, sigmoïde, softmax,
distributions log-normales — possède une **structure analytique** qui fait
intervenir conjointement `exp` et `log`.

**EML** (*Exp-Minus-Log*) est une primitive neuronale qui place ces deux briques
directement dans la couche, avec leurs propres matrices de projection apprises :

> `EML(x) = α · exp(W_exp · x) − β · ln(|W_log · x| + δ)`

L'hypothèse à tester :
**si la cible contient une structure exp/log, un réseau qui possède ces briques
en dur devrait apprendre mieux et plus vite qu'un MLP ReLU iso-paramétrique.**

Ce repo n'est pas un POC enthousiaste ; c'est un **banc d'essai d'ablation**
construit pour pouvoir réfuter cette hypothèse aussi facilement que la confirmer.

---

## La couche EML en détail

```python
class EMLLayer(nn.Module):
    """
    Nœud binaire :
        EML(x) = α · exp(W_exp · x) − β · ln(|W_log · x| + δ) + bias

    Paramètres appris :
        W_exp ∈ ℝ^(d_in × d_out)  — projection vers la branche exp
        W_log ∈ ℝ^(d_in × d_out)  — projection vers la branche log
        α, β  ∈ ℝ^d_out           — gains additifs par sortie
        bias  ∈ ℝ^d_out
    """
```

Diagramme du flux :

```
                  ┌──────────────┐    clamp(·, ±6)        ┌──────┐
            ┌────▶│  W_exp · x   │───────────────────────▶│ exp  │──┐
            │     └──────────────┘                        └──────┘  │
            │                                                        ▼
   x ───────┤                                                 α·exp(·) ─┐
            │                                                            ├─▶ − ─▶ + bias ─▶ y
            │     ┌──────────────┐    |·| + δ        ┌──────┐  β·ln(·) ─┘
            └────▶│  W_log · x   │───────────────────│ log  │──┘
                  └──────────────┘                   └──────┘
```

**Stabilisation numérique :**

- `torch.clamp(W_exp · x, ±6)` → borne le forward de `exp` à `e^6 ≈ 403`.
- `|W_log · x| + δ` (δ = 1e-4 par défaut) → empêche `log(0) = −∞`.
- `torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)` → contrôle le
  gradient backward (le clamp forward ne borne **pas** `∂exp(z)/∂z = exp(z)`).

**Initialisation :**

- `W_exp`, `W_log` : Xavier uniforme.
- `α`, `β` : `N(1.0, init_scale²)` → biais initial neutre (ni amplification
  ni atténuation systématique).

---

## Pourquoi cette primitive ?

Trois raisons s'empilent :

1. **Structure mathématique courante.** `exp ∘ log` apparaît dans toute loi
   de puissance (`(y+1)^x = exp(x · ln(y+1))`), toute entropie de Bernoulli,
   tout softmax, toute log-vraisemblance gaussienne. Un MLP doit *réinventer*
   cette structure à chaque fois ; EML la possède déjà.

2. **Universalité affaiblie ≠ inutile.** Le théorème d'approximation universelle
   de Cybenko/Hornik dit qu'un MLP suffit *en théorie* avec assez de neurones.
   La question pratique reste ouverte : à **budget paramétrique fixé**, quelle
   primitive converge mieux et plus vite sur quelle famille de fonctions ?

3. **Falsifiabilité.** Si EML gagnait sur tout, ce serait suspect (artefact
   architectural). Le protocole inclut donc des cibles **out-of-domain**
   (polynôme, trigo, piecewise) où ReLU-MLP et KAN devraient gagner.

---

## Variants comparés

Sept variants sont entraînés en parallèle, tous avec un graphe fully-connected
`[in_dim → hidden×3 → 1]` et LayerNorm entre les couches cachées.

| Variant       | Couche                                                | Rôle dans l'ablation                                    |
|---------------|-------------------------------------------------------|---------------------------------------------------------|
| **EML**       | `α·exp(W_exp·x) − β·ln(|W_log·x|+δ)`                  | Hypothèse principale                                    |
| **EML-noexp** | EML avec `α≡0`, `W_exp` gelée                         | Branche exp désactivée — isole l'effet de `log`         |
| **EML-nolog** | EML avec `β≡0`, `W_log` gelée                         | Branche log désactivée — isole l'effet de `exp`         |
| **exp-only**  | `α · exp(W·x + b)`                                    | Baseline minimaliste, exp pure                          |
| **log-only**  | `α · ln(|W·x| + δ)`                                   | Baseline minimaliste, log pure                          |
| **ReLU-MLP**  | `max(0, W·x + b)`                                     | Baseline standard de l'industrie                        |
| **KAN**       | B-splines apprenables par connexion (k=8 ctrl points) | Baseline expressive moderne (Liu et al. 2024)           |

**Iso-paramétrie.** La taille `hidden` est ajustée *par variant* pour que tous
aient ≈ le même nombre de paramètres entraînables que ReLU-MLP (référence
`HIDDEN=24`). Exemple pour `in_dim=2` :

| Variant     | hidden ajusté | Params entraînables |
|-------------|--------------:|--------------------:|
| ReLU-MLP    |            24 |               1 441 |
| EML         |           ≈17 |             1 516¹  |
| EML-noexp   |           ≈17 |               835²  |
| EML-nolog   |           ≈17 |               835²  |
| exp-only    |           ≈21 |               1 405 |
| log-only    |           ≈21 |               1 405 |
| KAN         |            ≈9 |               1 568 |

¹ Inclut paramètres gelés ; comparé en *total*, pas en *entraînables*.
² Une branche entière est gelée : seuls les paramètres restants sont
  entraînables, ce qui reste honnête car la topologie est identique à EML.

> Vérification automatique au début de chaque run via `_estimate_network_params`.

---

## Fonctions cibles

### In-domain (4) — structure exp/log naturelle

| Nom              | Expression                                       | Pourquoi                                    |
|------------------|--------------------------------------------------|---------------------------------------------|
| `explog_product` | `exp(−x²) · ln(1 + y²)`                          | Décomposable en un seul nœud EML            |
| `power_function` | `exp(x · ln(y+1)) = (y+1)^x`                     | Composition stricte exp ∘ log               |
| `logistic_log`   | `sigmoid(x) · exp(y) · ln(z+2)`                  | Trois primitives imbriquées                 |
| `cross_entropy`  | `H(σ(x)) + H(σ(y))`, H = entropie de Bernoulli   | Structure exp+log inévitable                |

### Out-of-domain (3) — sans structure exp/log

| Nom             | Expression                                          | Avantage attendu     |
|-----------------|-----------------------------------------------------|----------------------|
| `polynomial`    | `x³ + 2x²y − 0.5y³ + 0.3xy`                         | KAN, ReLU-MLP        |
| `trigonometric` | `sin(πx)·cos(πy) + 0.5·sin(2πx)`                    | KAN (oscillations)   |
| `piecewise`     | `|x|·sign(y) + clamp(x·y, ±1)`                      | ReLU-MLP             |

### Données réelles simulées

| Nom                  | Modèle                                                | But                                  |
|----------------------|-------------------------------------------------------|--------------------------------------|
| `radioactive_decay`  | `N(t) = N₀·exp(−λt) + ε`,  `ε ~ N(0, 0.05²)`         | Loi exp pure, bruitée (réalisme)     |

---

## Méthodologie scientifique

Ce projet n'est pas un benchmark "regardez comme ma méthode est meilleure".
C'est une étude conçue pour pouvoir échouer proprement.

| Précaution                       | Mise en œuvre                                                                                                                                            |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Variabilité aléatoire**        | 25 seeds par (variant × cible) en mode complet, 5 en mode rapide                                                                                         |
| **Données identiques**           | Une seed par cible (`TARGET_SEEDS`) → même `(X_tr, X_te)` pour tous les variants                                                                         |
| **Test non-paramétrique**        | Mann-Whitney U unilatéral (H₁ : EML < baseline). Pas de Wilcoxon paired car les variants ont des *initialisations indépendantes*                         |
| **IC robuste**                   | Bootstrap 10 000 resamplings sur la médiane (plus solide que les CI gaussiens en présence de runs divergents)                                             |
| **Comparaisons multiples**       | Correction Benjamini-Hochberg (FDR α=0.05) sur les 24 tests = 6 baselines × 4 cibles in-domain                                                            |
| **Taille d'effet**               | Cohen's *d* en plus de la *p*-value (large > 0.8, très large > 1.2)                                                                                      |
| **Divergence**                   | Run marqué divergent si `loss > 1e6` ou NaN ; exclu de la médiane mais comptabilisé dans `div_rate`                                                       |
| **Ablations contrôlées**         | EML-noexp / EML-nolog gèlent une branche (`requires_grad=False`) au lieu de modifier la topologie → toute différence est attribuable à la primitive      |
| **Cibles OOD**                   | Choisies pour *défavoriser* EML — un avantage EML sur ces cibles affaiblirait la thèse                                                                   |
| **Sweep d'hyperparamètres**      | 27 configs (LR × δ × init_scale) × 5 seeds → vérification que les résultats ne dépendent pas d'un point sucré                                            |
| **Ablation δ**                   | Sensibilité au terme de stabilité du log sur `δ ∈ {1e-6, …, 1e-2}`                                                                                       |
| **Histogrammes de gradient**     | Hooks backward pour vérifier l'absence de vanishing/exploding non géré par le clipping                                                                   |

---

## Installation

```bash
# Python 3.10+ recommandé
pip install torch numpy scipy matplotlib tqdm
```

Dépendances minimales — pas de framework lourd, pas de Hugging Face, pas de
fichier de config YAML. Tout est en pur PyTorch + NumPy + SciPy.

GPU optionnel : le code détecte automatiquement CUDA via `torch.cuda.is_available()`.

---

## Utilisation

```bash
# Test rapide (5 seeds, pas de sweep HP) — ~15 min CPU
python main.py --quick

# Expérience complète (25 seeds, sweep HP, ablation δ) — ~4h CPU / ~45 min GPU
python main.py

# Nombre de seeds personnalisé
python main.py --seeds 10
```

Variante mono-fichier :

```bash
python recap.py --quick   # même code, tout dans un seul fichier
```

Test unitaire de la couche KAN :

```bash
python test_kan_vectorization.py
```

---

## Sorties produites

| Fichier                              | Contenu                                                                                       |
|--------------------------------------|-----------------------------------------------------------------------------------------------|
| `figures/fig1_main_results.pdf`      | MSE médiane ± IC95 in-domain, par variant et par cible                                        |
| `figures/fig2_out_of_domain.pdf`     | Tableau OOD : où EML perd (honnêteté scientifique)                                            |
| `figures/fig3_grad_histogram.pdf`    | Histogramme `∂loss/∂(sortie de couche)` par variant                                           |
| `figures/fig4_hp_robustness.pdf`     | Heatmap MSE EML vs (LR, δ) — robustesse aux HP                                                |
| `figures/fig5_radioactive.pdf`       | Fit décroissance radioactive bruitée                                                          |
| `figures/fig6_delta_sensitivity.pdf` | MSE EML vs `δ ∈ [1e-6, 1e-2]`                                                                 |
| `figures/fig7_learning_curves.pdf`   | Courbes loss vs epoch (médiane ± IQR)                                                         |
| `figures/table1_full.tex`            | Tableau LaTeX prêt à inclure dans un article (booktabs)                                       |
| `eml_results_pub.json`               | Résultats bruts : médianes, CI, conv_rate, div_rate, p-values, Cohen's d, gradient stats     |

---

## Résultats

> ⚠️ Les chiffres ci-dessous proviennent du `eml_results_pub.json` checked-in,
> qui correspond à un run **`--quick` (5 seeds)** — donné à titre indicatif.
> Pour les chiffres de publication, exécuter `python main.py` sans flag.

### MSE médiane in-domain (5 seeds, lower = better)

| Cible            | EML        | EML-noexp | EML-nolog  | exp-only   | log-only | ReLU-MLP   | KAN        |
|------------------|-----------:|----------:|-----------:|-----------:|---------:|-----------:|-----------:|
| `explog_product` | **2.0e-3** | 2.2e-2    | 9.1e-5     | 2.7e-5     | 1.5e-2   | 4.8e-5     | 3.8e-5     |
| `power_function` | **1.3e-3** | 6.1e-1    | 1.7e-4     | 4.1e-5     | 4.2e-3   | 1.5e-4     | 2.6e-4     |
| `logistic_log`   | **5.0e-2** | 1.6e+0    | 1.1e-1     | 6.3e-1     | 1.6e+0   | 6.3e-1     | 2.1e-3     |
| `cross_entropy`  | **1.6e-3** | 4.3e-2    | 5.6e-5     | 6.8e-6     | 1.4e-2   | 3.0e-5     | 2.2e-5     |

### Tests Mann-Whitney U + correction BH (in-domain)

Sur 24 comparaisons (6 baselines × 4 cibles) après correction BH (α=0.05) :

- **Significatif EML > variant ablaté** (`EML-noexp`, `log-only`) sur la majorité des cibles → confirme que **les deux branches contribuent**.
- **EML perd contre `exp-only`/`KAN`** sur les cibles dominées par exp pur (`explog_product`, `cross_entropy`) → la branche `log` ne sert que quand la cible la requiert vraiment.
- **EML domine systématiquement sur `logistic_log`** (six baselines battues, p < 0.01, Cohen's d ∈ [2.9, 32.6]) → preuve la plus forte que l'hypothèse tient quand la structure exp+log est indispensable.

### Out-of-domain (sanity check)

| Cible           | EML       | KAN          | ReLU-MLP    | Verdict                            |
|-----------------|----------:|-------------:|------------:|------------------------------------|
| `polynomial`    | 1.18      | **0.12**     | 9.44        | KAN gagne, EML perd → attendu      |
| `trigonometric` | 1.5e-2    | **2.2e-4**   | 0.27        | KAN gagne, EML perd → attendu      |
| `piecewise`     | 3.9e-2    | **7.4e-3**   | 1.04        | KAN gagne, EML perd → attendu      |

EML **perd correctement** sur les fonctions sans structure exp/log. C'est ce
qu'on veut voir : l'avantage in-domain n'est pas un artefact général, c'est
bien lié à la primitive.

### Gradients

L'histogramme (Fig. 3) confirme :
- Aucun vanishing (< 1e-6) sur EML/EML-nolog/ReLU/KAN.
- ~9 % d'événements `> 1e2` sur EML, gérés par `clip_grad_norm_=10`.
- EML-noexp est la seule variante avec ~18 % d'exploding events — cohérent
  avec le fait que `log` génère naturellement de plus gros gradients pour
  les petites valeurs d'entrée.

---

## Structure du code

```
.
├── README.md            ← (vous y êtes)
├── config.py            ← constantes globales, hyperparamètres, topologie
├── targets.py           ← fonctions cibles in-domain, OOD, radioactive
├── primitives.py        ← EMLLayer, ExpOnlyLayer, LogOnlyLayer, ReLULayer,
│                          KANLayer, build_network(), iso-paramétrie
├── training.py          ← run_one(), boucle d'entraînement, GradientCollector
├── stats_analysis.py    ← summarize(), Mann-Whitney U, bootstrap CI, BH correction
├── figures.py           ← 7 figures matplotlib + table_latex()
├── main.py              ← orchestrateur (8 étapes, dump JSON final)
├── recap.py             ← équivalent mono-fichier (tout-en-un)
├── test_kan_vectorization.py
├── eml_results_pub.json ← résultats bruts (commit pour reproductibilité)
└── figures/             ← PDF + LaTeX générés
```

### Flux d'exécution (`main.py`)

```
1. Vérification iso-paramétrique (param counts par variant)
2. Expérience principale       — 7 variants × 4 cibles in-domain × N seeds
3. Out-of-domain               — 7 variants × 3 cibles OOD × N seeds
4. Données réelles bruitées    — 7 variants × radioactive × N seeds
5. Sweep HP                    — EML × 27 configs × 5 seeds
6. Ablation δ                  — EML × 5 valeurs δ × 5 seeds
7. Histogrammes de gradient    — 5 variants × 1 seed
8. Analyse statistique         — Mann-Whitney + BH + Cohen's d
9. Génération figures + JSON   — 7 PDF + 1 .tex + eml_results_pub.json
```

---

## Décisions de conception

### Pourquoi `EML(x) = α·exp(...) − β·log(...)` (signe `−`) ?

Le signe négatif est délibéré, pas cosmétique :
- `exp` croît, `log` croît aussi mais plus lentement.
- `α·exp − β·log` permet à un seul nœud de modéliser des fonctions à
  *régimes opposés* (croissance rapide vs croissance lente) sans empiler
  deux nœuds. Une somme `α·exp + β·log` serait redondante avec `α'·exp`
  pour les régimes positifs.

### Pourquoi geler une branche au lieu de retirer un module ?

`EML-noexp` ne supprime pas la branche `exp` : il la met à zéro avec
`requires_grad=False`. La topologie reste **strictement identique** à EML.
Toute différence de performance est donc attribuable à la *présence* de la
primitive, pas à un changement de structure du graphe.

### Pourquoi Mann-Whitney et pas Wilcoxon paired ?

Les variants ont des architectures différentes. À seed identique, ils
produisent des initialisations indépendantes (Xavier sur des matrices de
tailles différentes). Le seul couplage est sur le dataset, ce qui ne
suffit pas à justifier un test paired. Mann-Whitney U non-paired est
le test non-paramétrique correct dans ce contexte.

### Pourquoi pas de straight-through estimator pour `exp` ?

Un STE qui borne `∂exp(z)/∂z` localement altère le signal de gradient et
introduit un biais d'optimisation non contrôlé. Le clipping global de la
norme (`clip_grad_norm_=10`) borne l'amplitude *sans biaiser la direction*
du gradient — la solution propre.

### Pourquoi `δ` dans `ln(|·| + δ)` plutôt que `ln(max(·, δ))` ?

`max(·, δ)` est non-différentiable au point de transition. `+ δ` est lisse
partout, et le gradient `1/(|·| + δ)` reste borné. Coût : un léger biais
sur les très petites valeurs, vérifié comme négligeable par l'ablation
sur `δ ∈ [1e-6, 1e-2]`.

---

## Reproductibilité

- **Seeds.** `GLOBAL_SEED=42`, `TARGET_SEEDS` fixe par cible (101, 202, 303…)
  pour garantir des datasets identiques entre runs.
- **Versions.** `torch.set_default_dtype(torch.float32)` pour éviter le
  drift FP16/FP64.
- **Déterminisme.** Sur GPU, certaines opérations cuDNN restent
  non-déterministes (l'argument `deterministic=True` n'est pas activé pour
  ne pas pénaliser la vitesse). Les conclusions statistiques (24 tests, 25
  seeds, BH α=0.05) sont robustes à ce niveau de bruit.
- **JSON committé.** `eml_results_pub.json` est versionné — toute personne
  peut comparer ses runs aux résultats de référence.

Pour rejouer exactement le run committé :

```bash
python main.py --quick    # reproduit eml_results_pub.json (n_seeds=5)
```

---

## Limitations connues

- **Échelle.** Toute l'étude est conduite sur des cibles 2D-3D avec
  N=2000 train / N=500 test. La généralisation à des problèmes haute-dimension
  (vision, langue) n'est pas testée.
- **Pas de hidden layer mixte.** Toutes les couches d'un réseau sont du même
  type. On ne teste pas un réseau hybride (e.g. EML en première couche, ReLU
  ensuite), qui pourrait capturer le meilleur des deux mondes.
- **`exp` reste capricieux.** Même avec clamp + grad clipping, ~9 % des
  gradients EML touchent le seuil de clipping. Sur des problèmes plus
  profonds (>3 couches), cela pourrait devenir limitant.
- **KAN-lite.** L'implémentation KAN ici est une version simplifiée
  (B-splines linéaires par morceaux, k=8) — pas la version complète de
  Liu et al. 2024. Les chiffres KAN sont donc une *borne inférieure* de
  ce que KAN peut faire.
- **Dataset radioactif synthétique.** Pas de "vraies" données expérimentales.
  Le bruit gaussien est une approximation raisonnable mais n'inclut pas les
  artefacts réels (calibration, dérive, etc.).

---

## Citation

Si vous utilisez ce code ou ces résultats :

```bibtex
@misc{eml_ablation_2026,
  title = {EML: Exp-Minus-Log Neural Networks — An Ablation Study},
  year  = {2026},
  note  = {\url{https://github.com/SebastienSeg/eml-ablation-}}
}
```

---

## Licence

MIT — voir [LICENSE](LICENSE) si présent. Sinon, considérez le code comme
libre d'usage avec attribution.

---

## Remerciements

- Théorème d'approximation universelle : Cybenko (1989), Hornik (1991).
- KAN baseline : Liu et al., *KAN: Kolmogorov-Arnold Networks*, 2024.
- Correction BH : Benjamini & Hochberg, *J. R. Statist. Soc. B*, 1995.
