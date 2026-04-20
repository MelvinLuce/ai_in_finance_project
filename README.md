# Prédiction de la Courbe des Taux Américaine par Machine Learning

---

# 1. Project Information

- **Project Title:** Yield Curve Prediction Using Machine Learning
- **Group Members:**
  - Student 1 – Théo Zoccolella
  - Student 2 – Alex Provot
  - Student 3 – Melvin Luce

- **Course Name:** AI In Finance
- **Instructor:** Nicolas De Roux & Mohamed EL FAKIR
- **Submission Date:** 20/04/2026

---

# 2. Project Description

La courbe des taux représente la relation entre les rendements des obligations souveraines et leur maturité résiduelle. Elle est l'un des instruments les plus informatifs en macroéconomie et sur les marchés financiers, synthétisant simultanément les anticipations de politique monétaire, les primes de risque de durée, le cycle économique (une courbe inversée précède historiquement chaque récession américaine depuis 1960) et les anticipations d'inflation.

Les modèles économétriques classiques (ARIMA, VAR) supposent des relations linéaires et stationnaires. Or, la dynamique des taux est caractérisée par des non-linéarités de régime, des interactions complexes entre maturités et variables macroéconomiques, ainsi que des ruptures structurelles (crise 2008, COVID-19, inflation 2022). Les méthodes de machine learning — notamment les modèles d'ensemble (Random Forest, Gradient Boosting) et les réseaux récurrents (LSTM) — permettent de capturer ces dynamiques sans imposer de forme fonctionnelle linéaire explicite.

---

# 3. Project Goal

L'objectif du projet est de construire un modèle de machine learning permettant de prédire les composantes de la courbe des taux américaine (niveau, pente, courbure) à un horizon de 1 mois via la décomposition Nelson-Siegel. Le projet vise à :

- Reconstruire la courbe des taux complète pour chaque maturité à partir des facteurs prédits
- Comparer les approches économétriques classiques avec des méthodes ML d'ensemble et des réseaux de neurones
- Évaluer le pouvoir prédictif dans un cadre de validation temporelle rigoureux (sans fuite d'information)

---

# 4. Task Definition

- **Task Type:** Régression supervisée sur séries temporelles (prévision à 1 mois)
- **Input Variables:** 95 features organisées en 7 familles :
  - Rendements décalés (25 vars) : `yield_{mat}_lag{k}` pour k = 1, 3, 6, 12 mois
  - Facteurs décalés (18 vars) : level, slope, curvature avec leurs lags
  - Statistiques glissantes (21 vars) : moyennes mobiles, écarts-types, EWMA
  - Momentum / déviations (10 vars) : écart à la MA, delta, accélération
  - Spreads (6 vars) : 10Y-3M, 10Y-2Y, 30Y-2Y, 5Y-2Y, 10Y-5Y, std cross-sectionnel
  - Indicateurs de régime (6 vars) : courbe inversée, plate, volatilité élevée, taux bas, z-score
  - Variables macro (9 vars) : fed_funds, inflation, chômage, VIX, différences
- **Target Variable:** Facteurs Nelson-Siegel au mois suivant — `level_t1`, `slope_10y_3m_t1`, `curvature_t1` (niveaux et variations)
- **Evaluation Metric(s):**
  - **RMSE** : métrique principale, pénalise les grandes erreurs
  - **MAE** : interprétable en points de base
  - **R²** : proportion de variance expliquée
  - **DA** (Directional Accuracy) : proportion de bonnes prédictions directionnelles

---

# 5. Dataset Description

## Dataset Overview

- **Number of samples:** 401 observations complètes (après feature engineering et suppression des NaN)
- **Effective period:** 1991-07-31 à 2024-11-30 (mensuel)
- **Number of features:** 95 variables construites
- **Target variable:** Facteurs Nelson-Siegel — level, slope_10y_3m, curvature
- **Data source:** Federal Reserve Bank of St. Louis — [FRED](https://fred.stlouisfed.org/)

**Découpage temporel :**

| Split | Période | Observations |
|-------|---------|--------------|
| Train | jusqu'au 2018-12-31 | 330 |
| Validation | 2019-01-31 – 2021-12-31 | 36 |
| Test | 2022-01-31 – 2024-11-30 | 35 |

---

## Feature Description

| Feature | Description | Type |
|---------|-------------|------|
| DGS1MO – DGS30 | Taux des bons du Trésor US (11 maturités : 1M, 3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y, 30Y) | Numérique |
| FEDFUNDS | Taux directeur de la Réserve Fédérale | Numérique |
| CPIAUCSL | Indice des prix à la consommation (inflation YoY calculée) | Numérique |
| UNRATE | Taux de chômage américain | Numérique |
| T10YIE | Point mort d'inflation à 10 ans | Numérique |
| VIXCLS | Indice de volatilité VIX | Numérique |
| USREC | Indicateur de récession NBER | Binaire |
| level | Facteur de niveau Nelson-Siegel (moyenne des taux ou taux 10Y) | Numérique |
| slope_10y_3m | Facteur de pente (spread 10Y – 3M) | Numérique |
| curvature | Facteur de courbure (2×5Y – 3M – 10Y) | Numérique |
| yield_{mat}_lag{k} | Rendement décalé de k mois pour la maturité mat | Numérique |
| rolling_mean_{f}_{w} | Moyenne mobile sur w mois du facteur f | Numérique |
| regime_inverted | Indicateur de courbe inversée | Binaire |

---

## Target Variable

Les trois facteurs Nelson-Siegel prédits au mois t+1 :

- **level_t1** : niveau moyen de la courbe — reflète les anticipations d'inflation à long terme
- **slope_10y_3m_t1** : pente de la courbe — capture le cycle économique et la politique monétaire
- **curvature_t1** : courbure de la courbe — mesure le renflement intermédiaire

Des cibles en variation (`delta_level_t1`, `delta_slope_10y_3m_t1`, `delta_curvature_t1`) sont également utilisées.

---

## Data Types

- **Numérique (continu) :** taux, spreads, facteurs, variables macro, statistiques glissantes
- **Binaire :** indicateurs de régime (courbe inversée, récession, volatilité élevée)
- **Séries temporelles :** données mensuelles avec forte autocorrélation

---

## Data Distribution

- Les séries de taux présentent une forte persistance (proche d'une marche aléatoire)
- Le taux 1M est manquant avant juillet 2001 (32.86% de NaN)
- Les autres maturités sont complètes depuis janvier 1990
- Ruptures structurelles majeures : crise 2008 (taux proches de 0), COVID-19 2020, inflation 2022

---

## Data Quality

- **Valeurs manquantes :** 473 NaN totaux dans les features — principalement sur le taux 1M (avant 2001)
- **Traitement :** forward-fill pour les courtes lacunes ; suppression des lignes avec NaN résiduels
- **Pas de doublons** détectés
- **Ruptures structurelles** non traitées explicitement (gérées via les indicateurs de régime)

---

# 6. Data Preprocessing

1. **Gestion des valeurs manquantes :** forward-fill pour les courtes interruptions, suppression des observations initiales pour les séries à démarrage tardif (taux 1M depuis 2001)

2. **Construction des facteurs Nelson-Siegel :** décomposition de la courbe en trois facteurs économiquement interprétables (level, slope, curvature) via des proxies empiriques

3. **Feature engineering (95 variables) :**
   - Rendements décalés de 1, 3, 6, 12 mois sur 5 maturités
   - Statistiques glissantes (moyennes mobiles, écarts-types, EWMA) sur fenêtres de 3, 6, 12, 24, 36 mois
   - Indicateurs de régime et spreads entre maturités
   - Variables macro en niveaux et en variations

4. **Standardisation :** `StandardScaler` intégré dans un `Pipeline` scikit-learn pour éviter toute fuite d'information — le scaler est ajusté uniquement sur le train, puis appliqué au val/test

5. **Construction des cibles :** décalage temporel de +1 mois pour s'assurer qu'aucune information future n'est utilisée en prédiction

6. **Validation temporelle :** `TimeSeriesSplit` (4-5 folds) pour le tuning — l'ordre chronologique est préservé, sans mélange aléatoire

---

# 7. Modeling Approach

## Chosen Models

| Modèle | Type | Description |
|--------|------|-------------|
| Naive (Persistance) | Baseline | $\hat{f}_{t+1} = f_t$ — prédiction = dernière valeur observée |
| Ridge Regression | Linéaire | Régression avec régularisation L2 |
| Random Forest | Ensemble (bagging) | 200–400 arbres, max_depth 3–10 |
| Gradient Boosting | Ensemble (boosting) | 100–200 estimators, learning_rate 0.01–0.1 |
| XGBoost | Ensemble (boosting) | Régularisation L1/L2, colsample_bytree |
| LSTM | Réseau récurrent | PyTorch, 32–64 unités cachées, prédit les variations |

---

## Modeling Strategy

- **Baseline :** le modèle Naive sert de référence minimale — tout modèle doit le battre pour apporter de la valeur
- **Approche par facteur :** chaque modèle est entraîné indépendamment sur chaque facteur Nelson-Siegel (level, slope, curvature)
- **Hyperparameter tuning :** `GridSearchCV` avec `TimeSeriesSplit` (4–5 folds) optimisant le RMSE de validation croisée
- **Pipeline scikit-learn :** `StandardScaler` + modèle dans un pipeline pour éviter les fuites de données
- **LSTM :** entraîné sur les variations (`delta`) pour capturer les signaux directionnels

---

## Evaluation Metrics

- **RMSE** (Root Mean Squared Error) : métrique principale de sélection, pénalise davantage les grandes erreurs — pertinent car les erreurs sur les taux ont un impact financier asymétrique
- **MAE** (Mean Absolute Error) : interprétable en points de base, robuste aux outliers
- **R²** : proportion de variance expliquée — permet de comparer les modèles à une baseline de moyenne
- **DA** (Directional Accuracy) : proportion de bonnes prédictions directionnelles — pertinent pour les stratégies de trading basées sur la pente de la courbe

---

# 8. Results

## Performance sur le jeu de test — Niveaux des facteurs

| Modèle | level RMSE | slope RMSE | curvature RMSE | Avg RMSE |
|--------|-----------|------------|----------------|----------|
| **Naive (Persistance)** | **0.2959** | **0.3795** | **0.4464** | **0.3739** |
| Ridge | 0.4908 | 0.6291 | 0.6910 | 0.6036 |
| Random Forest | 0.5185 | 1.1182 | 0.9034 | 0.8467 |
| Gradient Boosting | 0.5338 | 1.1211 | 0.9592 | 0.8714 |
| XGBoost | 0.5859 | 1.2579 | 0.9941 | 0.9460 |


## Performance sur le jeu de test — Variations (deltas)

| Modèle | delta_level RMSE | Directional Accuracy |
|--------|-----------------|----------------------|
| LSTM Delta Compact | **0.2983** | **68.6%** |
| LSTM Delta Standard | 0.3033 | 62.9% |

## Conclusions

- **Le modèle Naive domine tous les modèles complexes** sur RMSE et R² — les taux d'intérêt sont fortement persistants et proches d'une marche aléatoire, ce qui limite le contenu informationnel exploitable
- **Ridge outperforms les modèles non-linéaires** — la structure linéaire est plus robuste hors échantillon sur ce type de données
- **Le LSTM Delta Compact** atteint 68.6% de précision directionnelle, suggérant un signal exploitable pour les stratégies de trading sur les variations de taux

---

# 9. Project Structure

Le notebook principal `notebooks/projet2_yield_curve_ML.ipynb` est organisé en 12 sections :

| Section | Contenu |
|---------|---------|
| **1. Introduction et Motivation Économique** | Rôle de la courbe des taux, décomposition Nelson-Siegel, justification du ML |
| **2. Objectif et Formulation ML** | Définition formelle du problème, métriques d'évaluation |
| **3. Collecte des Données** | Téléchargement via FRED API (taux US 11 maturités + variables macro) |
| **4. Construction des Facteurs** | Calcul des facteurs Nelson-Siegel (level, slope, curvature) et ACP |
| **5. Feature Engineering** | Construction des 95 variables (lags, rolling stats, spreads, régimes, macro) |
| **6. Modèles de Référence** | Baseline Naive (persistance) et Ridge Regression |
| **7. Modèles ML** | Random Forest et Gradient Boosting / XGBoost avec GridSearchCV |
| **8. Deep Learning — LSTM** | Réseau récurrent PyTorch sur les variations de facteurs |
| **9. Reconstruction de la Courbe** | Reconstruction des rendements par maturité à partir des facteurs prédits |
| **10. Interprétation Économique** | Feature importance, hiérarchie des modèles |
| **11. Discussion Critique** | Forces, limites, lecture prudente des résultats |
| **12. Conclusion** | Bilan, message principal, pistes d'amélioration, références |

---

# 10. Installation

```bash
pip install -r requirements.txt
```

Dépendances principales :

```bash
pip install pandas numpy scikit-learn xgboost torch fredapi matplotlib seaborn
```

Le notebook est conçu pour tourner sur **Google Colab** (Python 3.10).

---

# 11. References

- Nelson, C.R. & Siegel, A.F. (1987). *Parsimonious Modeling of Yield Curves*. Journal of Business.
- Diebold, F.X. & Li, C. (2006). *Forecasting the term structure of government bond yields*. Journal of Econometrics.
- Litterman, R. & Scheinkman, J. (1991). *Common Factors Affecting Bond Returns*. Journal of Fixed Income.
- Breiman, L. (2001). *Random Forests*. Machine Learning.
- Friedman, J.H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine*. Annals of Statistics.
- Hochreiter, S. & Schmidhuber, J. (1997). *Long Short-Term Memory*. Neural Computation.
- **Données :** Federal Reserve Bank of St. Louis — [FRED](https://fred.stlouisfed.org/)
