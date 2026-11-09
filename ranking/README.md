# Ranking — prédiction du classement final

Prédiction, **avant le début d'une saison**, du classement final de chaque équipe dans sa
conférence (deux classements d'environ 15 équipes par saison).

Ce dossier regroupe tout ce qui concerne la question (a). Le reste du projet (préparation des
données, question (b) sur les coachs) reste à la racine.

## Pipeline

```
ranking_feature_analysis.ipynb   ->  selected_features.json
ranking_predictions.ipynb        ->  results/*.csv + results/run_info.json
ranking_results_analysis.ipynb   ->  lit results/ et produit l'analyse post-prédiction
```

1. **Analyse des features** : choisit les variables à utiliser (uniquement sur 1995–2004).
2. **Prédictions** : entraîne 2 baselines naïves + 5 modèles et stocke tous les résultats.
3. **Analyse post-prédiction** : ne réentraîne rien ; elle compare, teste la significativité,
   regarde la stabilité par saison et produit le verdict.

## Fichiers

| Fichier | Rôle |
|---|---|
| `ranking_feature_analysis.ipynb` | Analyse des features : points percentage, persistance, fenêtres d'historique, corrélations, importance, blocs externes, audit de fuite, sélection greedy. **À lire en premier.** |
| `ranking_predictions.ipynb` | Entraîne les modèles et écrit `results/`. C'est le seul notebook qui entraîne. |
| `ranking_results_analysis.ipynb` | Analyse post-prédiction : classement, intervalles de confiance, stabilité, distribution des erreurs, exemple détaillé, backtest, verdict. |
| `ranking_utils.py` | Fonctions partagées : chargement, points percentage, features, walk-forward, métriques, bootstrap. |
| `ranking_features.py` | Features externes **sans fuite** depuis les tables brutes (tirs, goalies, séries, trophées, splits). |
| `selected_features.json` | Features retenues par l'analyse. Lu automatiquement par le notebook de prédiction. |
| `results/` | Résultats générés : `predictions.csv`, `metrics_by_season.csv`, `summary.csv`, `run_info.json`. |
| `archive/` | Anciens notebooks et CSV, conservés pour traçabilité, non maintenus. |

## Données

- `../data_prepared/teams.csv` : dataset de base préparé par `data_preparation.ipynb`.
- `../data/Teams.csv` : points et matchs bruts, utilisés pour calculer le **points percentage**
  (`Pts / (2 * GP)`).
- `../data/` : tables brutes utilisées pour les features externes
  (`Scoring.csv`, `Goalies.csv`, `SeriesPost.csv`, `TeamsPost.csv`, `AwardsPlayers.csv`,
  `TeamSplits.csv`).

Les chemins sont résolus à partir de l'emplacement des modules : les notebooks fonctionnent
qu'on les lance depuis la racine du projet ou depuis `ranking/`.

## Protocole

- **Cible** : z-score du **points percentage** de la saison au niveau de la ligue. Le classement est
  obtenu en triant les scores prédits dans chaque `(saison, conférence)`. Le points percentage est
  comparable entre les saisons à 80, 82 et 48 matchs, contrairement aux points bruts ou aux victoires.
- **Walk-forward** : pour prédire la saison *Y*, on entraîne sur toutes les saisons `< Y`.
  Aucune information de *Y* ou après n'est utilisée.
- **Sélection des features** : uniquement sur 1995–2004. Le test propre est 2005–2009 ; le backtest
  1995–2009 sert de contrôle de stabilité, pas de test totalement externe.
- **Métriques** : MAE en positions, RMSE, Spearman, recouvrement du top 8, précision par paires.
  Les comparaisons sont testées par bootstrap apparié sur des classements complets.

## Politique anti-fuite

Aucune feature n'utilise d'information de la saison prédite :

- toutes les statistiques d'équipe sont **laggées** (saison *Y-1* prédit *Y*) ;
- les colonnes de coach (`coach_changed`, ancienneté, intérim, nombre de coachs) sont **exclues** :
  c'est la cible de la tâche coach et elles décrivent la saison en cours ;
- la **composition du roster de la saison courante** est exclue : le roster n'est connu qu'après la
  saison (arrivées en cours de saison).

L'analyse contient une cellule « audit de fuite » qui échoue si une colonne coach/roster entre dans
les features.

## Modèles

Deux baselines naïves :

| Modèle | Définition |
|---|---|
| `Naive: random` | scores aléatoires, graine fixe (plancher, aucune compétence) |
| `Naive: previous available ranking` | points percentage de la dernière saison disponible, re-classé dans la conférence courante |

Cinq modèles :

| Famille | Modèle | Idée |
|---|---|---|
| External | Ridge + external | régression linéaire régularisée sur les features sélectionnées |
| External | Gradient boosting + external | modèle non linéaire sur les mêmes features |
| External | Ensemble + external | moyenne de Ridge et du gradient boosting |
| Pairwise | Pairwise logistic | apprentissage du classement par paires |
| Pairwise | Pairwise boosting | version non linéaire du précédent |

Les anciens modèles (random forest, blend pairwise, modèles d'historique seuls) sont dans
`archive/`.

## Résultats actuels

Sélection finale : `prev_points`, `shots_luck`, `experience` — soit le points percentage de la
saison précédente, les indicateurs de tirs/chance (PDO, différentiel de tirs, % d'arrêts) et
l'expérience de la franchise (`n_hist`).

- **Test propre 2005–2009** : la baseline `previous available ranking` reste la meilleure
  (MAE **3,387**). Le meilleur modèle entraîné est `Ensemble + external` (MAE **3,400**), soit
  **-0,013** position : aucun modèle n'est significativement meilleur que la baseline.
- **Backtest 1995–2009** : `Ridge + external` est le meilleur (MAE **2,897** contre **3,076** pour
  la baseline, soit **+0,179** position, IC 95 % [-0,010 ; +0,348] — dans le bruit).
- Conclusion honnête : sans les fuites (roster courant, coach), le gain des features externes est
  faible et non significatif sur le test propre. La baseline naïve est très forte en hockey.

## Exécution

Kernel utilisé dans le projet : `ac_hockey`.

```powershell
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_feature_analysis.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=2400
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_predictions.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=2400
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_results_analysis.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=1200
```

Exécuter dans cet ordre : l'analyse des features réécrit `selected_features.json`, puis les
prédictions écrivent `results/`, puis l'analyse post-prédiction lit `results/`.

## Limites

- **Échantillon de test court** : 5 saisons, 10 classements de conférence. Les intervalles restent
  larges et beaucoup de différences entre modèles sont dans le bruit.
- **Points percentage** : la cible est `Pts / (2 * GP)`, pas les points bruts. Cela change légèrement
  les valeurs mais pas le classement des équipes dans une saison normale.
- **Pas de fuite** : le roster courant et les colonnes coach sont volontairement absents. C'est ce
  qui explique que le gain externe soit plus faible qu'une analyse naïve avec le roster de la saison.
- **Données externes toujours internes au projet** : pas de cotes de marché, blessures ou masse
  salariale. C'est le prochain vrai levier pour gagner davantage.
- **Pas de série de victoires en cours** (`Attributs_AC.txt`) : impossible à construire à partir
  d'agrégats de saison, il faudrait des données match par match.
