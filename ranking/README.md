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
ranking_match_level.ipynb        ->  expérience optionnelle (lit results/predictions.csv)
ranking_era_drift.ipynb          ->  diagnostic de dérive + revue des remèdes
```

1. **Analyse des features** : choisit les variables à utiliser (uniquement sur 1995–2004).
2. **Prédictions** : entraîne 2 baselines naïves + 2 modèles + 1 blend et stocke tous les résultats.
3. **Analyse post-prédiction** : ne réentraîne rien ; elle compare, teste la significativité,
   regarde la stabilité par saison et produit le verdict.

Le rapport détaillé de l'enrichissement des features (gains, IC, décisions bloc par bloc) est dans
[`FEATURES_V2.md`](FEATURES_V2.md).

## Fichiers

| Fichier | Rôle |
|---|---|
| `ranking_feature_analysis.ipynb` | Analyse des features : points percentage, persistance, fenêtres d'historique, corrélations, importance, blocs externes, audit de fuite, sélection greedy. **À lire en premier.** |
| `ranking_predictions.ipynb` | Entraîne les modèles et écrit `results/`. C'est le seul notebook qui entraîne. |
| `ranking_results_analysis.ipynb` | Analyse post-prédiction : classement, intervalles de confiance, stabilité, distribution des erreurs, exemple détaillé, backtest, verdict. |
| `ranking_match_level.ipynb` | Expérience optionnelle : calendrier synthétique, simulation Monte Carlo des matchs, comparaison honnête au season-level. Ne bat pas les baselines sur un rang entier. |
| `ranking_era_drift.ipynb` | Diagnostic de la dérive 1995–2003 → 2005–2009 (corrélations, coefficients), revue des remèdes testés et feuille de route « comment faire mieux ». |
| `ranking_utils.py` | Fonctions partagées : chargement, points percentage, features, walk-forward, métriques, bootstrap. |
| `ranking_features.py` | Features externes **sans fuite** depuis les tables brutes (tirs, goalies, splits, TeamVsTeam, roster laggé, trophées, séries). |
| `selected_features.json` | Features retenues par l'analyse. Lu automatiquement par le notebook de prédiction. |
| `FEATURES_V2.md` | Rapport de la mission d'enrichissement : gain et IC de chaque bloc, décisions, blend, tableau final. |
| `results/` | Résultats générés : `predictions.csv`, `metrics_by_season.csv`, `summary.csv`, `run_info.json`. |
| `archive/` | Anciens notebooks et CSV, conservés pour traçabilité, non maintenus. |

## Données

- `../data_prepared/teams.csv` : dataset de base préparé par `data_preparation.ipynb`.
- `../data/Teams.csv` : points et matchs bruts, utilisés pour calculer le **points percentage**
  (`Pts / (2 * GP)`).
- `../data/` : tables brutes utilisées pour les features externes
  (`Scoring.csv`, `Goalies.csv`, `Master.csv`, `TeamVsTeam.csv`, `TeamSplits.csv`,
  `SeriesPost.csv`, `TeamsPost.csv`, `AwardsPlayers.csv`).

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
  saison (arrivées en cours de saison). Le **roster de *Y−1*** est autorisé (bloc `roster`), mais il
  a été rejeté par la sélection.

L'analyse contient une cellule « audit de fuite » qui échoue si une colonne coach, une statistique
non laggée ou une colonne de roster courant entre dans les features.

## Modèles

Deux baselines naïves :

| Modèle | Définition |
|---|---|
| `Naive: random` | scores aléatoires, graine fixe (plancher, aucune compétence) |
| `Naive: previous available ranking` | points percentage de la dernière saison disponible, re-classé dans la conférence courante |

Deux modèles :

| Famille | Modèle | Idée |
|---|---|---|
| External | Ridge + external | régression linéaire régularisée sur les features sélectionnées |
| External | Random forest + external | challenger non linéaire, forêt volontairement peu profonde (`max_depth=4`, `min_samples_leaf=20`, `max_features=0,7`, 500 arbres) |
| Blend | Naive + best model | `w × meilleur modèle + (1−w) × naive`, `w = 0,95` choisi sur 1995–2004 puis figé |

Le random forest est gardé comme **challenger non linéaire**, avec une réserve explicite : il reste
derrière Ridge (sélection 2,633 contre 2,611 ; test propre 3,480 contre 3,413). Une grille dev-only a
montré qu'un forest plus profond fait mieux sur 1995–2003 (2,592) mais se dégrade nettement sur
2005–2009 (3,653) : c'est de l'overfit, donc on garde la version conservatrice. Le test propre n'a
jamais servi à choisir les hyper-paramètres.

Les variantes testées pendant le projet (gradient boosting, ensemble, pairwise ranking) sont
conservées dans l'historique git ; d'anciens notebooks sont dans `archive/`.

## Résultats actuels

Sélection finale : `prev_points`, `shots_luck`, `experience` — soit le points percentage de la
saison précédente, les indicateurs de tirs/chance (PDO, différentiel de tirs, % d'arrêts) et
l'expérience de la franchise (`n_hist`). Aucun des nouveaux blocs (`teamvs`, `splits`, `roster`,
`awards`, `playoff_history`) n'entre dans la sélection.

- **Test propre 2005–2009** : la baseline `previous available ranking` reste la meilleure
  (MAE **3,400**). `Ridge + external` et le blend font **3,413** (**−0,013** position,
  IC 95 % [−0,374 ; +0,280]) et le random forest **3,480** (**−0,080** [−0,347 ; +0,173]) :
  rien n'est significatif.
- **Backtest 1995–2009** : le blend est le meilleur (MAE **2,893** contre **3,065** pour la
  baseline, soit **+0,174** position, IC 95 % [+0,010 ; +0,332]). Ridge fait **2,897**
  (+0,169 [−0,005 ; +0,333]) et le random forest **2,936** (+0,134 [−0,025 ; +0,306]).
  Le backtest a servi à la sélection : c'est un contrôle de stabilité, pas une preuve propre.
- **Piste match-level (optionnelle)** : le notebook `ranking_match_level.ipynb` ne bat **pas** les
  baselines sur un rang entier. Clean test : MAE **3,373** contre **3,400** (gain **+0,027**
  [−0,040 ; +0,080]) ; backtest : **3,041** contre **3,065** (+0,025 [−0,010 ; +0,060]). L'ancien
  +0,107 était un artefact du rang médian fractionnaire (voir §7 de `FEATURES_V2.md`).
- **Dérive entre époques** : les features de qualité de tir perdent beaucoup de leur pouvoir
  prédictif après 2004, alors que les coefficients Ridge restent stables. Les remèdes testés
  (recency, fenêtres glissantes, cibles alternatives, sous-ensembles) ne sont pas significatifs.
  Diagnostic et feuille de route dans `ranking_era_drift.ipynb` et §8 de `FEATURES_V2.md`.
- Conclusion honnête : sans les fuites (roster courant, coach), le gain des features externes est
  faible et non significatif sur le test propre. La baseline naïve est très forte en hockey. Le
  détail bloc par bloc est dans [`FEATURES_V2.md`](FEATURES_V2.md).

## Exécution

Kernel utilisé dans le projet : `ac_hockey`.

```powershell
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_feature_analysis.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=2400
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_predictions.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=2400
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_results_analysis.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=1200
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_match_level.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=3600
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_era_drift.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=1800
```

Exécuter dans cet ordre : l'analyse des features réécrit `selected_features.json`, puis les
prédictions écrivent `results/`, puis l'analyse post-prédiction lit `results/`. Les notebooks
match-level et era-drift sont autonomes ; ils lisent `results/predictions.csv` pour leurs
comparaisons.

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
  d'agrégats de saison. Le dataset ne contient pas les résultats match par match ; le notebook
  `ranking_match_level.ipynb` reconstruit un calendrier synthétique, mais il ne crée pas
  d'information nouvelle (voir `FEATURES_V2.md` §7).
