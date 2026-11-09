# FEATURES_V2 — enrichissement des features du ranking NHL

Rapport de la mission « enrichir les features sans violer la leak policy, mesurer rigoureusement
les gains, ajouter un blend avec le naive ». Toutes les valeurs sont des MAE walk-forward en
positions de classement (plus bas = mieux), avec un intervalle de confiance à 95 % obtenu par
bootstrap apparié sur des classements `(saison, conférence)` complets.

## 1. Ce qui a été fait

### 1.1 Corrections préalables (tâche 1)

| Correction | Détail |
|---|---|
| Historique par `franchID` | tous les `groupby` historiques utilisent la franchise (ANA→AND, CAR→HAR, COL→QUE, DAL→MNS, NJD→COR, PHO→WIN), plus le `tmID` instable. |
| Départage des rangs | `Pts`, puis `W`, puis `tmID` (au lieu de `tmID` seul). Côté prédiction, `score`, puis `wins_raw_lag1`, puis `tmID`. |
| Dataset régénéré | `data_prepared/teams.csv` passe de 30 à **44 colonnes** ; 732 équipes-saisons, 1980–2009. |
| Filtre 1980+ | explicite : `FIRST_YEAR = 1980` et `year >= FIRST_YEAR` (l'ancien `year > 1979` était équivalent mais ambigu). |

Effet secondaire assumé : le changement de départage déplace légèrement la baseline. Le naive
« previous available ranking » passe de 3,387 (ancien chiffre) à **3,400** sur 2005–2009. Les
résultats de `ranking/results/` ont été entièrement régénérés.

### 1.2 Nouveaux blocs de features (tâches 2 à 5)

Tous les blocs sont construits dans `ranking/ranking_features.py` et **laggés** : la valeur de la
saison *Y−1* prédit *Y*. Aucune statistique de *Y*, aucun `TeamsPost`/`SeriesPost` de *Y*, aucun
`AwardsPlayers` de *Y* et aucune colonne `coach*` n'entre dans les features.

| Bloc | Nb | Construction (toutes laggées) |
|---|---:|---|
| `teamvs` | 4 | taux de victoire contre les équipes qualifiées en playoffs, contre le top-8, contre la même conférence ; force moyenne des adversaires (pondérée par les matchs), depuis `TeamVsTeam` de *Y−1* |
| `splits` | 5 | points percentage domicile, extérieur, delta domicile−extérieur, seconde moitié (janv.–avr.) et printemps (mars–avr.), depuis `TeamSplits` de *Y−1* |
| `roster` | 8 | nombre de joueurs à 20 buts, nombre de joueurs à ≥ 1 point/match, part des points du top-6, âge moyen pondéré par le temps de jeu, continuité du roster (fraction des joueurs déjà là en *Y−2*), % d'arrêts et charge du gardien titulaire, % d'arrêts du backup |
| `awards` | 3 | nombre total de trophées, de sélections All-Star et de trophées majeurs gagnés par des joueurs du roster de *Y−1* |
| `playoff_history` | 4 | tours, matchs, victoires et % de victoires en séries de *Y−1* |

Notes d'implémentation :

- `teamvs` utilise la conférence de *Y−1* des adversaires (pas celle de *Y*), car la conférence
  peut changer entre les deux saisons.
- `close_games_winrate` a été abandonné : `TeamVsTeam` ne contient pas les scores des matchs,
  seulement les bilans agrégés par adversaire.
- Le roster laggé utilise `Scoring`, `Goalies` et `Master` de *Y−1*. Les doublons de scoring sur
  `(playerID, year, tmID)` sont agrégés avant de suivre la franchise. La continuité est calculée
  par `franchID` et mappée via la saison précédente réelle (2003→2005, la saison 2004 est absente).

### 1.3 Blend avec le naive (tâche 6)

Le meilleur modèle entraîné sur 1995–2004 est `Ridge + external`. Le blend retenu est :

```
score = 0.95 × z(Ridge + external) + 0.05 × z(previous available ranking)
```

Le poids `w = 0.95` est choisi sur une grille grossière (pas de 0,05) **sur 1995–2004 uniquement**,
puis figé pour le clean test et le backtest. Le gain de 0,95 sur 1,00 est de 0,007 position
(2,604 vs 2,611) : le blend est donc presque identique au modèle seul. Ce n'est pas un résultat
robuste, c'est un détail de la grille.

## 2. Protocole

- **Walk-forward** : pour chaque saison test *Y*, entraîner sur toutes les saisons `< Y`, prédire *Y*.
- **Sélection / tuning** : 1995–2004 uniquement (2004 absente du dataset).
- **Clean test** : 2005–2009, jamais utilisé pour une décision de design.
- **Backtest** : 1995–2009, utilisé comme contrôle de stabilité et de puissance. **Ce n'est pas un
  test propre** : la sélection des features a vu 1995–2004.
- **Unité d'évaluation** : le classement de conférence. Le bootstrap rééchantillonne des
  `(year, confID)` entiers. Un gain est significatif seulement si l'IC 95 % est entièrement > 0.

## 3. Résultats des blocs sur 1995–2004

Référence : base historique seule (`prev_points` + `history_points`), MAE **2,722**,
Spearman 0,604. Gain positif = le bloc améliore le classement.

| Bloc | MAE | Gain vs base | IC 95 % | Décision |
|---|---:|---:|---|---|
| `shots_luck` (7) | 2,622 | **+0,103** | [−0,048 ; +0,268] | **gardé** (déjà dans le set) |
| `special_teams` (3) | 2,715 | +0,008 | [−0,095 ; +0,101] | écarté |
| `playoff_history` (4) | 2,719 | 0,000 | [−0,103 ; +0,107] | écarté |
| `teamvs` (4) | 2,751 | −0,032 | [−0,127 ; +0,064] | écarté |
| `splits` (5) | 2,760 | −0,040 | [−0,146 ; +0,071] | écarté |
| `awards` (3) | 2,766 | −0,047 | [−0,138 ; +0,049] | écarté |
| `goalie` (5) | 2,891 | −0,166 | [−0,316 ; −0,016] | **écarté (significativement nuisible)** |
| `roster` (8) | 2,946 | −0,213 | [−0,379 ; −0,062] | **écarté (significativement nuisible)** |
| tous les blocs externes (39) | 2,860 | −0,134 | [−0,312 ; +0,031] | écarté |

Aucun nouvel intervalle n'est entièrement au-dessus de 0. Les deux seuls intervalles entièrement
sous 0 (`goalie`, `roster`) indiquent que ces blocs **dégradent** la prédiction : ils ajoutent du
bruit corrélé que la régression régularisée ne parvient pas à ignorer.

### 3.1 Sur le set déjà sélectionné

Référence : `prev_points` + `shots_luck` + `experience`, MAE **2,611**. C'est le test honnête :
« ce bloc apporte-t-il quelque chose que nous n'avons pas déjà ? »

| Bloc ajouté | MAE | Gain vs set courant | IC 95 % |
|---|---:|---:|---|
| `awards` | 2,641 | −0,032 | [−0,120 ; +0,055] |
| `playoff_history` | 2,641 | −0,032 | [−0,143 ; +0,086] |
| `special_teams` | 2,655 | −0,040 | [−0,156 ; +0,085] |
| `teamvs` | 2,657 | −0,047 | [−0,163 ; +0,063] |
| `splits` | 2,684 | −0,071 | [−0,145 ; +0,008] |
| `goalie` | 2,831 | −0,213 | [−0,345 ; −0,099] |
| `roster` | 2,894 | −0,269 | [−0,457 ; −0,109] |

**Conclusion : aucun nouveau bloc n'améliore le set sélectionné.**

### 3.2 Feature par feature

Les 39 features ont aussi été testées seules, ajoutées à la base historique. Aucune n'a d'IC
entièrement > 0. Les meilleures sont :

| Feature | Bloc | Gain | IC 95 % |
|---|---|---:|---|
| `shot_diff_lag1_z` | shots_luck | +0,071 | [−0,064 ; +0,217] |
| `shot_diff_pct_lag1_z` | shots_luck | +0,055 | [−0,078 ; +0,205] |
| `sv_pct_lag1_z` | shots_luck | +0,032 | [−0,032 ; +0,094] |
| `team_sv_lag1_z` | goalie | +0,032 | [−0,032 ; +0,094] |
| `pp_pct_lag1_z` | special_teams | +0,032 | [−0,032 ; +0,104] |
| `n_20goal_lag1_z` | roster | +0,024 | [−0,016 ; +0,079] |
| `home_pts_pct_lag1_z` | splits | +0,024 | [−0,064 ; +0,112] |

Les features de charge du gardien (`starter_gp_lag1_z`, `starter_min_lag1_z`) sont les seules à
avoir un IC entièrement négatif (−0,174 [−0,294 ; −0,048]).

## 4. Sélection finale

Sélection greedy forward, sur 1995–2004, en partant de `prev_points` :

| Étape | Groupe ajouté | MAE |
|---|---|---:|
| 0 | `prev_points` | 2,879 |
| 1 | `shots_luck` | 2,632 |
| 2 | `experience` | 2,611 |
| — | `rank_history` (gain 0,007 < seuil 0,01) | arrêt |

Set final : **`prev_points` + `shots_luck` + `experience`**, soit 9 features :

```
pts_pct_z_lag1,
sh_pct_lag1_z, sv_pct_lag1_z, pdo_lag1_z,
shot_diff_lag1_z, shot_diff_pct_lag1_z,
pdo_mean3_z, shot_diff_pct_mean3_z,
n_hist
```

`ranking/selected_features.json` est inchangé après l'ajout des nouveaux blocs : la sélection ne les
retient pas.

## 5. Blend et résultats finaux

### 5.1 Tableau final

| Split | Modèle | MAE | Spearman | Top-8 overlap | Gain vs naive | IC 95 % | Verdict |
|---|---|---:|---:|---:|---:|---|---|
| **Clean 2005–2009** | Naive précédent | 3,400 | 0,477 | 62,9 % | 0,000 | — | référence |
| | Ridge + external | 3,413 | 0,463 | 60,0 % | −0,013 | [−0,374 ; +0,280] | bruit |
| | **Blend (0,95 / 0,05)** | 3,413 | 0,464 | 61,4 % | −0,013 | [−0,374 ; +0,280] | bruit |
| | Ensemble + external | 3,400 | 0,474 | 62,9 % | 0,000 | [−0,413 ; +0,320] | bruit |
| **Backtest 1995–2009** | Naive précédent | 3,065 | 0,533 | 68,8 % | 0,000 | — | référence |
| | Ridge + external | 2,897 | 0,579 | 69,3 % | +0,168 | — | — |
| | **Blend (0,95 / 0,05)** | **2,893** | 0,577 | 70,3 % | **+0,174** | **[+0,010 ; +0,332]** | significatif (backtest seulement) |
| | Ensemble + external | 2,977 | 0,558 | 68,7 % | +0,088 | — | — |

Le détail complet est dans `ranking/results/summary.csv` ; le notebook
`ranking/ranking_results_analysis.ipynb` recalcule les intervalles et le verdict.

### 5.2 Lecture honnête

- **Sur le clean test 2005–2009, aucun modèle season-level ni le blend n'est significatif.** Ni les
  5 modèles, ni le blend ne battent le naive « previous available ranking » avec un IC entièrement
  > 0. Le meilleur « modèle » season-level du clean test est le naive lui-même (3,400, à égalité
  avec l'ensemble).
- **Le blend n'aide pas sur le clean test** (−0,013 [−0,374 ; +0,280]). Avec `w = 0,95`, il est
  presque identique à Ridge et hérite donc de son résultat. Le poids 0,95 plutôt que 1,00 est un
  artefact de la grille : le gain sur la sélection est de 0,007 position.
- **Sur le backtest, le blend est significatif** (+0,174 [+0,010 ; +0,332]). Mais 1995–2004 a servi
  à choisir le modèle, le poids et les features. Ce n'est **pas** une preuve propre : c'est un
  contrôle de stabilité.
- **Aucune piste testée n'est significative sur le clean test**, y compris la piste match-level
  optionnelle (§7) une fois évaluée sur de vrais rangs entiers. Les variantes essayées (recency,
  fenêtres glissantes, cibles alternatives, sous-ensembles de features, rank averaging, blend
  adaptatif, recalibration) ne changent pas le verdict. Le candidat le plus défendable reste
  `Ridge + recency (demi-vie 12 ans)`, choisi sur dev, mais son IC touche 0 (voir §8).
- **La parité NHL est le facteur dominant.** Le classement d'une saison est corrélé à celui de la
  saison précédente, mais la régression vers la moyenne est forte (~55–60 % de la déviation
  conservée). Les blocs riches (roster, goalies, splits, head-to-head) sont descriptivement
  intéressants mais n'ajoutent pas de signal hors échantillon au-delà de la qualité des tirs.

## 6. Pourquoi les nouveaux blocs échouent

- **Redondance.** Les splits, le head-to-head et l'historique de séries sont fortement corrélés au
  points percentage de *Y−1*, déjà dans le set. Une fois `prev_points` et `shots_luck` présents, ils
  n'apportent presque rien de conditionnellement indépendant.
- **Bruit.** Les blocs `roster` et `goalie` sont mesurés sur une seule saison et sont très instables
  d'une année sur l'autre (blessures, changements de gardien, départs). Le modèle linéaire leur donne
  un poids qui se retourne contre lui.
- **Petit échantillon.** 9 saisons de sélection, 15 équipes par conférence. Les écarts de 0,02–0,07
  position sont très inférieurs à la largeur des IC.
- **Pas de données match par match.** `TeamVsTeam` ne contient que des bilans agrégés par adversaire,
  pas le détail des rencontres. Impossible de construire un vrai momentum ou un vrai rating Elo
  séquentiel à partir de ces tables.

## 7. Piste match-level (optionnelle)

`ranking/ranking_match_level.ipynb` reconstruit un calendrier synthétique à partir des comptages de
`TeamVsTeam`, simule 4 000 saisons par saison (rating = points percentage laggé, avantage du terrain
fixe, nuls pour 1995–2003, prolongation pour 2005+), puis classe les équipes par **rang médian** de
la distribution simulée. Ce rang médian est ensuite transformé en **rang entier** (permutation avec
départage neutre : victoires de *Y−1*, puis `tmID`) avant toute évaluation.

| Split | Match-level MAE | Naive MAE | Gain vs naive | Direct lag-1 MAE | Gain vs direct |
|---|---:|---:|---|---:|---|
| Clean 2005–2009 | 3,373 | 3,400 | +0,027 [−0,040 ; +0,080] | 3,400 | +0,027 [−0,040 ; +0,080] |
| Backtest 1995–2009 | 3,041 | 3,065 | +0,025 [−0,010 ; +0,060] | 3,065 | +0,025 [−0,010 ; +0,060] |

**Correction importante.** Une version antérieure de ce notebook annonçait +0,107 [+0,053 ; +0,160]
sur le clean test. C'était un **artefact de métrique** : elle notait le rang médian **fractionnaire**,
qui n'est pas un classement. Un rang fractionnaire est mécaniquement plus proche du rang réel entier
qu'une permutation ne peut l'être. Avec un vrai rang entier, le gain tombe à +0,027 et l'IC contient 0.

- **Ce n'est pas plus de données.** Le rating est le même `pts_pct_z_lag1` que le naive. La
  décomposition match par match ne crée pas d'information.
- **Contre Ridge/blend, aucune différence significative** : +0,040 [−0,307 ; +0,467] vs Ridge et
  +0,040 [−0,293 ; +0,427] vs le blend sur le clean ; −0,144 [−0,314 ; +0,044] et
  −0,149 [−0,315 ; +0,030] sur le backtest.
- **Le bonus top-8 est absent** : 62,9 % pour le rang entier comme pour le contrôle direct ; la
  probabilité Monte Carlo (61,4 %) est même inférieure.
- **Contrôle « random tie-break »** : même rating, ordre aléatoire dans les égalités → MAE attendue
  3,373 sur le clean et 3,059 sur le backtest. Le modèle simulé ne fait pas mieux que ce contrôle.

Conclusion : **aucun signal match-level nouveau**. Le notebook est conservé comme expérience
documentée et comme générateur de probabilités (rang médian, P(top-8)), pas comme un gain de
performance. Il ne doit pas être présenté comme une façon de battre la baseline.

## 8. Dérive entre époques et comment faire mieux

Le détail exécutable est dans `ranking/ranking_era_drift.ipynb`. Résumé :

### 8.1 Le diagnostic

Corrélation de rang feature → cible, par époque :

| Feature | 1995–2003 | 2005–2009 | Δ |
|---|---:|---:|---:|
| `shot_diff_pct_mean3_z` | 0,501 | 0,214 | −0,288 |
| `n_hist` | 0,314 | 0,107 | −0,208 |
| `shot_diff_pct_lag1_z` | 0,554 | 0,350 | −0,204 |
| `shot_diff_lag1_z` | 0,552 | 0,348 | −0,204 |
| `pts_pct_z_lag1` | 0,594 | 0,506 | −0,088 |
| `sh_pct_lag1_z` | 0,232 | 0,166 | −0,066 |
| `pdo_mean3_z` | 0,363 | 0,307 | −0,056 |
| `pdo_lag1_z` | 0,292 | 0,310 | +0,019 |
| `sv_pct_lag1_z` | 0,209 | 0,302 | +0,093 |

Les blocs de **qualité de tir** perdent l'essentiel de leur pouvoir prédictif après 2004. Le points
percentage laggé reste la feature la plus stable — ce qui explique la force du naive.

Les **coefficients Ridge sont stables** entre les époques (écart-type ~0,006–0,047). Le modèle
n'adapte donc pas ses poids à la baisse de corrélation : il continue d'appliquer les mêmes
coefficients à des features devenues moins informatives. C'est visible par saison : Ridge bat le
naive presque partout avant 2004, puis perd en 2006 (−0,267) et 2008 (−0,533).

### 8.2 Les remèdes testés

Gain vs le modèle courant (`Ridge + 9 features`), walk-forward, IC 95 % apparié :

| Candidat | Dev MAE | Clean MAE | Clean gain | IC 95 % clean |
|---|---:|---:|---:|---|
| Ridge + 9 features (référence) | 2,611 | 3,413 | 0,000 | — |
| Recency, demi-vie 12 ans | 2,602 | 3,360 | +0,053 | [−0,000 ; +0,120] |
| Recency, demi-vie 20 ans | 2,617 | 3,347 | +0,067 | [−0,013 ; +0,160] |
| Fenêtre glissante 12 ans | 2,590 | 3,440 | −0,027 | [−0,173 ; +0,120] |
| Fenêtre glissante 10 ans | 2,605 | 3,453 | −0,040 | [−0,200 ; +0,107] |
| Cible `win_pct_z` | 2,594 | 3,387 | +0,027 | [−0,053 ; +0,107] |
| Cible `gd_z` | 2,620 | 3,400 | +0,013 | [−0,080 ; +0,120] |
| Core 4 features | 2,818 | 3,347 | +0,067 | [−0,200 ; +0,347] |
| Core 5 + pdo | 2,779 | 3,320 | +0,093 | [−0,200 ; +0,440] |
| Sans `n_hist` | 2,632 | 3,413 | 0,000 | — |

Aucun candidat n'est significatif. Les points importants :

- **`recency 12 ans`** est le compromis le plus net : mieux en dev *et* en clean, mais l'IC touche
  zéro. C'est un candidat documenté, pas une amélioration prouvée. La cible `win_pct_z` améliore
  aussi les deux périodes, mais son gain clean (+0,027 [−0,053 ; +0,107]) est encore plus fragile.
- **`recency 20 ans`** est meilleur en clean mais **pire en dev** : le retenir serait du
  cherry-picking sur le clean test.
- **`core 4` et `core 5 + pdo`** améliorent le clean mais dégradent fortement le dev. Ils illustrent
  la dérive, ils ne la corrigent pas : ils ne sont pas sélectionnables honnêtement.

### 8.3 Ce qu'il faudrait faire pour progresser

Par ordre de levier :

1. **Ajouter de l'information, pas des transformations.**
   - **Cotes / win totals de marché** avant la saison : le prédicteur connu le plus fort.
   - **Projections joueur par joueur** : buts/points projetés, âge, temps de jeu, qualité du gardien.
     Les agrégats annuels actuels sont trop instables.
   - **Blessures, masse salariale, contrats, choix de repêchage** : absents du dataset.
2. **Changer d'objectif.** La MAE du rang est grossière. Modéliser une distribution de rangs ou des
   comparaisons par paires (Plackett-Luce, Bradley-Terry), puis évaluer en log-loss / Brier.
3. **Augmenter la puissance du protocole** : plus de saisons (1950+ avec précautions), autres
   ligues, ou validation rolling-origin pré-enregistrée sur toutes les saisons disponibles. Avec
   10 classements de conférence dans le clean test, un gain < 0,1 position est indétectable.
4. **Si on reste sur les données actuelles** : garder `Ridge + 9 features` et le blend `w≈0,95` avec
   le naive ; documenter `recency 12 ans` avec sa réserve ; ne pas sélectionner sur le clean test.

Ce qu'il ne faut pas faire : choisir un modèle parce qu'il est meilleur sur 2005–2009, noter un rang
médian fractionnaire comme un classement, ou présenter un gain dont l'IC contient zéro comme une
amélioration.

## 9. Reproduire

```powershell
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_feature_analysis.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=3600
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_predictions.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=3600
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_results_analysis.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=1200
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_match_level.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=3600
jupyter nbconvert --to notebook --execute --inplace ranking\ranking_era_drift.ipynb --ExecutePreprocessor.kernel_name=ac_hockey --ExecutePreprocessor.timeout=1800
```

Ordre obligatoire pour les trois premiers : l'analyse des features réécrit
`selected_features.json`, les prédictions écrivent `results/`, l'analyse des résultats lit
`results/`. Le notebook de prédiction choisit et fige le blend sur 1995–2004 avant d'évaluer le
clean test. Les notebooks match-level et era-drift sont autonomes ; ils lisent
`results/predictions.csv` pour leurs comparaisons.
