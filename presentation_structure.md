# Presentation: Predicting Coach Changes in the NHL

## Slide 1: Title and Introduction
- **Project title:** Predicting NHL Coach Dismissals.
- **Names:** [Jules COLOMBU]
- **Date:** [Date]
- **Context:** Data Mining / Machine Learning project.
- **Problem statement:** How can we help a General Manager (GM) anticipate crises within their team?

## Slide 2: Business Understanding
- **Business objective:** A GM wants to know whether their coach is at risk, in order to prepare a backup plan (scouting, tactical changes).
- **Data Mining objective:** This is a **binary classification** problem (0 = No change, 1 = Change).
- **Main challenge:** The data is **highly imbalanced** (a change is a rare event). A model that always predicts "0" reaches 84% accuracy but is useless. We must optimize the **F1-Score**.

## Slide 3: Data Understanding & Preparation (1/2)
- **Data source:** Historical relational NHL database (since 1979).
- **Target construction:** A change is defined as a franchise having more than one coach "stint" (period) in the same season.
- **Target analysis:** [Insert the target distribution chart].
- **Conclusion:** The imbalance is confirmed (only ~16% of changes). The modeling strategy is validated.

## Slide 4: Feature Engineering (The Turning Point)
- **Our initial approach:** We had created a highly performant `coach_tenure` feature (coach seniority).
- **The problem (Data Leakage):** While analyzing feature importance, we discovered that `coach_tenure` used the target variable (the current year) in its computation. The model was cheating!
- **The fix:** We recoded `coach_tenure` so that it looks **only at the past**. The F1-score dropped from 0.72 to 0.31, which is far more realistic.
- **New features:** To compensate, we extracted new data from the database schema (Power Play %, Penalty Kill, previous year's goal differential).

## Slide 5: Exploratory Data Analysis (EDA)
- **Relationship visualization:** [Insert the boxplots chart].
- **Business interpretation:**
    - Teams that change coach tend to have fewer points and a worse goal differential the previous year.
    - Instability (frequent changes in the last 3 years) is a strong indicator.
    - An ineffective Power Play is a pressure factor.

## Slide 6: Modeling (Approach and Challenges)
- **Chronological split:** Use of `TimeSeriesSplit` to train on the past and test on the future, simulating real conditions.
- **Threshold optimization:** Instead of using a 0.5 probability threshold, we wrote a function that finds the **optimal threshold** maximizing the F1-Score.
- **Model diversity:** We tested Logistic Regression, SVM, Random Forest, Gradient Boosting and **XGBoost**.

## Slide 7: Results and Metrics (The Core of the Analysis)
- **Why not Accuracy?** The naive model has 84% accuracy but an F1-Score of 0.
- **Advanced metrics:** We used **Balanced Accuracy** and **Macro F1** (which penalizes models that ignore the minority class).
- **The winner: XGBoost.** It offers the best balance with an **F1-Score of 0.31** and a **Balanced Accuracy of 0.59**.
- [Insert the grouped chart with all metrics].

## Slide 8: Model Interpretation (XGBoost)
- **Confusion matrix:** [Insert the confusion matrix]. The model is conservative: it only predicts a dismissal when it is very confident (high precision), which is ideal for a decision-support tool.
- **Precision-Recall curve:** [Insert the PR curve]. More informative than the ROC curve for imbalanced data.
- **Feature importance:** [Insert the feature importance chart]. The key factors are:
    1.  Performance regression (`pts_change_1y`).
    2.  Historical instability (`changes_last_3y`).
    3.  Ineffective Power Play (`prev_PP_pct`).

## Slide 9: Conclusion and Limitations
- **What we learned:**
    - Predicting a coach dismissal is a hard problem (F1 ~0.31 is a good score).
    - The most important data is the analysis of past performance.
    - Detecting the data leakage was crucial.
- **Limitations:**
    - The model does not capture internal dynamics (locker-room conflicts, injuries).
    - The binary target (0/1) is simplistic; a survival analysis would be finer.

## Slide 10: Improvement Ideas
- **Features:** Integrate player data (star departures, injuries).
- **Modeling:** Test time-series models (LSTM) or survival models.
- **Tools:** Build an interactive dashboard (Streamlit, Dash) for GMs.

## Slide 11: Questions / Answers
- [Acknowledgements]
