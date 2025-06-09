# ipl-win-predection-ml-project
# 🏏 IPL Win Predictor using XGBoost 🚀

This project is a machine learning model that predicts the **probability of a team winning** an IPL match based on the current match situation — built using **XGBoost** and **Scikit-learn Pipelines**.

### 🎯 Objective

Predict win probability during a live match using features like:
- Batting team
- Bowling team
- First innings target
- Runs scored so far
- Wickets fallen
- Current over progress (including legal deliveries)
- Match city

---

## 📊 Dataset

**Kaggle Dataset**: [IPL Complete Dataset (2008-2020)](https://www.kaggle.com/datasets)

- `matches.csv`: match-level data
- `deliveries.csv`: ball-by-ball data

---

## ⚙️ ML Pipeline

### Features used
| Feature                | Type          |
|------------------------|---------------|
| Batting team           | Categorical   |
| Bowling team           | Categorical   |
| City                   | Categorical   |
| First innings runs     | Numeric       |
| Wickets fallen         | Numeric       |
| Cumulative runs scored | Numeric       |
| Current over float     | Numeric (computed) |

### Preprocessing
- **Categorical**: OneHotEncoder
- **Numerical**: StandardScaler

### Model
- **XGBoostClassifier**
    - `n_estimators=100`
    - `max_depth=7`
    - `learning_rate=0.15`
    - `eval_metric='logloss'`

---

## 🏆 Results

| Metric          | Score |
|-----------------|-------|
| Accuracy        | ~97%  |
| ROC AUC Score   | ~0.99 |

The model performs extremely well, as expected given the structured nature of IPL outcomes and feature selection.

---

## 💻 How to Use

### 1️⃣ Clone repo and install requirements
```bash
git clone https://github.com/yourusername/ipl-win-predictor.git
cd ipl-win-predictor
pip install -r requirements.txt
