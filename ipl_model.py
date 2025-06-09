import numpy as np
import xgboost as xgb
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

match=pd.read_csv('/kaggle/input/ipl-complete-dataset-20082020/matches.csv')
delevry=pd.read_csv('/kaggle/input/ipl-complete-dataset-20082020/deliveries.csv')

match.rename(columns={'target_runs': 'first_ining_runs'}, inplace=True)

total_score_df=delevry.groupby(['match_id','inning']).sum()['total_runs'].reset_index()

total_score_df=total_score_df[total_score_df['inning']==2]

match=match.merge(total_score_df[['match_id','total_runs']], left_on='id', right_on='match_id')

match.rename(columns={'total_runs': 'second_ining_runs'}, inplace=True)

teams=[
    'Royal Challengers Bangalore',
    'Mumbai Indians', 
    'Kolkata Knight Riders',
    'Rajasthan Royals',
     'Chennai Super Kings',
    'Sunrisers Hyderabad',
    'Delhi Capitals',
    'Punjab Kings',
    'Lucknow Super Giants',
    'Gujarat Titans'
]

match['team1']=match['team1'].str.replace('Delhi Daredevils','Delhi Capitals')
match['team2']=match['team2'].str.replace('Delhi Daredevils','Delhi Capitals')

match['team1']=match['team1'].str.replace('Kings XI Punjab','Punjab Kings')
match['team2']=match['team2'].str.replace('Kings XI Punjab','Punjab Kings')

match['team1']=match['team1'].str.replace('Deccan Chargers','Sunrisers Hyderabad')
match['team2']=match['team2'].str.replace('Deccan Chargers','Sunrisers Hyderabad')

match['team1']=match['team1'].str.replace('Royal Challengers Bengaluru','Royal Challengers Bangalore')
match['team2']=match['team2'].str.replace('Royal Challengers Bengaluru','Royal Challengers Bangalore')

match['team1']=match['team1'].str.replace('Gujarat Lions','Gujarat Titans')
match['team2']=match['team2'].str.replace('Gujarat Lions','Gujarat Titans')

match=match[match['team1'].isin(teams)]
match=match[match['team2'].isin(teams)]
match=match[match['method'].isna()]
match=match[match['super_over'].isin(['N'])]

match['win_label'] = np.where(match['winner'] == match['team1'], 1, 0)

delevry['cumulative_runs'] = delevry.groupby(['match_id', 'inning'])['total_runs'].cumsum()

delevry=delevry[delevry['inning'] == 2]

# First prepare a dataframe with match_id and first_ining_runs
target_df = match[['id', 'first_ining_runs']].rename(columns={'id': 'match_id'})

# Now merge it into delevry
delevry = delevry.merge(target_df, on='match_id', how='left')

# First prepare a dataframe with match_id and first_ining_runs
target_df = match[['id', 'win_label']].rename(columns={'id': 'match_id'})

# Now merge it into delevry
delevry = delevry.merge(target_df, on='match_id', how='left')

delevry['runs_left_to_win']=delevry['first_ining_runs']-delevry['cumulative_runs']+1

delevry['wicket_fallen']=delevry.groupby(['match_id', 'inning'])['is_wicket'].cumsum()

delevry['is_legal_delivery'] = np.where( (delevry['extras_type']=='legbyes') | (delevry['extra_runs'] == 0 ), 1, 0 )
# Now compute legal ball number inside current over:
delevry['legal_ball_in_over'] = delevry.groupby(['match_id', 'inning', 'over'])['is_legal_delivery'].cumsum()

# Now compute current_over_float:
delevry['current_over_float'] = delevry['over'] + (delevry['legal_ball_in_over']) / 10


X = delevry[['batting_team','bowling_team','first_ining_runs','wicket_fallen','cumulative_runs','current_over_float']]
y = delevry['win_label']

X = X.dropna()
y = y.loc[X.index]  # keep y aligned with X after dropping rows

# Define feature columns
categorical_cols = ['batting_team', 'bowling_team','city']  # categorical features
numeric_cols = ['first_ining_runs', 'wicket_fallen', 'cumulative_runs', 'current_over_float']  # numeric features

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])

# Create XGBoost classifier
xgb_clf = xgb.XGBClassifier(
    use_label_encoder=False,
    eval_metric='logloss',
    n_estimators=100,
    max_depth=7,
    learning_rate=0.15,
    random_state=42
)

# Full pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', xgb_clf)
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)

# Fit model
model.fit(X_train, y_train)

# Predict probabilities
win_probs = model.predict_proba(X_test)[:, 1]

# Evaluate
accuracy = accuracy_score(y_test, model.predict(X_test))
roc_auc = roc_auc_score(y_test, win_probs)

print(f"Accuracy: {accuracy:.2f}")
print(f"ROC AUC Score: {roc_auc:.2f}")
import random
# Show some predicted win probabilities vs actual
for i in range(5):
    j=random.randint(10,len(win_probs))
    print(f"Predicted Win %: {win_probs[j]*100:.1f}%, Actual: {y_test.iloc[j]}")


def predict_custom_input():
    print("==== IPL Win Probability Predictor ====\n")

    # List of valid options
    teams = [
        'Royal Challengers Bangalore',
        'Mumbai Indians', 
        'Kolkata Knight Riders',
        'Rajasthan Royals',
        'Chennai Super Kings',
        'Sunrisers Hyderabad',
        'Delhi Capitals',
        'Punjab Kings',
        'Lucknow Super Giants',
        'Gujarat Titans'
    ]
    
    cities = match['city'].dropna().unique().tolist()
    
    # Show team options
    print("Available Teams:")
    for i, team in enumerate(teams):
        print(f"{i+1}. {team}")
    
    batting_team_idx = int(input("\nSelect Batting Team (Enter number): ")) - 1
    bowling_team_idx = int(input("Select Bowling Team (Enter number): ")) - 1
    
    batting_team = teams[batting_team_idx]
    bowling_team = teams[bowling_team_idx]
    
    # Show city options
    print("\nAvailable Cities:")
    for i, city in enumerate(cities):
        print(f"{i+1}. {city}")
    
    city_idx = int(input("\nSelect City (Enter number): ")) - 1
    city = cities[city_idx]

    # Remaining inputs
    first_ining_runs = int(input("\nFirst inning total runs (target): "))
    wickets_fallen = int(input("Wickets fallen so far: "))
    cumulative_runs = int(input("Runs scored so far: "))
    current_over_float = float(input("Current over (e.g., 10.3 means 10 overs + 3 balls): "))

    # Prepare DataFrame
    input_df = pd.DataFrame({
        'batting_team': [batting_team],
        'bowling_team': [bowling_team],
        'city': [city],
        'first_ining_runs': [first_ining_runs],
        'wicket_fallen': [wickets_fallen],
        'cumulative_runs': [cumulative_runs],
        'current_over_float': [current_over_float]
    })

    # Predict
    win_prob = model.predict_proba(input_df)[0][1]

    print(f"\n==== Result ====")
    print(f"Predicted win probability for {batting_team}: {win_prob*100:.2f}%\n")

