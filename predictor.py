import os
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from pulp import LpProblem, LpVariable, LpMaximize, lpSum, value, PULP_CBC_CMD


#Importing GenAI 
try:
    from ai_part_final import compute_shap_for_squad, generate_all_commentaries, generate_team_audio
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False



ROLES = ['BAT', 'BOWL', 'AR', 'WK']

_ilp_call_counter = [0]

def select_team_ilp(pred_df: pd.DataFrame, problem_name: str = "Dream11") -> pd.DataFrame:

    pred_df = pred_df.reset_index(drop=True).copy()
    n       = len(pred_df)

    _ilp_call_counter[0] += 1
    uid = _ilp_call_counter[0]

    prob = LpProblem(f"{problem_name}_{uid}", LpMaximize)
    x    = [LpVariable(f"x_{uid}_{i}", cat='Binary') for i in range(n)]


    prob += lpSum(pred_df['predicted_fp'].iloc[i] * x[i] for i in range(n))


    prob += lpSum(x) == 11

    for role in ROLES:
        idx = pred_df.index[pred_df['role'] == role].tolist()
        if idx:
            prob += lpSum(x[i] for i in idx) >= 1
            prob += lpSum(x[i] for i in idx) <= 8

    for team in pred_df['team'].unique():
        idx = pred_df.index[pred_df['team'] == team].tolist()
        if idx:
            prob += lpSum(x[i] for i in idx) >= 1

    prob.solve(PULP_CBC_CMD(msg=0))

    selected = pred_df[[value(x[i]) == 1 for i in range(n)]].copy()
    return selected.sort_values('predicted_fp', ascending=False).reset_index(drop=True)


def get_confidence(career_matches: int) -> str:
    if career_matches < 5:  return 'Low'
    if career_matches < 15: return 'Medium'
    return 'High'


_loaded_artifact = None

def get_model_artifact():
    """Caches the model in memory so it only loads from the hard drive ONCE."""
    global _loaded_artifact
    if _loaded_artifact is None:
        with open('src/model_artifacts/ProductUI_Model', 'rb') as f:
            _loaded_artifact = pickle.load(f)
    return _loaded_artifact



def predict_team(squad_df: pd.DataFrame, team1="Team 1", team2="Team 2", generate_ai=False) -> tuple:
    squad_df = squad_df.reset_index(drop=True).copy()

    # Load model from memory, not disk
    artifact = get_model_artifact()
    models_loaded = artifact['models']
    feature_cols = artifact['feature_cols']

    if 'WK' not in squad_df['role'].values:
        bat_rows = squad_df[squad_df['role'] == 'BAT']
        if not bat_rows.empty:
            squad_df.loc[bat_rows.index[0], 'role'] = 'WK'

    #BATCH PREDICTION 
    squad_df['predicted_fp'] = 0.0
    for role in squad_df['role'].unique():
        role_mask = squad_df['role'] == role
        model = models_loaded.get(role, models_loaded['BAT'])
        X = squad_df.loc[role_mask, feature_cols].fillna(0)
        predictions = model.predict(X)
        squad_df.loc[role_mask, 'predicted_fp'] = np.maximum(0.0, np.round(predictions, 2))

    squad_df['confidence'] = squad_df['career_matches'].apply(
        lambda x: get_confidence(int(x)) if pd.notnull(x) else 'Low'
    )

    # Run the Integer Linear Programming (ILP) optimizer
    selected = select_team_ilp(squad_df, problem_name="predict")

    selected['captain'] = False
    selected['vc']      = False
    selected.loc[0, 'captain'] = True
    selected.loc[1, 'vc']      = True

    selected['display_fp'] = selected['predicted_fp']
    selected.loc[selected['captain'], 'display_fp'] *= 2.0
    selected.loc[selected['vc'],      'display_fp'] *= 1.5

   
    audio_path = None
    
    
    #GENAI PIPELINE 
    if generate_ai:
        try:
            # Attempt to generate LLM text
            selected = generate_all_commentaries(selected)
            
            # Attempt to generate TTS Audio
            audio_path = generate_team_audio(selected, team1, team2, save_path="src/model_artifacts/team_summary.mp3")
            
        except Exception as e:
            print(f"Network/AI Error: {e}")
            # If offline or API fails, fallback to basic ML safely
            if 'commentary' not in selected.columns:
                selected['commentary'] = "Offline Mode: AI insight currently unavailable."
            audio_path = None
    else:
        selected['commentary'] = ""
        audio_path = None


    return selected, audio_path


def get_actual_dream_team(match_df: pd.DataFrame) -> pd.DataFrame:

    match_df = match_df.reset_index(drop=True).copy()
    eval_df  = match_df[['player', 'team', 'role']].copy()
    eval_df['predicted_fp'] = match_df['fantasy_points'].values
    return select_team_ilp(eval_df, problem_name="actual")







if __name__=="__main__":
    df = pd.read_csv("cleaned_dataset.csv", low_memory=False)
    df['date']     = pd.to_datetime(df['date'])
    df['match_id'] = df['match_id'].astype(str)


    TARGET = 'fantasy_points'

    DROP_COLS = [
    'match_id', 'date', 'venue', 'match_type',
    'team', 'opposition', 'player', 'role', 'dismissal_kind'
    ]

    LEAKY_COLS = [
    'balls_per_boundary', 'boundary_percent', 'boundary_runs',
    'bowling_sr', 'dot_ball_percent', 'dots', 'extras',
    'fielding_points_raw', 'fifty_plus', 'five_wicket_haul',
    'hundred', 'is_out', 'maiden_overs', 'overs', 'three_wicket_haul',
    'venue_high_scoring'
    ]
    

    FEATURE_COLS = [
    col for col in df.columns
    if col not in DROP_COLS + [TARGET] + LEAKY_COLS
    ]

    ROLES = ['BAT', 'BOWL', 'AR', 'WK']



    CUTOFF    = '2024-06-30'
    VAL_START = '2024-05-01'

    train_df = df[df['date'] <= pd.Timestamp(CUTOFF)].copy()
    test_df  = df[df['date'] >  pd.Timestamp(CUTOFF)].copy()

    assert train_df['date'].max() <= pd.Timestamp(CUTOFF), "LEAKAGE: training data past cutoff"
    assert len(test_df[test_df['date'] <= pd.Timestamp(CUTOFF)]) == 0, "Test set contaminated"

    local_train = train_df[train_df['date'] <  VAL_START]
    local_val   = train_df[train_df['date'] >= VAL_START]

    print(f"local_train : {len(local_train):,} rows")
    print(f"local_val   : {len(local_val):,} rows")
    print(f"train_full  : {len(train_df):,} rows")
    print(f"test        : {len(test_df):,} rows")



    BEST_PARAMS = {
    'BAT': {
        'n_estimators': 408, 'max_depth': 8,
        'learning_rate': 0.012925, 'subsample': 0.7776,
        'colsample_bytree': 0.6359, 'min_child_weight': 6,
        'random_state': 42, 'n_jobs': 1
    },
    'BOWL': {
        'n_estimators': 437, 'max_depth': 6,
        'learning_rate': 0.018627, 'subsample': 0.7655,
        'colsample_bytree': 0.7753, 'min_child_weight': 6,
        'random_state': 42, 'n_jobs': 1
    },
    'AR': {
        'n_estimators': 378, 'max_depth': 7,
        'learning_rate': 0.011623, 'subsample': 0.8169,
        'colsample_bytree': 0.6415, 'min_child_weight': 2,
        'random_state': 42, 'n_jobs': 1
    },
    'WK': {
        'n_estimators': 674, 'max_depth': 6,
        'learning_rate': 0.014482, 'subsample': 0.8677,
        'colsample_bytree': 0.9286, 'min_child_weight': 9,
        'random_state': 42, 'n_jobs': 1
    },
    }


    print("\nValidation MAEs (local split):")
    val_maes = {}

    for role in ROLES:
        r_train = local_train[local_train['role'] == role]
        r_val   = local_val[local_val['role'] == role]

        model = XGBRegressor(**BEST_PARAMS[role])
        model.fit(r_train[FEATURE_COLS].fillna(0), r_train[TARGET])

        preds          = model.predict(r_val[FEATURE_COLS].fillna(0))
        mae            = mean_absolute_error(r_val[TARGET], preds)
        val_maes[role] = mae
        print(f"  {role:<5} MAE: {mae:.2f}")

    total_rows   = sum(len(local_val[local_val['role'] == r]) for r in ROLES)
    weighted_mae = sum(val_maes[r] * len(local_val[local_val['role'] == r]) for r in ROLES) / total_rows
    print(f"  Weighted MAE: {weighted_mae:.2f}")



    print("\nTraining final models on full training set:")
    final_models = {}

    for role in ROLES:
        role_data = train_df[train_df['role'] == role]
        model     = XGBRegressor(**BEST_PARAMS[role])
        model.fit(role_data[FEATURE_COLS].fillna(0), role_data[TARGET])
        final_models[role] = model
        print(f"  {role:<5} trained on {len(role_data):,} rows")



    print("\nTop 10 features per role:")
    for role in ROLES:
        importance = pd.DataFrame({
        'feature':    FEATURE_COLS,
        'importance': final_models[role].feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n{role}:")
        print(importance.head(10).to_string(index=False))



    os.makedirs('src/model_artifacts', exist_ok=True)

    artifact = {
    'models':          final_models,
    'feature_cols':    FEATURE_COLS,
    'training_cutoff': CUTOFF,
    }

    with open('src/model_artifacts/ProductUI_Model', 'wb') as f:
        pickle.dump(artifact, f)

    print("\nProductUI_Model saved -> src/model_artifacts/ProductUI_Model")



    print("\nEvaluating on 20 test matches...")
    match_ids = test_df['match_id'].unique()[:20]
    mae_list  = []

    for match_id in match_ids:
        match_df = test_df[test_df['match_id'] == match_id].copy().reset_index(drop=True)

        if len(match_df) < 11:
            print(f"  Match {match_id} skipped — only {len(match_df)} players")
            continue

   
        pred_team, _ = predict_team(match_df)

  
        actual_team = get_actual_dream_team(match_df)

   
        fp_map      = match_df.set_index('player')['fantasy_points'].to_dict()
        pred_total  = sum(fp_map.get(p, 0) for p in pred_team['player'])
        dream_total = actual_team['predicted_fp'].sum()

  
        assert dream_total <= match_df['fantasy_points'].sum() + 1, \
            f"Match {match_id}: impossible dream total {dream_total} — bug in ILP"

        mae = abs(dream_total - pred_total)
        mae_list.append(mae)

        print(f"  Match {match_id}  |  Dream: {dream_total:.1f}  |  Predicted: {pred_total:.1f}  |  MAE: {mae:.1f}")

    print(f"\nMean MAE : {np.mean(mae_list):.2f}")
    print(f"Std  MAE : {np.std(mae_list):.2f}")


    print("\n" + "=" * 60)
    print("SMOKE TEST — First test match")
    print("=" * 60)

    sample_match = test_df[test_df['match_id'] == test_df['match_id'].iloc[0]].copy().reset_index(drop=True)

    pred_team , _  = predict_team(sample_match)
    actual_team = get_actual_dream_team(sample_match)

    fp_map      = sample_match.set_index('player')['fantasy_points'].to_dict()
    pred_total  = sum(fp_map.get(p, 0) for p in pred_team['player'])
    dream_total = actual_team['predicted_fp'].sum()

    print(f"\nDream Team total  : {dream_total:.1f}")
    print(f"Your team total   : {pred_total:.1f}")
    print(f"Gap (MAE)         : {abs(dream_total - pred_total):.1f}")

    print("\nYour predicted team:")
    print(pred_team[['player', 'team', 'role', 'predicted_fp', 'display_fp', 'confidence', 'captain', 'vc']].to_string(index=False))

    print("\nActual dream team:")
    actual_team['actual_fp'] = actual_team['predicted_fp']
    print(actual_team[['player', 'team', 'role', 'actual_fp']].to_string(index=False))
