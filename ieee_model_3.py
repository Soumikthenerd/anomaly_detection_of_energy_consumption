import gc
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn import metrics
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import torch
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Hardware & Configuration
# ---------------------------------------------------------
DATA_PATH = '/home/soumik-mondal/Downloads/LEAD/clean_reduced_lead_dataset.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS = 5

print(f'Active Device: {DEVICE}')
if DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')

EXCLUDE_FROM_X = [
    'anomaly', 'building_id', 'timestamp',
    'gte_building_id', 'gte_meter_building_id',
    'gte_meter_building_id_hour', 'gte_meter_building_id_weekday',
    'gte_meter_building_id_month', 'weekday_hour',
    'building_weekday_hour', 'building_weekday',
    'building_month', 'building_hour', 'building_meter',
]


def downcast_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        col_type = df[col].dtype
        if col_type == 'float64':
            df[col] = df[col].astype(np.float32)
        elif col_type == 'int64':
            c_min, c_max = df[col].min(), df[col].max()
            if c_min >= -32768 and c_max <= 32767:
                df[col] = df[col].astype(np.int16)
            elif c_min >= -2147483648 and c_max <= 2147483647:
                df[col] = df[col].astype(np.int32)
    return df


# ---------------------------------------------------------
# 2. V3 Feature Engineering (Diurnal Profiles + Min/Max Clipping + Acceleration)
# ---------------------------------------------------------
def build_v3_features(file_path: str):
    print(f'\nLoading {file_path}...')
    df = pd.read_csv(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

    print('Engineering V3 diurnal schedules, rolling envelopes & sensor clipping features...')
    grp = df.groupby('building_id')['meter_reading']

    # 1. Physical & Weather Normalization
    df['meter_per_sqft'] = df['meter_reading'] / (df['square_feet'] + 1.0)
    df['temp_power_interaction'] = df['meter_per_sqft'] * (df['air_temperature'] + 30.0)
    df['is_weekend'] = (df['weekday'] >= 5).astype(np.int8)

    # 2. Diurnal Hour-of-Week Baseline per Building (strictly on input meter_reading -> zero target leak)
    diurnal_grp = df.groupby(['building_id', 'hour', 'is_weekend'])['meter_reading']
    diurnal_med = diurnal_grp.transform('median')
    diurnal_q75 = diurnal_grp.transform(lambda s: s.quantile(0.75))
    diurnal_q25 = diurnal_grp.transform(lambda s: s.quantile(0.25))
    diurnal_iqr = (diurnal_q75 - diurnal_q25) + 1e-3

    df['diurnal_ratio'] = (df['meter_reading'] / (diurnal_med + 1e-3)).clip(0.0, 10.0)
    df['diurnal_robust_z'] = ((df['meter_reading'] - diurnal_med) / diurnal_iqr).clip(-15.0, 15.0)
    df['diurnal_diff_per_sqft'] = (df['meter_reading'] - diurnal_med) / (df['square_feet'] + 1.0)

    # 3. Multi-Scale Rolling Statistics (6h, 24h, 168h)
    df['rolling_mean_24h'] = grp.transform(lambda s: s.rolling(24, min_periods=1).mean())
    df['rolling_std_24h'] = grp.transform(lambda s: s.rolling(24, min_periods=1).std()).fillna(0.0)
    df['rolling_std_6h'] = grp.transform(lambda s: s.rolling(6, min_periods=1).std()).fillna(0.0)

    rm_168 = grp.transform(lambda s: s.rolling(168, min_periods=24).mean())
    df['rolling_mean_168h'] = rm_168.fillna(df['rolling_mean_24h'])

    rmin_168 = grp.transform(lambda s: s.rolling(168, min_periods=24).min()).fillna(df['meter_reading'])
    rmax_168 = grp.transform(lambda s: s.rolling(168, min_periods=24).max()).fillna(df['meter_reading'])
    df['pos_in_168h_range'] = (
        (df['meter_reading'] - rmin_168) / ((rmax_168 - rmin_168) + 1e-3)
    ).clip(0.0, 1.0)

    # Centered 24h & 72h rolling windows
    rm_center_24 = grp.transform(lambda s: s.rolling(25, min_periods=1, center=True).mean())
    rm_center_72 = grp.transform(lambda s: s.rolling(73, min_periods=1, center=True).mean())
    df['diff_from_centered_24h'] = df['meter_reading'] - rm_center_24
    df['ratio_to_centered_72h'] = (df['meter_reading'] / (rm_center_72 + 1e-3)).clip(0.0, 10.0)

    # Building-level global robust stats & clipping flags
    b_min = grp.transform('min')
    b_max = grp.transform('max')
    b_median = grp.transform('median')
    b_iqr = (grp.transform(lambda s: s.quantile(0.75)) - grp.transform(lambda s: s.quantile(0.25))) + 1e-3

    df['robust_z_bldg'] = ((df['meter_reading'] - b_median) / b_iqr).clip(-15.0, 15.0)
    df['is_at_bldg_min'] = ((df['meter_reading'] - b_min).abs() < 1e-4).astype(np.int8)
    df['is_at_bldg_max'] = ((b_max - df['meter_reading']).abs() < 1e-4).astype(np.int8)

    df['z_score_24h'] = (
        (df['meter_reading'] - df['rolling_mean_24h']) / (df['rolling_std_24h'] + 1e-5)
    ).clip(-10.0, 10.0)
    df['ratio_to_168h_mean'] = (
        df['meter_reading'] / (df['rolling_mean_168h'] + 1e-3)
    ).clip(0.0, 10.0)

    # 4. Bidirectional Lags, Leads & Second-Order Acceleration
    df['lag_diff_1h'] = grp.diff(1).fillna(0.0)
    df['lag_diff_2h'] = grp.diff(2).fillna(0.0)
    df['lag_diff_24h'] = grp.diff(24).fillna(0.0)
    df['lag_diff_168h'] = grp.diff(168).fillna(0.0)

    df['lead_diff_1h'] = grp.diff(-1).fillna(0.0)
    df['lead_diff_2h'] = grp.diff(-2).fillna(0.0)
    df['lead_diff_24h'] = grp.diff(-24).fillna(0.0)
    df['lead_diff_168h'] = grp.diff(-168).fillna(0.0)

    # Second-order difference (detects artificial linear interpolation ramps where accel == 0)
    df['accel_1h'] = (df['lag_diff_1h'] + df['lead_diff_1h']).astype(np.float32)

    # Normalized step jumps (scale-invariant across small and huge buildings)
    df['norm_jump_prev_1h'] = (df['lag_diff_1h'] / (diurnal_med + 1.0)).clip(-10.0, 10.0)
    df['norm_jump_next_1h'] = (df['lead_diff_1h'] / (diurnal_med + 1.0)).clip(-10.0, 10.0)

    # 5. Exact Sensor Stagnation / Flatline Streak Counter
    is_same_as_prev = (df['lag_diff_1h'].abs() < 1e-6) & (df['building_id'] == df['building_id'].shift(1))
    streak_group = (~is_same_as_prev).cumsum()
    df['flatline_streak_hours'] = df.groupby(streak_group).cumcount().clip(0, 168).astype(np.int16)

    if 'primary_use' in df.columns:
        df['primary_use'] = df['primary_use'].astype('category')

    groups = df['building_id'].values
    y = df['anomaly'].astype(np.int8)
    feature_cols = [c for c in df.columns if c not in EXCLUDE_FROM_X]
    X = downcast_dtypes(df[feature_cols].copy())

    del df
    gc.collect()

    # Exact same 80/20 zero-shot split as V1 & V2
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train, g_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy(), groups[train_idx]
    X_val, y_val, g_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy(), groups[val_idx]

    raw_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    sqrt_weight = float(np.sqrt(raw_ratio))

    print(f'Train Rows: {len(X_train):,} | Val Rows: {len(X_val):,}')
    print(f'Total V3 Features: {len(feature_cols)} | Calibrated sqrt(weight): {sqrt_weight:.2f}')

    return X_train, y_train, g_train, X_val, y_val, g_val, sqrt_weight


# ---------------------------------------------------------
# 3. Evaluation & Logit-Space Simplex Optimizer
# ---------------------------------------------------------
def find_best_f1_threshold(y_true, y_scores):
    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_scores)
    f1_scores = np.divide(
        2 * (precisions * recalls),
        (precisions + recalls),
        out=np.zeros_like(precisions),
        where=(precisions + recalls) != 0,
    )
    best_idx = int(np.argmax(f1_scores))
    best_th = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
    return best_th, float(f1_scores[best_idx])


def to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs, 1e-6, 1.0 - 1e-6)
    return logit(p)


def optimize_logit_weights(oof_list, y_true):
    """Finds optimal non-negative weights in logit space maximizing OOF PR-AUC."""
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(y_true), size=min(400000, len(y_true)), replace=False)
    y_sub = y_true.values[sub_idx]
    logits_sub = np.column_stack([to_logits(p[sub_idx]) for p in oof_list])

    best_score = -1.0
    best_w = np.ones(len(oof_list)) / len(oof_list)

    # Dirichlet random + grid search over simplex (guaranteed no Nelder-Mead stalling)
    candidates = rng.dirichlet(np.ones(len(oof_list)), size=1500)
    # Include uniform and pairwise anchors
    candidates = np.vstack([
        candidates,
        np.eye(len(oof_list)),
        np.ones((1, len(oof_list))) / len(oof_list),
    ])

    for w in candidates:
        blended_probs = expit(logits_sub @ w)
        score = metrics.average_precision_score(y_sub, blended_probs)
        if score > best_score:
            best_score = score
            best_w = w

    return best_w, best_score


def evaluate_predictions(y_true, y_scores, model_name: str, fixed_thresh: float = None):
    roc_auc = metrics.roc_auc_score(y_true, y_scores)
    pr_auc = metrics.average_precision_score(y_true, y_scores)
    best_th, best_f1 = find_best_f1_threshold(y_true, y_scores)

    eval_th = fixed_thresh if fixed_thresh is not None else best_th
    preds = (y_scores >= eval_th).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, preds).ravel()

    prec = metrics.precision_score(y_true, preds, zero_division=0)
    rec = metrics.recall_score(y_true, preds, zero_division=0)
    f1_at_th = metrics.f1_score(y_true, preds, zero_division=0)
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = metrics.balanced_accuracy_score(y_true, preds)
    mcc = metrics.matthews_corrcoef(y_true, preds)

    print('=' * 86)
    print(f'📊 {model_name}')
    print('=' * 86)
    print(f'  • PR-AUC (Avg Prec): {pr_auc:.5f}  |  ROC-AUC:           {roc_auc:.5f}')
    print(f'  • Peak F1-Score:     {best_f1:.5f}  |  Peak Threshold:    {best_th:.4f}')
    if fixed_thresh is not None:
        print(f'  • OOF-Thresh F1:     {f1_at_th:.5f}  |  OOF Threshold:     {fixed_thresh:.4f}')
    print(f'  • Precision:         {prec:.4f}   |  Recall:            {rec:.4f}')
    print(f'  • Specificity:       {spec:.4f}   |  Balanced Acc:      {bal_acc:.4f}  |  MCC: {mcc:.4f}')
    print(f'  • Confusion Matrix:  TP={tp} | FP={fp} | FN={fn} | TN={tn}')
    print('=' * 86 + '\n')

    return {
        'Model': model_name,
        'PR-AUC': round(pr_auc, 5),
        'ROC-AUC': round(roc_auc, 5),
        'Peak F1': round(best_f1, 5),
        'Precision': round(prec, 4),
        'Recall': round(rec, 4),
        'MCC': round(mcc, 4),
        'Opt_Thresh': round(eval_th, 4),
    }


# ---------------------------------------------------------
# 4. 5-Fold Diverse GBDT Training & Logit Ensembling
# ---------------------------------------------------------
def run_v3_pipeline(X_train, y_train, g_train, X_val, y_val, sqrt_weight: float):
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_xgb8 = np.zeros(len(X_train), dtype=np.float32)
    oof_xgb10 = np.zeros(len(X_train), dtype=np.float32)
    oof_lgb = np.zeros(len(X_train), dtype=np.float32)
    oof_lgb_et = np.zeros(len(X_train), dtype=np.float32)

    val_xgb8 = np.zeros(len(X_val), dtype=np.float32)
    val_xgb10 = np.zeros(len(X_val), dtype=np.float32)
    val_lgb = np.zeros(len(X_val), dtype=np.float32)
    val_lgb_et = np.zeros(len(X_val), dtype=np.float32)

    feature_gains = np.zeros(X_train.shape[1], dtype=np.float64)

    for fold, (tr_idx, oof_idx) in enumerate(gkf.split(X_train, y_train, groups=g_train), 1):
        print(f'\n--- Training Fold {fold}/{N_FOLDS} ---')
        X_tr_f, y_tr_f = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va_f, y_va_f = X_train.iloc[oof_idx], y_train.iloc[oof_idx]

        # 1. XGBoost Depth-8 (Balanced Regularized Trees)
        m_xgb8 = XGBClassifier(
            n_estimators=2000,
            learning_rate=0.025,
            max_depth=8,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.70,
            reg_alpha=1.5,
            reg_lambda=3.0,
            scale_pos_weight=sqrt_weight,
            eval_metric='aucpr',
            early_stopping_rounds=75,
            enable_categorical=True,
            tree_method='hist',
            device='cuda',
            random_state=100 + fold,
        )
        m_xgb8.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
        oof_xgb8[oof_idx] = m_xgb8.predict_proba(X_va_f)[:, 1]
        val_xgb8 += m_xgb8.predict_proba(X_val)[:, 1] / N_FOLDS
        feature_gains += m_xgb8.feature_importances_ / N_FOLDS

        # 2. XGBoost Depth-10 (Deep Multi-Condition Isolation)
        m_xgb10 = XGBClassifier(
            n_estimators=2000,
            learning_rate=0.02,
            max_depth=10,
            min_child_weight=15,
            subsample=0.75,
            colsample_bytree=0.60,
            reg_alpha=2.0,
            reg_lambda=5.0,
            scale_pos_weight=sqrt_weight,
            eval_metric='aucpr',
            early_stopping_rounds=75,
            enable_categorical=True,
            tree_method='hist',
            device='cuda',
            random_state=200 + fold,
        )
        m_xgb10.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
        oof_xgb10[oof_idx] = m_xgb10.predict_proba(X_va_f)[:, 1]
        val_xgb10 += m_xgb10.predict_proba(X_val)[:, 1] / N_FOLDS

        # 3. LightGBM Leaf-Wise (num_leaves=127)
        m_lgb = lgb.LGBMClassifier(
            n_estimators=2000,
            learning_rate=0.025,
            num_leaves=127,
            max_depth=9,
            min_child_samples=45,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.70,
            reg_alpha=1.5,
            reg_lambda=3.0,
            scale_pos_weight=sqrt_weight,
            random_state=300 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        m_lgb.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            eval_metric='average_precision',
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        oof_lgb[oof_idx] = m_lgb.predict_proba(X_va_f)[:, 1]
        val_lgb += m_lgb.predict_proba(X_val)[:, 1] / N_FOLDS

        # 4. LightGBM Extra-Trees (Randomized Split Thresholds for Zero-Shot Variance Reduction)
        m_lgb_et = lgb.LGBMClassifier(
            n_estimators=2200,
            learning_rate=0.025,
            num_leaves=127,
            max_depth=9,
            min_child_samples=40,
            extra_trees=True,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.65,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=sqrt_weight,
            random_state=400 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        m_lgb_et.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            eval_metric='average_precision',
            callbacks=[lgb.early_stopping(75, verbose=False)],
        )
        oof_lgb_et[oof_idx] = m_lgb_et.predict_proba(X_va_f)[:, 1]
        val_lgb_et += m_lgb_et.predict_proba(X_val)[:, 1] / N_FOLDS

    # Print Top 15 Most Important Features
    imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_gains}).sort_values(
        by='Importance', ascending=False
    )
    print('\n🔍 Top 15 V3 Features by XGBoost Gain:')
    print(imp_df.head(15).to_string(index=False))

    # Optimize Logit-Space Ensemble Weights Strictly on OOF
    oof_models = [oof_xgb8, oof_xgb10, oof_lgb, oof_lgb_et]
    val_models = [val_xgb8, val_xgb10, val_lgb, val_lgb_et]
    best_w, oof_pr = optimize_logit_weights(oof_models, y_train)

    print(f'\nLearned OOF Logit Weights (XGB8, XGB10, LGB, LGB_ET): {np.round(best_w, 4)} (OOF PR-AUC: {oof_pr:.5f})')

    oof_logit_blend = expit(np.column_stack([to_logits(p) for p in oof_models]) @ best_w)
    oof_best_th, oof_best_f1 = find_best_f1_threshold(y_train, oof_logit_blend)
    print(f'Learned OOF Optimal Threshold: {oof_best_th:.4f} (OOF F1: {oof_best_f1:.5f})\n')

    # Evaluate on Unseen 20% Validation Buildings
    results = []
    results.append(evaluate_predictions(y_val, val_xgb8, '1. 5-Fold XGBoost Depth-8 (V3)'))
    results.append(evaluate_predictions(y_val, val_xgb10, '2. 5-Fold XGBoost Depth-10 (V3)'))
    results.append(evaluate_predictions(y_val, val_lgb, '3. 5-Fold LightGBM Standard (V3)'))
    results.append(evaluate_predictions(y_val, val_lgb_et, '4. 5-Fold LightGBM ExtraTrees (V3)'))

    val_logit_blend = expit(np.column_stack([to_logits(p) for p in val_models]) @ best_w)
    results.append(evaluate_predictions(
        y_val, val_logit_blend, '5. ENSEMBLE: OOF Logit-Space 4-Model Blend', fixed_thresh=oof_best_th
    ))

    # Also test Probability-Space Blend with the same weights
    val_prob_blend = np.column_stack(val_models) @ best_w
    results.append(evaluate_predictions(
        y_val, val_prob_blend, '6. ENSEMBLE: Probability-Space 4-Model Blend'
    ))

    leaderboard = pd.DataFrame(results).sort_values(by='PR-AUC', ascending=False)
    print('\n' + '=' * 105)
    print('🏆 V3 ZERO-SHOT UNSEEN BUILDING LEADERBOARD')
    print('=' * 105)
    print(leaderboard.to_string(index=False))
    print('=' * 105 + '\n')


if __name__ == '__main__':
    X_train, y_train, g_train, X_val, y_val, _, sqrt_weight = build_v3_features(DATA_PATH)
    run_v3_pipeline(X_train, y_train, g_train, X_val, y_val, sqrt_weight)