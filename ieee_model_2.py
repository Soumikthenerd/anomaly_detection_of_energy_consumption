import gc
import warnings
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn import metrics
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
import torch
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. Configuration
# ---------------------------------------------------------
DATA_PATH = '/home/soumik-mondal/Downloads/LEAD/clean_reduced_lead_dataset.csv'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
N_FOLDS = 5

print(f'Active Device: {DEVICE} | CatBoost Available: {HAS_CATBOOST}')
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
# 2. Bidirectional Time-Series & Stagnation Features
# ---------------------------------------------------------
def build_v2_features(file_path: str):
    print(f'\nLoading {file_path}...')
    df = pd.read_csv(file_path)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['building_id', 'timestamp']).reset_index(drop=True)

    print('Engineering bidirectional lags, rolling envelopes & flatline detectors...')
    grp = df.groupby('building_id')['meter_reading']

    # Physical normalization
    df['meter_per_sqft'] = df['meter_reading'] / (df['square_feet'] + 1.0)
    df['temp_power_interaction'] = df['meter_per_sqft'] * (df['air_temperature'] + 30.0)

    # Multi-scale rolling statistics (6h, 24h, 168h)
    df['rolling_mean_24h'] = grp.transform(lambda s: s.rolling(24, min_periods=1).mean())
    df['rolling_std_24h'] = grp.transform(lambda s: s.rolling(24, min_periods=1).std()).fillna(0.0)
    df['rolling_std_6h'] = grp.transform(lambda s: s.rolling(6, min_periods=1).std()).fillna(0.0)

    rm_168 = grp.transform(lambda s: s.rolling(168, min_periods=24).mean())
    df['rolling_mean_168h'] = rm_168.fillna(df['rolling_mean_24h'])

    # Centered 24h rolling mean (uses past + future context within the same building)
    rm_center_24 = grp.transform(lambda s: s.rolling(25, min_periods=1, center=True).mean())
    df['diff_from_centered_24h'] = df['meter_reading'] - rm_center_24

    # Building-level robust baseline (median & IQR)
    b_median = grp.transform('median')
    b_q75 = grp.transform(lambda s: s.quantile(0.75))
    b_q25 = grp.transform(lambda s: s.quantile(0.25))
    df['robust_z_bldg'] = ((df['meter_reading'] - b_median) / ((b_q75 - b_q25) + 1e-3)).clip(-15.0, 15.0)

    df['z_score_24h'] = (
        (df['meter_reading'] - df['rolling_mean_24h']) / (df['rolling_std_24h'] + 1e-5)
    ).clip(-10.0, 10.0)
    df['ratio_to_168h_mean'] = (
        df['meter_reading'] / (df['rolling_mean_168h'] + 1e-3)
    ).clip(0.0, 10.0)

    # Backward lag differences (onset detection)
    df['lag_diff_1h'] = grp.diff(1).fillna(0.0)
    df['lag_diff_24h'] = grp.diff(24).fillna(0.0)
    df['lag_diff_168h'] = grp.diff(168).fillna(0.0)

    # Forward lead differences (recovery / spike-exit detection)
    df['lead_diff_1h'] = grp.diff(-1).fillna(0.0)
    df['lead_diff_24h'] = grp.diff(-24).fillna(0.0)
    df['lead_diff_168h'] = grp.diff(-168).fillna(0.0)

    # Exact sensor stagnation / flatline run-length counter per building
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

    # Exact same 80/20 zero-shot split as previous benchmark
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, val_idx = next(gss.split(X, y, groups=groups))

    X_train, y_train, g_train = X.iloc[train_idx].copy(), y.iloc[train_idx].copy(), groups[train_idx]
    X_val, y_val, g_val = X.iloc[val_idx].copy(), y.iloc[val_idx].copy(), groups[val_idx]

    raw_ratio = float((y_train == 0).sum() / max(1, (y_train == 1).sum()))
    sqrt_weight = float(np.sqrt(raw_ratio))

    print(f'Train Rows: {len(X_train):,} | Val Rows: {len(X_val):,}')
    print(f'Total V2 Features: {len(feature_cols)} | Calibrated sqrt(weight): {sqrt_weight:.2f}')

    return X_train, y_train, g_train, X_val, y_val, g_val, sqrt_weight


# ---------------------------------------------------------
# 3. Evaluation & Temporal Post-Processing Helpers
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


def temporal_smooth_by_building(probs: np.ndarray, building_ids: np.ndarray) -> np.ndarray:
    """Applies a 3-hour centered smoothing filter strictly within each building."""
    s = pd.Series(probs)
    g = pd.Series(building_ids)
    prev_p = s.groupby(g).shift(1).fillna(s)
    next_p = s.groupby(g).shift(-1).fillna(s)
    return (0.15 * prev_p + 0.70 * s + 0.15 * next_p).values


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

    print('=' * 84)
    print(f'📊 {model_name}')
    print('=' * 84)
    print(f'  • PR-AUC (Avg Prec): {pr_auc:.5f}  |  ROC-AUC:           {roc_auc:.5f}')
    print(f'  • Peak F1-Score:     {best_f1:.5f}  |  Peak Threshold:    {best_th:.4f}')
    if fixed_thresh is not None:
        print(f'  • OOF-Thresh F1:     {f1_at_th:.5f}  |  OOF Threshold:     {fixed_thresh:.4f}')
    print(f'  • Precision:         {prec:.4f}   |  Recall:            {rec:.4f}')
    print(f'  • Specificity:       {spec:.4f}   |  Balanced Acc:      {bal_acc:.4f}  |  MCC: {mcc:.4f}')
    print(f'  • Confusion Matrix:  TP={tp} | FP={fp} | FN={fn} | TN={tn}')
    print('=' * 84 + '\n')

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
# 4. 5-Fold GroupKFold Training & OOF Ensemble Optimization
# ---------------------------------------------------------
def run_v2_pipeline(X_train, y_train, g_train, X_val, y_val, g_val, sqrt_weight: float):
    gkf = GroupKFold(n_splits=N_FOLDS)

    oof_xgb = np.zeros(len(X_train), dtype=np.float32)
    oof_lgb = np.zeros(len(X_train), dtype=np.float32)
    oof_cb = np.zeros(len(X_train), dtype=np.float32)

    val_xgb = np.zeros(len(X_val), dtype=np.float32)
    val_lgb = np.zeros(len(X_val), dtype=np.float32)
    val_cb = np.zeros(len(X_val), dtype=np.float32)

    if HAS_CATBOOST:
        X_tr_cb = X_train.copy()
        X_va_cb = X_val.copy()
        cat_cols = X_tr_cb.select_dtypes(include=['category']).columns.tolist()
        for c in cat_cols:
            X_tr_cb[c] = X_tr_cb[c].astype(str)
            X_va_cb[c] = X_va_cb[c].astype(str)

    for fold, (tr_idx, oof_idx) in enumerate(gkf.split(X_train, y_train, groups=g_train), 1):
        print(f'\n--- Training Fold {fold}/{N_FOLDS} ---')
        X_tr_f, y_tr_f = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
        X_va_f, y_va_f = X_train.iloc[oof_idx], y_train.iloc[oof_idx]

        # 1. XGBoost (CUDA + Early Stopping on PR-AUC)
        xgb_model = XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=8,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=sqrt_weight,
            eval_metric='aucpr',
            early_stopping_rounds=60,
            enable_categorical=True,
            tree_method='hist',
            device='cuda',
            random_state=42 + fold,
        )
        xgb_model.fit(X_tr_f, y_tr_f, eval_set=[(X_va_f, y_va_f)], verbose=False)
        oof_xgb[oof_idx] = xgb_model.predict_proba(X_va_f)[:, 1]
        val_xgb += xgb_model.predict_proba(X_val)[:, 1] / N_FOLDS

        # 2. LightGBM (Leaf-wise + Early Stopping on average_precision)
        lgb_model = lgb.LGBMClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            num_leaves=95,
            max_depth=8,
            min_child_samples=40,
            subsample=0.8,
            subsample_freq=1,
            colsample_bytree=0.75,
            reg_alpha=1.0,
            reg_lambda=2.0,
            scale_pos_weight=sqrt_weight,
            random_state=42 + fold,
            n_jobs=-1,
            verbose=-1,
        )
        lgb_model.fit(
            X_tr_f, y_tr_f,
            eval_set=[(X_va_f, y_va_f)],
            eval_metric='average_precision',
            callbacks=[lgb.early_stopping(60, verbose=False)],
        )
        oof_lgb[oof_idx] = lgb_model.predict_proba(X_va_f)[:, 1]
        val_lgb += lgb_model.predict_proba(X_val)[:, 1] / N_FOLDS

        # 3. CatBoost (CUDA + Early Stopping)
        if HAS_CATBOOST:
            cb_model = CatBoostClassifier(
                iterations=1800,
                learning_rate=0.045,
                depth=7,
                l2_leaf_reg=5.0,
                scale_pos_weight=sqrt_weight,
                eval_metric='PRAUC',
                early_stopping_rounds=80,
                cat_features=cat_cols,
                task_type='GPU',
                random_seed=42 + fold,
                verbose=0,
            )
            cb_model.fit(X_tr_cb.iloc[tr_idx], y_tr_f, eval_set=(X_tr_cb.iloc[oof_idx], y_va_f))
            oof_cb[oof_idx] = cb_model.predict_proba(X_tr_cb.iloc[oof_idx])[:, 1]
            val_cb += cb_model.predict_proba(X_va_cb)[:, 1] / N_FOLDS

    # Learn optimal ensemble weights strictly on OOF predictions (zero holdout leakage)
    if HAS_CATBOOST:
        oof_matrix = np.column_stack([oof_xgb, oof_lgb, oof_cb])
        val_matrix = np.column_stack([val_xgb, val_lgb, val_cb])
        init_w = [0.45, 0.45, 0.10]
    else:
        oof_matrix = np.column_stack([oof_xgb, oof_lgb])
        val_matrix = np.column_stack([val_xgb, val_lgb])
        init_w = [0.50, 0.50]

    # Fast subsampled OOF optimization for PR-AUC
    rng = np.random.default_rng(42)
    sub_idx = rng.choice(len(y_train), size=min(300000, len(y_train)), replace=False)
    y_sub = y_train.values[sub_idx]
    oof_sub = oof_matrix[sub_idx]

    def loss_func(weights):
        w = np.array(weights)
        w = w / np.sum(w)
        blend = oof_sub @ w
        return -metrics.average_precision_score(y_sub, blend)

    res = minimize(loss_func, init_w, bounds=[(0, 1)] * len(init_w), method='Nelder-Mead')
    best_w = np.maximum(0, res.x)
    best_w = best_w / np.sum(best_w)
    print(f'\nLearned OOF Optimal Blend Weights (XGB, LGB{", CB" if HAS_CATBOOST else ""}): {np.round(best_w, 4)}')

    oof_blend = oof_matrix @ best_w
    oof_best_th, oof_best_f1 = find_best_f1_threshold(y_train, oof_blend)
    print(f'Learned OOF Optimal Threshold: {oof_best_th:.4f} (OOF F1: {oof_best_f1:.5f})\n')

    # Evaluate on Unseen 20% Validation Buildings
    results = []
    results.append(evaluate_predictions(y_val, val_xgb, '1. 5-Fold XGBoost (CUDA + V2 Features)'))
    results.append(evaluate_predictions(y_val, val_lgb, '2. 5-Fold LightGBM (Leaf-Wise + V2 Features)'))
    if HAS_CATBOOST:
        results.append(evaluate_predictions(y_val, val_cb, '3. 5-Fold CatBoost (CUDA + V2 Features)'))

    val_blend = val_matrix @ best_w
    results.append(evaluate_predictions(
        y_val, val_blend, '4. ENSEMBLE: OOF-Weighted GBDT Blend', fixed_thresh=oof_best_th
    ))

    # Post-processing: 3-hour temporal smoothing per building
    val_blend_smooth = temporal_smooth_by_building(val_blend, g_val)
    oof_blend_smooth = temporal_smooth_by_building(oof_blend, g_train)
    smooth_th, _ = find_best_f1_threshold(y_train, oof_blend_smooth)

    results.append(evaluate_predictions(
        y_val, val_blend_smooth, '5. POST-PROCESSED: Temporally Smoothed Blend', fixed_thresh=smooth_th
    ))

    leaderboard = pd.DataFrame(results).sort_values(by='PR-AUC', ascending=False)
    print('\n' + '=' * 105)
    print('🏆 V2 ZERO-SHOT UNSEEN BUILDING LEADERBOARD')
    print('=' * 105)
    print(leaderboard.to_string(index=False))
    print('=' * 105 + '\n')


if __name__ == '__main__':
    X_train, y_train, g_train, X_val, y_val, g_val, sqrt_weight = build_v2_features(DATA_PATH)
    run_v2_pipeline(X_train, y_train, g_train, X_val, y_val, g_val, sqrt_weight)