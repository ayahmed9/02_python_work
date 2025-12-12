import os
import numpy as np
import pandas as pd
import joblib

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, accuracy_score, f1_score
from math import sqrt
from collections import defaultdict

# -----------------------------
# Options and flags from Stata
# -----------------------------
opt = _st.options
verbose_flag = "Verbose" in opt
overwrite_flag = "Overwrite" in opt
uncertainty_flag = "Uncertainty" in opt

n_estimators_impute = int(opt.get('NImp', 10))
n_estimators_model = int(opt.get('NModel', 500))
max_iter_impute = int(opt.get('MaxIter', 10))
random_state = int(opt.get('RandomState', 0))
n_jobs = int(opt.get('NJobs', 1))
top_n = int(opt.get('TopFeatures', 10))
max_depth = int(opt.get('MaxDepth', 0))
min_samples_leaf = int(opt.get('MinSamplesLeaf', 1))

rf_max_depth = None if max_depth <= 0 else max_depth

save_path = opt['Saveas']
new_data_path = opt.get('NewData', "")

# -----------------------------
# Safety checks and overwrite control
# -----------------------------
def fail(msg):
    _st.console_write("ERROR: " + msg + "\n")
    raise RuntimeError(msg)

# Saveas existence
if os.path.exists(save_path) and not overwrite_flag:
    fail(f"Output file already exists: {save_path}. Use Overwrite option to replace.")

# NewData existence and extensiohn, if provided
if new_data_path:
    if not os.path.exists(new_data_path):
        fail(f"NewData file not found: {new_data_path}")
    if not new_data_path.lower().endswith(".dta"):
        fail("NewData must be a Stata .dta file")

# -----------------------------
# Load training data from Stata
# -----------------------------
df_train = pd.DataFrame({col: list(_st[col]) for col in _st.list_vars()})

if df_train.empty:
    fail("Training dataset is empty.")

# -----------------------------
# Determine target variables
# -----------------------------
if "Target" in opt and opt['Target'].strip() != "":
    target_vars = opt['Target'].split()  # user-specified targets
else:
    # Auto-detect: all columns with missing values
    target_vars = df_train.columns[df_train.isnull().any()].tolist()
    _st.console_write(f"Target not specified. Auto-detected variables with missing values: {target_vars}\n")

# Skip target varibales with no missing values (if any got included)
target_vars = [t for t in target_vars if df_train[t].isnull().any()]

if len(target_vars) == 0:
    _st.console_write("No target variables with missing values detected. Proceeding with joint imputation of the dataset and training RF models for all variables is disabled.\n")
        
# -----------------------------
# Detect categorical columns
# -----------------------------
cat_cols = df_train.select_dtypes(iniclude=['object', 'category']).columns.tolist()

# Build encoders with robust handling of unknown categories
encoders = {}
if len(cat_cols) > 0:
    enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    df_train[cat_cols] = enc.fit_transform(df_train[cat_cols])
    for c in cat_cols:
        encoders[c] = enc  # one shared encoder object for all categorical columns

# -----------------------------
# MissForest-style joint imputation
# -----------------------------        
# We impute the entire dataset jointly. This is closer to MissForest: RF-based iterative imputation on encoded data.
imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=n_estimators_impute,
        random_state=random_state,
        n_jobs=n_jobs,
        max_depth=rf_max_depth,
        min_samples_leaf=min_samples_leaf
    ),       
    max_iter=max_iter_impute,
    random_state=random_state,
    verbose=verbose_flag
)

df_imputed_num = pd.DataFrame(imputer.fit_transform(df_train), columns=df_train.columns)

# Decode categoricals back to original labels
if len(cat_cols) > 0:
    # Round and cast categorical codes before inverse_transform
    df_imputed_num[cat_cols] = np.rint(df_imputed_num[cat_cols]).astype(int)
    df_decoded = df_imputed_num.copy()
    # OrdinalEncoder with a shared instance can inverse_transform on the block
    try:
        df_decoded[cat_cols] = enc.inverse_transform(df_imputed_num[cat_cols])
    except Exception as e:
        _st.consoloe_write(f"WARNING: inverse_transform failed for categoricals; leaving integer codes. Details: {e}\n")
        df_decoded = df_imputed_num
else:
    df_decoded = df_imputed_num

df_imputed = df_decoded

# -----------------------------
# Save imputed training dataset
# -----------------------------
df_imputed.to_stata(save_path, write_index=False)
_st.console_write(f"Imputed training dataset saved: {save_path}\n")

# -----------------------------
# Determine modeling targets and types
# -----------------------------
# If user gave targe_vars, use those; otherwise, consider auto-detected targets (already set).
# For classification vs regression: if column is categorical/object originally -> classification; else regression.
# We need the original dtype info: re-check on the original df (not encoded/imputed).
original_dtypes = pd.DataFrame({c: df_train[c].dtype for c in df_train.columns}, index=['dtype']).T

def target_type(col):
    # If originally object/category: treat as classification
    # Else numeric: regression
    if col in cat_cols:
        return "class"
    else:
        return "regr"

# If target_vars is empty (no missing values), you can choose to skip modeling or allow user-provided targets without missing variables
# Here we will allow training models for user-specified targets even if they had no missing values:
if "Target" in opt and opt['Target'].strip() != "":
    target_vars = opt['Target'].split()

if len(target_vars) == 0:
    _st.console_write("No target variables specified; skipping model training and feature importances.\n")
    rf_models = {}
else:
    # -----------------------------
    # Train RF models per target
    # -----------------------------
    rf_models = {}
    metrics_report = []
    feature_importances = []

    for target in target_vars:
        if target not in df_imputed.columns:
            _st.console_write(f"WARNING: target '{target}' not found in dataset; skipping.\n")
            continue

        X = df_imputed.drop(columns=[target])
        y = df_imputed[target]

        # Choose model type based on original dtype
        ttype = target_type(target)
        if ttype == "class":
            # Ensure y is categorical labels if inverse_tranform worked; otherwise, cast to int codes
            if y.dtype.kind in ['O', 'U']:
                y_labels = y
            else:
                # If numeric codes remain, keep as ints; sklearn will handle them as class labels
                y_labels = y.astype(int)
            
            model = RandomForestClassifier(
                n_estimators=n_estimators_model,
                random_state=random_state,
                n_jobs=n_jobs,
                max_depth=rf_max_depth,
                min_samples_leaf=min_samples_leaf
            )
            model.fit(X, y_labels)
            y_pred = model.predict(X)

            # Metrics: accuracy, F1 (macro)
            acc = accuracy_score(y_labels, y_pred)
            f1m = f1_score(y_labels, y_pred, average='macro')
            metrics_report.append({
                'Target': target, 'Type': 'classification',
                'Accuracy': acc, 'F1_macro': f1m
            })

            # Feature importances
            importances = model.feature_importances_
            features = X.columns
            # Top-N filter
            order = np.argsort(importances)[::-1]
            cutoff = len(features) if top_n <= 0 else min(top_n, len(features))
            for idx in order[:cutoff]:
                feature_importances.append({'Target': target, 'Feature': features[idx], 'Importances': float(importances[idx])})
        
        else:
            # Regression
            model = RandomForestRegressor(
                n_estimators=n_estimators_model,
                random_state=random_state,
                n_jobs=n_jobs,
                max_depth=rf_max_depth,
                min_samples_leaf=min_samples_leaf
            )
            model.fit(X, y)
            y_pred = model.predict(X)

            # Metrics: R2, RMSE, MAE
            r2 = r2_score(y, y_pred)
            rmse = sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            metrics_report.append({
                'Target': target, 'Type': 'regression',
                'R2': r2, 'RMSE': rmse, 'MAE': mae
            })

             # Feature importances
            importances = model.feature_importances_
            features = X.columns
            # Top-N filter
            order = np.argsort(importances)[::-1]
            cutoff = len(features) if top_n <= 0 else min(top_n, len(features))
            for idx in order[:cutoff]:
                feature_importances.append({'Target': target, 'Feature': features[idx], 'Importances': float(importances[idx])})
        
        rf_models[target] = model
    
    # Print metrics
    if len(metrics_report) > 0:
        _st.console_write("\nModel evaluation metrics (training data):\n")
        for m in metrics_report:
            if m['Type'] == 'regression':
                _st.console_write(f"Target: {m['Target']} | R2: {m['R2']:.4f} | RMSE: {m['RMSE']:.4f} | MAE: {m['MAE']:.4f}\n")
            else:
                _st.console_write(f"Target: {m['Target']} | Accuracy: {m['Accuracy']:.4f} | F1_macro: {m['F1_macro']}:.4f\n")
    
    # Print feature importances (Top-N)
    if len(feature_importances) > 0:
        _st.console_write("\nTop feature importances:\n")
        # Group by target to print groupe output
        by_target = defaultdict(list)
        for row in feature_importances:
            by_target[row['Target']].append(row)
        for tgt, rows in by_target.items():
            _st.console_write(f"Target: {tgt}\n")
            for r in rows:
                _st.console_write(f"  Feature: {r['Feature']} | Importance: {r['Importance']}:.4f\n")

# -----------------------------
# Save models, encoders, imputer as a bundle
# -----------------------------
bundle = {
    'models': rf_models,
    'encoders': encoders if len(encoders) > 0 else None,
    'imputer': imputer,
    'metadata': {
        'cat_cols': cat_cols,
        'random_state': random_state,
        'n_estimators_impute': n_estimators_impute,
        'n_estimators_model': n_estimators_model,
        'max_iter_impute': max_iter_impute,
        'n_jobs': n_jobs,
        'top_n': top_n,
        'max_depth': max_depth,
        'min_samples_leaf': min_samples_leaf,
        'target_vars': target_vars
    }
}
bundle_path = save_path.replace(".dta", "_bundle.joblib")
if os.path.exists(bundle_path) and not overwrite_flag:
    fail(f"Bundle already exists: {bundle_path}. Use Overwrite to replace.")
joblib.dump(bundle, bundle_path)
_st.console_write(f"Model/encoder/imputer bundle saved: {bundle_path}\n")

# -----------------------------
# Predict on NewData (optional)
# -----------------------------
if new_data_path:
    df_new = pd.read_stata(new_data_path)
    if df_new.empty:
        fail("NewData is empty.")
    
    # Validate columns: ensure NewData has at least the feature columns
    missing_cols = [c for c in df_imputed.columns if c not in df_new.columns]
    if len(missing_cols) > 0:
        _st.console_write(f"WARNING: NewData missing columns present in training: {missing_cols}\n")
        # proceed but model will drop targets appropriately
    
    # Apply encoders on categoricals present in new data
    # Use same cat_cols from training; if some are missing in new data, skip
    cat_in_new = [c for c in cat_cols if c in df_new.columns]
    if len(cat_in_new) > 0 and bundle['encoders'] is not None:
        try:
            df_new[cat_in_new] = enc.transform(df_new[cat_in_new])
        except Exception as e:
            _st.console_write(f"WARNING: categorical encoding on NewData had issues; unknowns will be set to -1. Details: {e}\n")
            # As OrdinalEncoder is already configured to handle_unknown, we attempted transform; issues logged.
    
    # Impute NewData jointly using the training-fitted imputer
    try:
        # Align columns with training columns for imputer
        # If NewData is missing some columns, we add them as NaN to allow imputer to work
        for col in df_imputed.columns:
            if col not in df_new.columns:
                df_new[col] = np.nan
        
        # Ensure column order matches training for transform, then restore original order
        df_new = df_new[df_imputed.columns]
        df_new_imputed_num = pd.DataFrame(imputer.transform(df_new), columns=df_imputed.columns)
    except Exception as e:
        fail(f"Imputation of NewData failed during transform: {e}\n")

    # Decode categoricals back
    if len(cat_in_new) > 0 and bundle['encoders'] is not None:
        try:
            df_new_imputed_num[cat_in_new] = np.rint(df_new_imputed_num[cat_in_new]).astype(int)
            df_new_decoded = df_new_imputed_num.copy()
            df_new_decoded[cat_in_new] = enc.inverse_transform(df_new_imputed_num[cat_in_new])
        except Exception as e:
            _st.console_write(f"WARNING: inverse_transform on NewData categoricals failed; leaving integer codes. Details: {e}\n")
            df_new_decoded = df_new_imputed_num
        else:
            df_new_decoded = df_new_imputed_num
        
        df_new_final = df_new_decoded

        # Predict with each model and optionally compute uncertainty
        if len(rf_models) == 0:
            _st.console_write("No trained models to apply on NewData (no targets). Skipping prediction.\n")
        else:
            preds_dict = {}
            uncert_dict = {}

            for target, model in rf_models.items():
                if target not in df_new_final.columns:
                    _st.console_write(f"WARNING: Target {'target'} not in NewData; prediction will still be added based on available features.\n")
                
                X_new = df_new_final.drop(columns=[target]) if target in df_new_final.columns else df_new_final.copy()

                # Prediction
                if isinstance(model, RandomForestClassifier):
                    preds = model.predict(X_new)
                    preds_dict[target] = preds

                    if uncertainty_flag:
                        # Use predictive entropy from class probabilities as uncertainty
                        proba = model.predict(X_new)
                        # entropy: -sum p*log(p)
                        entropy = np.array([-np.sum(p * np.log(np.clip(p, 1e-12, 1.0))) for p in proba])
                        uncert_dict[target] = entropy
                else:
                    # Regression
                    preds = model.predict(X_new)
                    preds_dict[target] = preds

                    if uncertainty_flag:
                        # Per-tree predictions std as uncertainty
                        # Note: estimators_ gives list of trees: average already used by model.predict
                        tree_preds = np.array([est.predict(X_new) for est in model.estimators_])
                        std = tree_preds.std(axis=0)
                        uncert_dict[target] = std
                
                # Add columns to NewData frame
                pred_var = "predicted_" + target
                df_new_final[pred_var] = preds_dict[target]
                if uncertainty_flag and target in uncert_dict:
                    df_new_final[pred_var + "_uncert"] = uncert_dict[target]
            
            # Save predicted dataset
            new_save_path = new_data_path.replace(".dta", "_predicted.dta")
            if os.path.exists(new_save_path) and not overwrite_flag:
                fail(f"Predicted output already exists: {new_save_path}. Use Overwrite to replace.")
            df_new_final.to_stata(new_save_path, write_index=False)
            _st.console_write(f"\nPredictions added to NewData and saved: {new_save_path}\n")

            # Print preview of predictions (first 10 rows) for readability
            _st.console_write("\nPrediction preview (first 10 rows):\n")
            preview_cols = []
            for target in rf_models.keys():
                pv = "predicted_" + target
                preview_cols.append(pv)
                if uncertainty_flag:
                    preview_cols.append(pv + "_uncert")
            preview = df_new_final[preview_cols].head(10).round(4)
            for i, row in preview.iterrows():
                row_str = ", ".join([f"{col}: {row[col]}" for col in preview_cols])
                _st.console_write(f"Row {i+1}: {row_str}\n")

# ----------------------------------
# Final Message
# ----------------------------------
_st.console_write("\nPipeline complete.\n")

