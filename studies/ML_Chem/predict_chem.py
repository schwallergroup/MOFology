import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pymatgen.core import Composition
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, GroupShuffleSplit
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold
import joblib
import os
import json

# Set paths - use environment variable or default to local
import os as _os
_BASE = _os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = f"{_BASE}/studies/data/chemcial_properties.csv"
OUTPUT_DIR = f"{_BASE}/results/ML_Chem/prediction_results"
MODEL_DIR = f"{_BASE}/results/ML_Chem/models"
RESULTS_DIR = f"{_BASE}/results/ML_Chem/prediction_results"

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Handle features

def get_mof_family(uri):
    """Extract base MOF identifier, grouping parent and functionalized variants.

    This ensures that functionalized MOFs (e.g., FuncMOF_HKUST1_deen) are grouped
    with their parent MOFs (e.g., MOF_HKUST1) to prevent data leakage in train/test splits.
    """
    if pd.isna(uri):
        return "unknown"
    # Extract the fragment after the hash
    frag = uri.split("#")[-1] if "#" in uri else str(uri)

    if frag.startswith("FuncMOF_"):
        # FuncMOF_HKUST1_deen -> HKUST1
        parts = frag.replace("FuncMOF_", "").split("_")
        return parts[0] if parts else frag
    elif frag.startswith("MOF_STAB_"):
        # MOF_STAB_xyz -> STAB_xyz (hypothetical family)
        return frag.replace("MOF_", "")
    elif frag.startswith("MOF_qmof-"):
        # MOF_qmof-abc -> qmof-abc
        return frag.replace("MOF_", "")
    elif frag.startswith("MOF_"):
        # MOF_HKUST1 -> HKUST1
        return frag.replace("MOF_", "").split("_")[0]
    return frag


def get_morgan_fingerprint(smiles, n_bits=1024):
    """Generates a Morgan fingerprint for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=n_bits)
            return np.array(fp)
    except:
        pass
    return np.zeros(n_bits)

def process_smiles_field(smiles_field, n_bits=1024):
    """Processes a semicolon-separated list of SMILES and returns the average fingerprint."""
    if pd.isna(smiles_field) or not smiles_field:
        return np.zeros(n_bits)
    
    smiles_list = [s.strip() for s in smiles_field.split(';')]
    fps = [get_morgan_fingerprint(s, n_bits) for s in smiles_list]
    valid_fps = [fp for fp in fps if np.any(fp)]
    
    if not valid_fps:
        return np.zeros(n_bits)
    return np.mean(valid_fps, axis=0)

def get_composition_features(formula):
    """Parses formula and returns fractions of common elements."""
    elements = ['C', 'H', 'N', 'O', 'S', 'Zn', 'Cu', 'Fe', 'Al', 'Zr', 'Co', 'Ni']
    features = {el: 0.0 for el in elements}
    try:
        comp = Composition(formula)
        total_atoms = sum(comp.values())
        for el in elements:
            features[el] = comp.get_el_amt_dict().get(el, 0.0) / total_atoms
    except:
        pass
    return features

def extract_mofid_smiles(mofid):
    """Extracts the SMILES-like part from a MOFid string."""
    if pd.isna(mofid) or not mofid:
        return ""
    # Usually MOFid format is "SMILES1.SMILES2 MOFid-v1.topology.cat"
    return mofid.split(' ')[0]


def detect_high_coverage_properties(df, min_samples=500):
    """
    Automatically detect properties with at least min_samples non-null values.
    Returns list of property names that meet the threshold.
    """
    total_samples = len(df)
    high_coverage_props = []

    # Get all numeric columns (excluding metadata columns)
    metadata_cols = ['mof_uri', 'csd_code', 'chemical_formula', 'mofid',
                     'topology', 'metal_cluster_elements', 'linker_smiles',
                     'space_group', 'crystal_system', 'Number of atoms',
                     'Unit cell volume', 'Space group number']

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    property_cols = [col for col in numeric_cols if col not in metadata_cols]

    print(f"\nAnalyzing coverage for {len(property_cols)} properties...")
    coverage_info = []

    for prop in property_cols:
        non_null_count = df[prop].notna().sum()
        coverage = non_null_count / total_samples if total_samples > 0 else 0

        coverage_info.append({
            'property': prop,
            'coverage': coverage,
            'non_null_count': non_null_count,
            'total_samples': total_samples
        })

        if non_null_count >= min_samples:
            high_coverage_props.append(prop)

    # Save coverage report
    coverage_df = pd.DataFrame(coverage_info)
    coverage_df = coverage_df.sort_values('coverage', ascending=False)
    coverage_path = os.path.join(OUTPUT_DIR, 'property_coverage_report.csv')
    coverage_df.to_csv(coverage_path, index=False)
    print(f"Coverage report saved to {coverage_path}")

    print(f"\nFound {len(high_coverage_props)} properties with >= {min_samples} samples:")
    for prop in high_coverage_props:
        count = coverage_df[coverage_df['property'] == prop]['non_null_count'].values[0]
        print(f"  - {prop}: {int(count)} samples")

    return high_coverage_props


def perform_grid_search(model, param_grid, X_train, y_train, cv=5, scoring='r2'):
    """Perform grid search for hyperparameter optimization."""
    grid_search = GridSearchCV(
        model, 
        param_grid, 
        cv=cv, 
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


# 2.Processing and Modeling

def main():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} samples with {len(df.columns)} columns")
    
    # Automatically detect properties with >= 500 samples
    targets = detect_high_coverage_properties(df, min_samples=500)
    
    if len(targets) == 0:
        print("No properties found with needed coverage. Exiting.")
        return
    
    # Input structural/chemical features to process
    print("\nEngineering features...")
    
    # Process SMILES (Linker and MOFid component)
    linker_fps = np.stack(df['linker_smiles'].apply(lambda x: process_smiles_field(x)).values)
    mofid_smiles = df['mofid'].apply(extract_mofid_smiles)
    mofid_fps = np.stack(mofid_smiles.apply(lambda x: process_smiles_field(x)).values)
    
    # Combine fingerprints
    fp_features = np.hstack([linker_fps, mofid_fps])
    fp_cols = [f'fp_{i}' for i in range(fp_features.shape[1])]
    df_fp = pd.DataFrame(fp_features, columns=fp_cols)

    # Process Formulas
    comp_features = df['chemical_formula'].apply(get_composition_features).apply(pd.Series)
    
    # Process Categorical
    df_cat = pd.get_dummies(df[['crystal_system', 'space_group']], drop_first=True)
    
    # Other Structural Numerical Features
    struct_cols = ['Number of atoms', 'Unit cell volume', 'Space group number']
    df_struct = df[struct_cols].copy()
    
    # Combine all features
    X_all = pd.concat([df_fp, comp_features, df_cat, df_struct], axis=1)
    
    # Feature selection: remove low variance features
    selector = VarianceThreshold(threshold=0.1)
    X_reduced = pd.DataFrame(selector.fit_transform(X_all), columns=X_all.columns[selector.get_support()])
    
    print(f"Feature engineering complete. Total features: {X_reduced.shape[1]}")

    # Store all results
    all_results_summary = []

    for target in targets:
        print(f"\n{'='*60}")
        print(f"Training Models for: {target}")
        print(f"{'='*60}")
        
        # Prepare target and mask missing values
        y = df[target]
        mask = y.notna()
        
        # Get the original indices for the filtered data BEFORE filtering X_reduced
        original_indices = df[mask].index.values
        
        # Filter features and target, reset index to ensure 0-based indexing
        X = X_reduced[mask].reset_index(drop=True)
        y = y[mask].reset_index(drop=True)
        
        if len(y) < 50:
            print(f"  Skipping {target}: insufficient samples ({len(y)} < 50)")
            continue

        # Family-aware train/test split to prevent leakage between parent and functionalized MOFs
        if 'mof_uri' in df.columns:
            # Get MOF families for the filtered subset
            mof_uris_filtered = df.loc[df[target].notna(), 'mof_uri'].reset_index(drop=True)
            groups = mof_uris_filtered.apply(get_mof_family)
            n_unique_groups = groups.nunique()

            if n_unique_groups >= 5:
                # Use GroupShuffleSplit for family-aware splitting
                gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                train_idx, test_idx = next(gss.split(X, y, groups=groups))
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                print(f"  Using family-aware split: {n_unique_groups} unique MOF families")
            else:
                # Fall back to random split if too few groups
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                print(f"  Using random split (only {n_unique_groups} MOF families)")
        else:
            # No mof_uri column, use random split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            print(f"  Using random split (no mof_uri column)")

        # Map test indices back to original dataframe indices
        # Use iloc indices from X_test to map back
        test_original_indices = original_indices[X_test.index.values]
        
        # Imputation and Scaling
        imputer = SimpleImputer(strategy='median')
        scaler = StandardScaler()
        
        X_train_processed = scaler.fit_transform(imputer.fit_transform(X_train))
        X_test_processed = scaler.transform(imputer.transform(X_test))
        
        # Store predictions for this target
        target_predictions = {
            'mof_uri': df.loc[test_original_indices, 'mof_uri'].values if 'mof_uri' in df.columns else None,
            'actual': y_test.values
        }
        
        # Hyperparameter grids for each model
        param_grids = {
            'RandomForest': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'XGBoost': {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'Ridge': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            }
        }
        
        results = {}
        
        # Random Forest with Grid Search
        print(f"\n  Training RandomForest with grid search...")
        rf_base = RandomForestRegressor(random_state=42)
        rf_best, rf_params, rf_cv_score = perform_grid_search(
            rf_base, param_grids['RandomForest'], X_train_processed, y_train
        )
        rf_best.fit(X_train_processed, y_train)
        y_pred_rf = rf_best.predict(X_test_processed)
        
        results['RandomForest'] = {
            'R2': r2_score(y_test, y_pred_rf),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'MAE': mean_absolute_error(y_test, y_pred_rf),
            'best_params': rf_params,
            'cv_score': rf_cv_score
        }
        target_predictions['RandomForest_predicted'] = y_pred_rf
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{target.replace(' ', '_')}_RandomForest.joblib")
        joblib.dump(rf_best, model_path)
        print(f"    R2 = {results['RandomForest']['R2']:.4f}, RMSE = {results['RandomForest']['RMSE']:.4f}")
        print(f"    Best params: {rf_params}")
        
        # XGBoost with Grid Search
        print(f"\n  Training XGBoost with grid search...")
        xgb_base = xgb.XGBRegressor(random_state=42)
        xgb_best, xgb_params, xgb_cv_score = perform_grid_search(
            xgb_base, param_grids['XGBoost'], X_train_processed, y_train
        )
        xgb_best.fit(X_train_processed, y_train)
        y_pred_xgb = xgb_best.predict(X_test_processed)
        
        results['XGBoost'] = {
            'R2': r2_score(y_test, y_pred_xgb),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_xgb)),
            'MAE': mean_absolute_error(y_test, y_pred_xgb),
            'best_params': xgb_params,
            'cv_score': xgb_cv_score
        }
        target_predictions['XGBoost_predicted'] = y_pred_xgb
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{target.replace(' ', '_')}_XGBoost.joblib")
        joblib.dump(xgb_best, model_path)
        print(f"    R2 = {results['XGBoost']['R2']:.4f}, RMSE = {results['XGBoost']['RMSE']:.4f}")
        print(f"    Best params: {xgb_params}")
        
        # Ridge with Grid Search
        print(f"\n  Training Ridge with grid search...")
        ridge_base = Ridge()
        ridge_best, ridge_params, ridge_cv_score = perform_grid_search(
            ridge_base, param_grids['Ridge'], X_train_processed, y_train
        )
        ridge_best.fit(X_train_processed, y_train)
        y_pred_ridge = ridge_best.predict(X_test_processed)
        
        results['Ridge'] = {
            'R2': r2_score(y_test, y_pred_ridge),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            'MAE': mean_absolute_error(y_test, y_pred_ridge),
            'best_params': ridge_params,
            'cv_score': ridge_cv_score
        }
        target_predictions['Ridge_predicted'] = y_pred_ridge
        
        # Save model
        model_path = os.path.join(MODEL_DIR, f"{target.replace(' ', '_')}_Ridge.joblib")
        joblib.dump(ridge_best, model_path)
        print(f"    R2 = {results['Ridge']['R2']:.4f}, RMSE = {results['Ridge']['RMSE']:.4f}")
        print(f"    Best params: {ridge_params}")
        
        # Save prediction data for this target
        pred_df = pd.DataFrame(target_predictions)
        pred_path = os.path.join(RESULTS_DIR, f"predictions_{target.replace(' ', '_')}.csv")
        pred_df.to_csv(pred_path, index=False)
        print(f"\n  Prediction data saved to {pred_path}")
        
        # Store summary results
        for model_name, metrics in results.items():
            all_results_summary.append({
                'property': target,
                'model': model_name,
                'R2': metrics['R2'],
                'RMSE': metrics['RMSE'],
                'MAE': metrics['MAE'],
                'cv_score': metrics['cv_score'],
                'best_params': json.dumps(metrics['best_params'])
            })
    
    # Save overall results summary
    summary_df = pd.DataFrame(all_results_summary)
    summary_path = os.path.join(OUTPUT_DIR, 'model_performance_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    print(f"\n{'='*60}")
    print(f"All models trained and saved successfully.")
    print(f"Performance summary saved to {summary_path}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()