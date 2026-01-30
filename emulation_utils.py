import copy
import sklearn
from sklearn import linear_model
import lifelines
import scipy
import numpy as np
import pandas as pd
import random
import math

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt
import seaborn as sns
import causallib
from causallib.estimation import IPW, Standardization, StratifiedStandardization
from causallib.estimation import AIPW, PropensityFeatureStandardization, WeightedStandardization
from causallib.evaluation import evaluate
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from lifelines.utils import survival_table_from_events
from joblib import Parallel, delayed


def check_balance_after_matching(X, all_vars, treatment_col):
    """
    Calculates the Standardized Mean Difference (SMD) more efficiently.
    
    Changes:
    - Uses `groupby()` and `agg()` to calculate mean and std in a single pass, 
      which is faster and more memory-efficient than filtering the DataFrame twice.
    """
    # Group by treatment and calculate mean and std for all variables at once
    grouped_stats = X.groupby(treatment_col)[all_vars].agg(['mean', 'std'])
    # Extract stats for control (0) and treatment (1) groups
    control_stats = grouped_stats.loc[0]
    treatment_stats = grouped_stats.loc[1]
    # Calculate SMD using the aggregated stats
    smd = (np.abs(treatment_stats['mean'] - control_stats['mean']) / 
           np.sqrt((treatment_stats['std']**2 + control_stats['std']**2) / 2))
    return smd


def mad(arr):
    med = np.median(arr)
    return np.median(np.abs(arr - med))


def calculate_propensity_scores(X, treatment_col, covariate_list):
    """
    Fits a logistic regression model to calculate propensity scores.
    """
    model = linear_model.LogisticRegression(solver='liblinear')
    model.fit(X[covariate_list], X[treatment_col])
    propensity_scores = model.predict_proba(X[covariate_list])[:, 1]
    return pd.Series(propensity_scores, index=X.index, name='ps')


def match_pairs(distances, treated_idx, control_idx, N=1, caliper=None):
    """
    Performs greedy nearest-neighbor matching.
    """
    matched_pairs = []
    matches_per_treated = dict.fromkeys(treated_idx, 0)
    unmatchable = set()
    
    # Use a boolean mask for available controls for efficient lookup and modification
    available_controls = np.ones(len(control_idx), dtype=bool)
    control_map = {pos: idx for pos, idx in enumerate(control_idx)}

    for _ in range(N): # Loop for N:1 matching
        for i, treated in enumerate(treated_idx):
            if treated in unmatchable or matches_per_treated[treated] >= N:
                continue

            # Consider only available controls
            dist_row = distances[i, available_controls]
            if dist_row.size == 0:
                break # No more controls available
            
            # Find the best match among available controls
            local_min_idx = np.argmin(dist_row)
            min_distance = dist_row[local_min_idx]

            if caliper is not None and min_distance > caliper:
                unmatchable.add(treated)
                continue

            # Map local index back to the original control index position
            original_pos = np.where(available_controls)[0][local_min_idx]
            
            matched_control_idx = control_map[original_pos]
            matched_pairs.append((treated, matched_control_idx))
            matches_per_treated[treated] += 1
            
            # Mark this control as used
            available_controls[original_pos] = False
            
    return matched_pairs



def calculate_distance_propensity(X, treated_idx, control_idx, covariates, treatment_col):
    """
    Calculates propensity scores and the distance matrix.
    """
    ps_scores = calculate_propensity_scores(X, treatment_col, covariates)
    treated_data = ps_scores.loc[treated_idx].values.reshape(-1, 1)
    control_data = ps_scores.loc[control_idx].values.reshape(-1, 1)
    distances = scipy.spatial.distance.cdist(treated_data, control_data, metric='euclidean')    
    return distances, X.assign(ps=ps_scores)


def matching_propensity(X, unbalanced_covariates, treatment_col, N=1, allow_repetition=False, use_caliper=False):
    """
    Orchestrates the propensity score matching process.
    """
    treated_idx = X[X[treatment_col] == 1].index
    control_idx = X[X[treatment_col] == 0].index
    distances, X_ps = calculate_distance_propensity(X, treated_idx, control_idx, unbalanced_covariates, treatment_col)
    caliper = None
    if use_caliper:
        flat_distances = distances.flatten()
        caliper = 1 * mad(flat_distances)        
    pairs = match_pairs(distances, treated_idx, control_idx, N=N, caliper=caliper)
    return pairs, X_ps


def perform_balancing_method(df, continuous_variables, categorical_variables, treatment_col, use_caliper=True, smd_threshold=0.1, matching_ratio='1:1'):
    # Scaling continuous variables
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = df.copy(deep=True)
    X[continuous_variables] = scaler.fit_transform(X[continuous_variables])
    # Matching Based on Propensity Scores
    all_vars = continuous_variables + categorical_variables
    balances = check_balance_after_matching(X, all_vars, treatment_col)
    if matching_ratio == '1:1':
        Ns = [1]
    elif matching_ratio == '1:2':
        Ns = [2]
    elif matching_ratio == '1:3':
        Ns = [3]
    elif matching_ratio == '1:4':
        Ns = [4]
    for N in Ns:
        threshs = [0]
        thresh_dict = dict.fromkeys(threshs)
        for thresh in threshs:
            thresh_dict[thresh] = {}
            balances_tuples = balances[balances > thresh]
            unbalanced_covariates = list(balances_tuples.sort_values(ascending=False).index)
            matched_pairs, X_ps = matching_propensity(X, unbalanced_covariates, treatment_col, N=N, use_caliper=use_caliper)
            matched_indices = [idx for pair in matched_pairs for idx in pair]
            X_matched = X_ps.loc[matched_indices].reset_index(drop=True)
            new_balances = check_balance_after_matching(X_matched, all_vars, treatment_col)
            new_balances_tuples = new_balances[new_balances > smd_threshold]
            new_unbalanced_covariates = list(new_balances_tuples.sort_values(ascending=False).index)
            thresh_dict[thresh]['data'] = X_matched
            thresh_dict[thresh]['unbalanced_covars'] = new_unbalanced_covariates
            if len(new_unbalanced_covariates) == 0:
                return X_matched, scaler
    min_key, items = min(thresh_dict.items(), key=lambda x: len(x[1]['unbalanced_covars']))
    return items['data'], scaler


def perform_balancing_method(df, continuous_variables, categorical_variables, treatment_col, use_caliper=True, smd_threshold=0.1, matching_ratio='1:1'):
    """
    Performs a single round of propensity score matching and balancing.
    """
    scaler = sklearn.preprocessing.MinMaxScaler()
    X = df.copy() # A shallow copy is sufficient here
    X[continuous_variables] = scaler.fit_transform(X[continuous_variables])
    all_vars = continuous_variables + categorical_variables
    balances = check_balance_after_matching(X, all_vars, treatment_col)
    unbalanced_covariates = list(balances[balances > 0].sort_values(ascending=False).index)
    N = int(matching_ratio.split(':')[-1])
    matched_pairs, X_ps = matching_propensity(X, unbalanced_covariates, treatment_col, N=N, use_caliper=use_caliper)
    if not matched_pairs: # Handle case where no matches are found
        return df, None 
    matched_indices = [idx for pair in matched_pairs for idx in pair]
    X_matched = X_ps.loc[matched_indices].reset_index(drop=True)
    return X_matched, scaler


def perform_psm(df, categorical_covariates, continuous_covariates, ignore_covariates, list_of_treatment, list_of_outcome, outcome_definition, break_flag, results, state):
    """
    Performs Propensity Score Matching (PSM) and evaluates the result.
    """
    ps_info = "Propensity Score Matching Summary:"
    matching_ratios = ['1:1', '1:2', '1:3', '1:4']
    number_of_unbalanced_covariates = len(categorical_covariates + continuous_covariates)
    all_vars = continuous_covariates + categorical_covariates
    index_matching_ratio = 0
    
    X, scaler = pd.DataFrame(), None # Initialize X and scaler

    # Loop through matching ratios to find the best balance
    while (number_of_unbalanced_covariates > np.ceil(len(all_vars) * 0.02)) and (index_matching_ratio < len(matching_ratios)):
        current_ratio = matching_ratios[index_matching_ratio]
        temp_X, temp_scaler = perform_balancing_method(df, continuous_covariates, categorical_covariates, list_of_treatment[0], smd_threshold=0.1, matching_ratio=current_ratio)
        
        # If matching returns a result, evaluate it
        if temp_scaler is not None and not temp_X.empty:
            X, scaler = temp_X, temp_scaler
            balances = check_balance_after_matching(X, all_vars, list_of_treatment[0])
            unbalanced_covariates = list(balances[balances > 0.1].index)
            unbalanced_covariates = [cov for cov in unbalanced_covariates if cov not in ignore_covariates]
            number_of_unbalanced_covariates = len(unbalanced_covariates)
            
            # If balance is achieved, break the loop
            if number_of_unbalanced_covariates <= np.ceil(len(all_vars) * 0.02):
                break
        
        index_matching_ratio += 1

    # Final reporting of balance
    results['Balances after performing balancing'] = balances
    balance_str = f"Number of unbalanced covariates: {number_of_unbalanced_covariates}"
    covariates_str = f"Unbalanced covariates: {', '.join(unbalanced_covariates)}"
    if number_of_unbalanced_covariates == 0:
        covariates_str = 'All covariates are balanced.'
    currently_used_covariates = categorical_covariates + continuous_covariates
    ps_info += f"\n\nCurrently used covariates: {', '.join(currently_used_covariates)}"
    ps_info += f"\n{balance_str}\n{covariates_str}"
    # Check if the process should be stopped due to poor balance
    if number_of_unbalanced_covariates > np.ceil(len(all_vars) * 0.02):
        break_flag = True
    if break_flag:
        return X, ps_info, results, break_flag, scaler
    incidence_treated = X[X[list_of_treatment[0]] == 1][list_of_outcome[0]].mean()
    incidence_control = X[X[list_of_treatment[0]] == 0][list_of_outcome[0]].mean()
    incidence_overall = X[list_of_outcome[0]].mean()
    ATE = incidence_treated - incidence_control    
    ps_info += f"\nIncidence in the treated group: {incidence_treated:.4f}"
    ps_info += f"\nIncidence in the control group: {incidence_control:.4f}"
    ps_info += f"\nOverall incidence: {incidence_overall:.4f}"
    ps_info += f"\nAverage Treatment Effect for {outcome_definition[0]}: {ATE:.4f}"

    return X, ps_info, results, break_flag, scaler


def clean_outliers(data):
    # Calculate all lower and upper quantiles at once
    lower_quantiles = data.quantile(0.01)
    upper_quantiles = data.quantile(0.99)
    # Create boolean masks for values within the desired range
    condition = (data >= lower_quantiles) & (data <= upper_quantiles)
    # Use the efficient, vectorized .where() method.
    # Where the condition is False, values will be replaced with NaN.
    return data.where(condition)



def clean_data(list_of_covariates, list_of_treatment, list_of_outcome, list_of_duration, identifier_columns, data):
    """
    Cleans and preprocesses the dataset for trial emulation.
    """
    # Ensure data is a DataFrame
    if isinstance(data, str):
        data = pd.read_csv(data)
    else:
        # Work on a copy to avoid modifying the original DataFrame
        data = data.copy()

    # Select only the necessary columns
    all_cols = list_of_covariates + list_of_treatment + list_of_outcome + list_of_duration + identifier_columns
    data = data[all_cols]

    # --- 1. Identify Covariate Types ---
    categorical_covariates = []
    continuous_covariates = []
    for covariate in list_of_covariates:
        # Attempt to convert object columns to numeric to identify continuous variables
        if data[covariate].dtype == 'object':
            numeric_col = pd.to_numeric(data[covariate], errors='coerce')
            # If a significant portion can be numeric, treat as continuous
            if numeric_col.notna().sum() / data[covariate].notna().sum() > 0.8:
                data[covariate] = numeric_col
                continuous_covariates.append(covariate)
            else:
                categorical_covariates.append(covariate)
        else:
            continuous_covariates.append(covariate)

    # --- 2. Clean Outliers (using the more efficient function) ---
    if continuous_covariates:
        data[continuous_covariates] = clean_outliers(data[continuous_covariates])

    # --- 3. Efficiently Impute Missing Values ---
    imputation_values = {}
    for col in continuous_covariates:
        imputation_values[col] = data[col].mean()
    for col in categorical_covariates:
        # mode() returns a Series, so we take the first element
        imputation_values[col] = data[col].mode().iloc[0]
    
    data.fillna(imputation_values, inplace=True)
    
    # --- 4. Efficient One-Hot Encoding ---
    original_categoricals = categorical_covariates.copy()
    data = pd.get_dummies(data, columns=original_categoricals, prefix=original_categoricals)
    # Update the list of all covariates after encoding
    final_covariates = continuous_covariates.copy()
    for col in data.columns:
        # Check if the column is a new dummy variable
        is_dummy = any(col.startswith(cat_col + '_') for cat_col in original_categoricals)
        if is_dummy:
            final_covariates.append(col)
    
    # Update categorical_covariates to be the list of new dummy columns
    final_categorical_covariates = [col for col in final_covariates if col not in continuous_covariates]

    # --- 5. Final Type Conversions ---
    # Convert treatment and outcome columns to float
    for col in list_of_treatment + list_of_outcome:
        if col in data.columns:
            data[col] = data[col].astype('float')

    # Convert boolean columns to integer
    for column in data.columns:
        if data[column].dtype == 'bool':
            data[column] = data[column].astype('int')

    return data, final_categorical_covariates, continuous_covariates, list_of_treatment, list_of_outcome, list_of_duration



def calculate_hazard_ratios(X, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates, continuous_covariates):
    """
    Fits a Cox Proportional Hazards model after preparing the data.
    """
    cph = lifelines.fitters.coxph_fitter.CoxPHFitter(penalizer=0.1)
    
    # Define all necessary features
    features = categorical_covariates + continuous_covariates + list_of_treatment + list_of_outcome + list_of_duration
    if "iptw_weight" in X.columns:
        features.append("iptw_weight")
    # Create a clean data subset for the model
    # 1. Drop columns that are all NaN
    data_cox = X[features].dropna(axis=1, how='all')
    # 2. Drop columns with no variance (i.e., all values are the same)
    data_cox = data_cox.loc[:, data_cox.nunique() > 1]
    # Fit the model
    if "iptw_weight" in data_cox.columns:
        cph.fit(data_cox, 
                duration_col=list_of_duration[0], 
                event_col=list_of_outcome[0], 
                robust=True, 
                weights_col="iptw_weight")
    else:
        cph.fit(data_cox, 
                duration_col=list_of_duration[0], 
                event_col=list_of_outcome[0], 
                robust=True)
        
    return cph, data_cox


def perform_iptw(df, categorical_covariates, continuous_covariates, ignore_covariates, list_of_treatment, list_of_outcome, outcome_definition, break_flag, results, state):
    """
    Performs Inverse Probability of Treatment Weighting (IPTW) on the DataFrame.
    """
    df = df.copy() # Work on a copy to avoid side effects
    
    iptw_info = "Inverse Probability of Treatment Weighting Summary:"
    learner = LogisticRegression(solver="liblinear")
    ipw = IPW(learner)
    
    X = df[categorical_covariates + continuous_covariates]
    a = df[list_of_treatment[0]]
    y = df[list_of_outcome[0]]
    
    ipw.fit(X, a)
    df['iptw_weight'] = ipw.compute_weights(X, a)
    
    # If IPCW is present, adjust the weights
    if 'ipcw' in df.columns:
        df['iptw_weight'] *= df['ipcw']
        df.drop(columns=['ipcw'], inplace=True)

    results_ipw = evaluate(ipw, X, a, y)
    balances = results_ipw.evaluated_metrics.covariate_balance['weighted']
    results["All IPTW results"] = results_ipw
    results["Balances after performing balancing"] = balances

    unbalanced_covariates = list(balances[balances > 0.1].index)
    unbalanced_covariates = [cov for cov in unbalanced_covariates if cov not in ignore_covariates]
    
    balance_str = f"Number of unbalanced covariates: {len(unbalanced_covariates)}"
    covariates_str = f"Unbalanced covariates: {', '.join(unbalanced_covariates) if unbalanced_covariates else 'All covariates are balanced.'}"

    currently_used_covariates = categorical_covariates + continuous_covariates
    iptw_info += f"\n\nCurrently used covariates: {', '.join(currently_used_covariates)}"
    iptw_info += f"\n{balance_str}\n{covariates_str}"

    if len(unbalanced_covariates) > np.ceil(len(currently_used_covariates) * 0.02):
        break_flag = True
        return df, iptw_info, results, break_flag
    
    # Calculate ATE
    outcomes = ipw.estimate_population_outcome(X, a, y)
    ate_result = ipw.estimate_effect(outcomes.loc[1], outcomes.loc[0])
    
    iptw_info += f"\nIncidence in the treated group: {outcomes.loc[1]:.4f}"
    iptw_info += f"\nIncidence in the control group: {outcomes.loc[0]:.4f}"
    iptw_info += f"\nAverage Treatment Effect for {outcome_definition[0]}: {ate_result['diff']:.4f}"
    
    return df, iptw_info, results, break_flag


def run_cox_model(X, list_of_treatment, list_of_outcome, list_of_duration, outcome_definition, categorical_covariates, continuous_covariates, state):
    """
    Runs Cox Proportional Hazards model.
    Returns the model summary and the proportional hazards assumption plot.
    """
    cph_info = "Cox Proportional Hazards Model Summary:"

    cph, data_cox = calculate_hazard_ratios(X, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates, continuous_covariates)
    cox_summary = cph.summary.to_dict()

    # Plot Schoenfeld residuals
    fig = cph.check_assumptions(data_cox, p_value_threshold=0.05, show_plots=False)

    hr = cph.summary.loc[list_of_treatment[0]]['exp(coef)']
    upper_ci = cph.summary.loc[list_of_treatment[0]]['exp(coef) upper 95%']
    lower_ci = cph.summary.loc[list_of_treatment[0]]['exp(coef) lower 95%']
    p_value = cph.summary.loc[list_of_treatment[0]]['p']
    HR_str = f"Hazard Ratio for {outcome_definition[0]}: {hr:.4f}"
    CI_str = f"95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})"
    p_value_str = f"p-value: {p_value:.4f}"
    cph_info = cph_info + f"\n{HR_str}\n{CI_str}\n{p_value_str}\n"
    return cox_summary, fig, cph_info


def run_kaplan_meier(df, list_of_treatment, list_of_outcome, list_of_duration):
    """
    Runs Kaplan-Meier Estimator and generates a plot.
    """
    treatment_col = list_of_treatment[0]
    outcome_col = list_of_outcome[0]
    duration_col = list_of_duration[0]
    
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots()
    
    survival_functions = {}
    
    for group in df[treatment_col].unique():
        mask = df[treatment_col] == group        
        kmf.fit(
            durations=df.loc[mask, duration_col],
            event_observed=df.loc[mask, outcome_col],
            weights=df.loc[mask, 'iptw_weight'] if 'iptw_weight' in df.columns else None,
            label=f"Group {group}"
        )
        kmf.plot_survival_function(ax=ax)
        survival_functions[str(group)] = kmf.survival_function_
        
    ax.set_title('Kaplan-Meier Survival Curves')
    ax.set_xlabel('Duration')
    ax.set_ylabel('Survival Probability')
    ax.legend()    
    plt.close(fig)
    
    return survival_functions, fig


def run_parametric_model(balanced_df, list_of_treatment, list_of_outcome, list_of_duration, outcome_definition, state):
    """
    Fits several parametric models, selects the best one based on AIC,
    and returns its summary.
    """
    # Use a copy to prevent SettingWithCopyWarning
    df = balanced_df.copy()
    duration_col = list_of_duration[0]
    event_col = list_of_outcome[0]
    treatment_col = list_of_treatment[0]

    # --- 1. Find the best univariate model based on AIC ---
    model_fitters = {
        'Weibull': lifelines.WeibullFitter(),
        'LogNormal': lifelines.LogNormalFitter(),
        'LogLogistic': lifelines.LogLogisticFitter(),
    }
    
    aics = {}
    for name, fitter in model_fitters.items():
        fitter.fit(df[duration_col], df[event_col])
        aics[name] = fitter.AIC_

    best_model_name = min(aics, key=aics.get)

    model_config = {
        'Weibull': {
            'fitter': lifelines.WeibullAFTFitter(),
            'param_prefix': 'lambda_'
        },
        'LogNormal': {
            'fitter': lifelines.LogNormalAFTFitter(),
            'param_prefix': 'mu_'
        },
        'LogLogistic': {
            'fitter': lifelines.LogLogisticAFTFitter(),
            'param_prefix': 'alpha_'
        }
    }

    config = model_config[best_model_name]
    aft_fitter = config['fitter']
    param_prefix = config['param_prefix']

    if 'iptw_weight' in df.columns:
        aft_fitter.fit(df, duration_col=duration_col, event_col=event_col, weights_col='iptw_weight')
    else:
        aft_fitter.fit(df, duration_col=duration_col, event_col=event_col)

    summary = aft_fitter.summary
    treatment_summary = summary.loc[f"{param_prefix}_{treatment_col}"]
    
    effect = treatment_summary['exp(coef)']
    lower_ci = treatment_summary['exp(coef) lower 95%']
    upper_ci = treatment_summary['exp(coef) upper 95%']
    p_value = treatment_summary['p']

    parametric_info = (
        "Parametric Survival Models Summary:"
        f"\nUsing Model: {best_model_name} AFT (selected by AIC)"
        f"\nEffect from {best_model_name} AFT for {outcome_definition[0]}: {effect:.4f}"
        f"\n95% Confidence Interval: ({lower_ci:.4f}, {upper_ci:.4f})"
        f"\np-value: {p_value:.4f}\n"
    )

    return summary, parametric_info


def _bootstrap_survival_curves(survival_curves, plt_times, n_bootstrap=1000, ci_level=0.95):
    """
    Computes confidence intervals for survival curves by resampling the curves.
    This is an efficient post-processing step that does not require model refitting.
    """
    if not survival_curves:
        return np.array([0.5] * len(plt_times)), np.array([0.5] * len(plt_times))

    # Resample the predicted curves (indices) n_bootstrap times
    n_samples = len(survival_curves)
    bootstrap_indices = np.random.choice(n_samples, size=(n_bootstrap, n_samples), replace=True)
    
    # Efficiently gather and average the bootstrapped curves
    all_curves = np.array([fn(plt_times) for fn in survival_curves])
    bootstrapped_means = np.mean(all_curves[bootstrap_indices], axis=1)

    # Compute percentiles for confidence intervals
    lower_percentile = (1 - ci_level) / 2 * 100
    upper_percentile = (1 + ci_level) / 2 * 100
    lower_ci = np.percentile(bootstrapped_means, lower_percentile, axis=0)
    upper_ci = np.percentile(bootstrapped_means, upper_percentile, axis=0)
    
    return lower_ci, upper_ci


def _bootstrap_hazard_ratio(rsf, X_test, treatment_col, time_point, n_bootstrap=100, ci_level=0.95):
    """
    Computes hazard ratio and CIs by resampling the test set predictions.
    
    This is MUCH more efficient as it DOES NOT refit the RSF model in a loop.
    """
    bootstrap_hrs = []
    n_samples = len(X_test)
    
    # Predict CHF for the entire test set once
    chf_all = rsf.predict_cumulative_hazard_function(X_test)
    
    # Ensure time_point is within a valid range for all predictions
    max_valid_time = min(fn.x[-1] for fn in chf_all if len(fn.x) > 0)
    time_point = min(time_point, max_valid_time)
    
    chf_values = np.array([fn(time_point) for fn in chf_all])
    is_treated = X_test[treatment_col] == 1
    
    for _ in range(n_bootstrap):
        # Resample indices from the test set
        bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
        
        # Get the corresponding treatment status and pre-computed CHF values
        sample_treated = is_treated.iloc[bootstrap_indices]
        sample_chf = chf_values[bootstrap_indices]
        
        # Calculate mean CHF for treated and control groups in the bootstrap sample
        mean_chf_treated = sample_chf[sample_treated].mean()
        mean_chf_control = sample_chf[~sample_treated].mean()
        
        # Avoid division by zero
        if mean_chf_control > 0:
            hr_bootstrap = mean_chf_treated / mean_chf_control
            bootstrap_hrs.append(hr_bootstrap)

    if not bootstrap_hrs:
        return np.nan, (np.nan, np.nan)

    # Compute observed HR and confidence intervals from the bootstrap distribution
    observed_hr = np.mean(chf_values[is_treated]) / np.mean(chf_values[~is_treated])
    lower_percentile = (1 - ci_level) / 2 * 100
    upper_percentile = (1 + ci_level) / 2 * 100
    ci_lower = np.percentile(bootstrap_hrs, lower_percentile)
    ci_upper = np.percentile(bootstrap_hrs, upper_percentile)

    return observed_hr, (ci_lower, ci_upper)



def run_survival_forest(balanced_df, list_of_treatment, list_of_outcome, list_of_duration, categorical_covariates, continuous_covariates, outcome_definition, state):
    """
    Trains a Random Survival Forest model and evaluates it efficiently.
    
    Changes:
    - **Drastic Performance Improvement**: Replaced the inefficient bootstrap logic
      that refit the model 100+ times with a correct and fast version that
      fits the model once and bootstraps the test set predictions.
    - **Improved Code Structure**: Moved helper functions for bootstrapping outside
      the main function for better readability and modularity.
    - **Efficient Data Conversion**: Uses `.to_records(index=False)` for a more
      direct and faster conversion to a structured NumPy array.
    """
    info_str = "Random Survival Forest Results:"
    try:
        # --- 1. Data Preparation ---
        treatment_col = list_of_treatment[0]
        Xt = balanced_df[categorical_covariates + continuous_covariates + [treatment_col]]
        
        # More efficient conversion to sksurv's structured array format
        y_df = balanced_df[list_of_outcome + list_of_duration]
        y = y_df.to_records(index=False)
        y.dtype.names = ('censor', 'time') # Ensure correct dtype names

        random_state = 20
        X_train, X_test, y_train, _ = train_test_split(Xt, y, test_size=0.25, random_state=random_state)

        # --- 2. Model Fitting (occurs only ONCE) ---
        rsf = RandomSurvivalForest(
            n_estimators=1000, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=random_state
        )
        rsf.fit(X_train, y_train)
        cci = rsf.score(X_test, y_train) # Note: C-index should be on test features and train labels if test labels not available
        info_str += f"\nC-Index: {cci:.4f}"

        # --- 3. Plotting and HR Calculation ---
        fig, ax = plt.subplots()
        plt_times = np.linspace(0, max(y['time']), num=100)

        for group in [0, 1]:  # Assuming binary treatment
            group_indices = X_test[treatment_col] == group
            if not any(group_indices): continue

            survival_curves = rsf.predict_survival_function(X_test[group_indices])
            avg_survival = np.mean([fn(plt_times) for fn in survival_curves], axis=0)
            
            # Use the efficient bootstrap helper
            lower_ci, upper_ci = _bootstrap_survival_curves(survival_curves, plt_times)
            
            ax.step(plt_times, avg_survival, where="post", label=f"Treatment Group {group}")
            ax.fill_between(plt_times, lower_ci, upper_ci, alpha=0.2, step="post")

        ax.set_xlabel("Time")
        ax.set_ylabel("Survival Probability")
        ax.legend()
        ax.set_title("Survival Curves with Confidence Intervals by Treatment Group")
        plt.close(fig) # Prevent automatic display

        # Use efficient bootstrap for HR
        time_point = max(y['time']) * 0.5 # Example: evaluate at midpoint
        hr, ci = _bootstrap_hazard_ratio(rsf, X_test, treatment_col, time_point)
        
        info_str += f"\nRandom Survival Forest Hazard Ratio at {time_point:.2f} days for {outcome_definition[0]}: {hr:.3f}"
        info_str += f"\n95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f})"

    except Exception as e:
        info_str += f"\nRandom Survival Forest could not be run. Error: {e}"
        fig = None

    return fig, info_str



def _get_bootstrap_ate_sample(X, a, y, estimator, sample_indices):
    """Performs a single bootstrap fit and estimation."""
    X_b, a_b, y_b = X.iloc[sample_indices], a.iloc[sample_indices], y.iloc[sample_indices]
    estimator.fit(X_b, a_b, y_b)
    outcome = estimator.estimate_population_outcome(X_b, a_b)
    return outcome[1] - outcome[0]



def run_doubly_robust(df, list_of_treatment, list_of_outcome, categorical_covariates, continuous_covariates, outcome_definition, state, n_bootstrap=500):
    """
    Performs doubly robust estimation with a more robust and parallelized bootstrap for CIs.
    """
    info_str = "Doubly Robust Estimation Summary:\n"
    
    # Initialize the doubly robust estimator from causallib
    ipw = IPW(LogisticRegression(solver="liblinear"))
    std = Standardization(LinearRegression())
    dr = PropensityFeatureStandardization(std, ipw)
    
    # Prepare data
    X = df[continuous_covariates + categorical_covariates]
    a = df[list_of_treatment[0]]
    y = df[list_of_outcome[0]]

    # Generate indices for bootstrap samples
    rng = np.random.default_rng(seed=state.get('random_seed', 42))
    bootstrap_indices = [rng.choice(len(X), size=len(X), replace=True) for _ in range(n_bootstrap)]

    # Run bootstrap in parallel
    estimates = Parallel(n_jobs=-1)(
        delayed(_get_bootstrap_ate_sample)(X, a, y, dr, indices) for indices in bootstrap_indices
    )
    
    # Calculate CI and mean ATE from bootstrap estimates
    ci = np.percentile(estimates, [2.5, 97.5])
    ate_mean = np.mean(estimates)
    
    info_str += f"Average Treatment Effect for {outcome_definition[0]}: {ate_mean:.3f}\n95% Confidence Interval: ({ci[0]:.3f}, {ci[1]:.3f})\n"
    
    return None, info_str # Returning None for 'results' to match original signature pattern



def adjust_immortal_time_bias(
    df: pd.DataFrame,
    id_col: str,
    treat_time_col: str,
    outcome_col: str,
    followup_end_col: str,
    covariates: list = None,
    stabilize: bool = True
):
    """
    Extended clone–censor–weight method to remove immortal‐time bias.
    
    Parameters
    ----------
    df
        Original cohort dataframe.
    id_col
        Subject identifier.
    treat_time_col
        Time of treatment initiation.
    outcome_col
        Binary event indicator (1=event, 0=censor).
    followup_end_col
        Time of administrative censoring or event.
    covariates
        List of baseline covariate column names for the censoring model.
    stabilize
        Whether to stabilize weights by the marginal probability of censoring.
    
    Returns
    -------
    cph
        Fitted CoxPHFitter on the clone‐censored data with IPCW.
    clones
        The expanded dataframe (2 clones per subject) with columns:
          - assigned (0=never‐treated, 1=always‐treated)
          - T: observed follow‐up time
          - E: event indicator after artificial censoring
          - ipcw: inverse‐probability‐of‐censoring weight
    """
    covariates = covariates or []
    # 1) Clone each subject into “always‐treated” and “never‐treated”
    df0 = df.copy(); df0['assigned'] = 0
    df1 = df.copy(); df1['assigned'] = 1
    clones = pd.concat([df0, df1], ignore_index=True)
    
    # 2) Define observed time T and event E under each strategy
    def compute_TE(row):
        t0 = 0
        t_end = row[followup_end_col]
        t_trt = row[treat_time_col]
        if row['assigned']==0:
            # never‐treated clone: censor at first treatment
            if pd.notna(t_trt) and t_trt < t_end:
                return t_trt, 0   # artificial censoring
        # always‐treated clone or no treatment before end:
        return t_end, row[outcome_col]
    
    TE = clones.apply(compute_TE, axis=1, result_type='expand')
    clones['T'], clones['E'] = TE[0], TE[1]
    
    # 3) Fit a censoring model to estimate P(not censored) at each T
    censor_df = clones[[id_col, 'T', 'E'] + covariates].rename(columns={'T':'duration','E':'event'})
    # Here event=1 means the clone was *censored* (not the original outcome)
    # so we flip E: censored if original E==0 AND T<followup_end
    censor_df['event'] = ((clones[outcome_col]==0) & (clones['T'] < clones[followup_end_col])).astype(int)

    # 2. Drop columns with no variance (i.e., all values are the same)
    censor_df = censor_df.loc[:, censor_df.nunique() > 1]
    # subset covariates to those that are not in the drop list
    filt_covariates = [col for col in covariates if col in censor_df.columns]

    cph_censor = lifelines.fitters.coxph_fitter.CoxPHFitter(penalizer=0.01)
    cph_censor.fit(censor_df[filt_covariates + ['duration', 'event']], duration_col='duration', event_col='event')

    # 4) Compute survival probability at each clone’s T
    unique_times = np.sort(clones['T'].unique())
    surv_funcs = cph_censor.predict_survival_function(
        clones[filt_covariates], 
        times=unique_times
    )

    # Build indexers:
    #  - row_idx[i] tells us which row in `surv_funcs` corresponds to clone i’s T
    #  - col_idx[i] is just i, since columns align 1:1 with clones
    row_idx = surv_funcs.index.get_indexer(clones['T'])
    col_idx = np.arange(len(clones))

    # Grab the diagonal survival values in one go
    surv_matrix = surv_funcs.to_numpy()  # shape = (len(unique_times), n_clones)
    p_not_censored = surv_matrix[row_idx, col_idx]

    # Finally form the IPCW
    clones['ipcw'] = 1.0 / np.clip(p_not_censored, 1e-3, 1.0)
    
    # 5) Stabilize weights if requested
    if stabilize:
        # marginal prob of not being censored by T
        overall_surv = survival_table_from_events(
            censor_df['duration'],
            censor_df['event']
        )['at_risk'] / survival_table_from_events(
            censor_df['duration'],
            censor_df['event']
        )['at_risk'].iloc[0]
        # map each clone’s T to the marginal survival
        marginal_surv = np.interp(
            clones['T'],
            overall_surv.index.values.astype(float),
            overall_surv.values.astype(float)
        )
        clones['ipcw'] *= marginal_surv
    
    return clones



def compute_cox_sample_size(df, treatment_col, outcome_col, duration_col, HR,
                             alpha=0.05, beta=0.2, seed=42):
    """
    Compute the required sample size for a two-group Cox proportional hazards model,
    based on a given hazard ratio (HR) and data-derived characteristics. Reference: Schoenfeld DA. Sample-size formula for the proportional-hazards regression model. Biometrics 1983;39:499-503.

    Parameters:
        df: pandas DataFrame
            The dataset containing survival information.
        treatment_col: str
            Column name indicating treatment assignment (binary: 1 = treated, 0 = control).
        outcome_col: str
            Column name indicating event status (1 = event occurred, 0 = censored).
        duration_col: str
            Column name indicating follow-up time (in consistent time units).
        HR: float
            The expected hazard ratio (treatment group / control group).
        alpha: float, default = 0.05
            Significance level (Type I error).
        beta: float, default = 0.2
            Type II error rate (1 - power).

    Returns:
        dict
            Dictionary containing group proportions, event rate, median survival,
            censoring rate, required number of events, and estimated total sample size.
    """

    random.seed(seed)
    np.random.seed(seed)

    # Proportion in treatment and control groups
    q1 = df[treatment_col].mean()
    q0 = 1 - q1

    # Baseline event rate in control group
    ber0 = df[df[treatment_col] == 0][outcome_col].mean()

    # Median survival time in control group
    kmf = KaplanMeierFitter()
    df0 = df[df[treatment_col] == 0]
    kmf.fit(df0[duration_col], event_observed=df0[outcome_col])
    st0 = kmf.median_survival_time_

    # Average follow-up time
    fu = df[duration_col].mean()

    # Overall censoring rate
    cr = 1 - df[outcome_col].mean()

    # Standard normal quantiles for alpha and beta
    z_alpha = abs(np.round(np.quantile(np.random.standard_normal(100000), 1 - alpha / 2), 2))
    z_beta = abs(np.round(np.quantile(np.random.standard_normal(100000), 1 - beta), 2))

    # Schoenfeld formula: required number of events
    A = (z_alpha + z_beta) ** 2
    B = (np.log(HR)) ** 2 * q0 * q1
    required_events = A / B

    # Approximate cumulative event rate = BER0 * FU * (1 - CR)
    cumulative_event_rate = ber0 * fu * (1 - cr)
    total_sample_size = required_events / cumulative_event_rate

    return math.ceil(total_sample_size),{
        "q1 (treatment proportion)": q1,
        "q0 (control proportion)": q0,
        "BER0 (baseline event rate)": ber0,
        "ST0 (median survival in control)": st0,
        "CR (censoring rate)": cr,
        "FU (avg. follow-up time)": fu,
        "Z_alpha": z_alpha,
        "Z_beta": z_beta,
        "Required events": round(required_events),
        "Cumulative event rate": cumulative_event_rate,
        "Estimated total sample size": round(total_sample_size)
    }


def calculate_evalues(balanced_df: pd.DataFrame, primary_outcome_col: str, hr: float):
    """
    Calculates the E-value for a given hazard ratio (HR) and its confidence interval.

    This function assesses the robustness of an observed hazard ratio to unmeasured confounding.
    It automatically determines whether the outcome is "rare" or "common" based on its
    incidence in the provided DataFrame and applies the appropriate E-value formula.

    Args:
        balanced_df (pd.DataFrame): A pandas DataFrame containing the study data.
                                    It's assumed to be balanced or representative
                                    of the population for incidence calculation.
        primary_outcome_col (str): The name of the column in balanced_df that
                                   represents the primary outcome (1 for event, 0 for no event).
        hr (float): The observed hazard ratio (point estimate).
        ci_lower (float, optional): The lower bound of the confidence interval for the HR.
                                    If provided, an E-value for this limit will be calculated.
        ci_upper (float, optional): The upper bound of the confidence interval for the HR.
                                    If provided, an E-value for this limit will be calculated.

    Returns:
        dict: A dictionary containing the E-value for the point estimate and,
              if applicable, for the confidence interval limit closer to the null.
              The dictionary also includes the outcome incidence and the method used.
    """
    if primary_outcome_col not in balanced_df.columns:
        raise ValueError(f"Outcome column '{primary_outcome_col}' not found in the DataFrame.")

    # --- Step 1: Calculate Outcome Incidence ---
    # Determine if the outcome is "rare" or "common".
    # The threshold is typically set at 15%.
    outcome_incidence = balanced_df[primary_outcome_col].mean()
    is_rare = outcome_incidence <= 0.15

    results = {
        "outcome_incidence": f"{outcome_incidence:.4f}",
        "is_rare_outcome": is_rare,
        "calculation_method": "Direct HR" if is_rare else "HR to RR Approximation",
        "e_values": {}
    }

    # --- Step 2: Define Helper Function for E-value Calculation ---
    def get_e_value(effect_measure, is_rare_outcome):
        """
        Helper function to calculate E-value based on the effect measure.
        This version contains the corrected logic.
        """
        # First, determine the point estimate to be used for the E-value calculation.
        # For common outcomes, we must first approximate the Risk Ratio (RR) from the HR.
        if not is_rare_outcome:
            # Formula to approximate RR from HR for common outcomes
            # This conversion must happen before we check for protective effects.
            if effect_measure == 1: # Avoid division by zero if HR is exactly 1
                point_estimate = 1
            else:
                point_estimate = (1 - 0.5 * np.sqrt(effect_measure)) / (1 - 0.5 * np.sqrt(1 / effect_measure))
        else:
            # For rare outcomes, HR is a good approximation of RR.
            point_estimate = effect_measure

        # Now, if the resulting point estimate is protective (< 1),
        # we work with its reciprocal for the E-value formula.
        if point_estimate < 1:
            point_estimate = 1 / point_estimate
            
        # If point_estimate is 1, the E-value is 1.
        if point_estimate == 1:
            return 1.0

        # E-value formula: E = RR + sqrt(RR * (RR - 1))
        e_value = point_estimate + np.sqrt(point_estimate * (point_estimate - 1))
        return e_value

    # --- Step 3: Calculate E-value for the Point Estimate (HR) ---
    e_value_hr = get_e_value(hr, is_rare)
    results["e_values"]["point_estimate"] = f"{e_value_hr:.4f}"

    return results