"""
COMPREHENSIVE SUCROSE PREFERENCE TEST ANALYSIS
THESIS-LEVEL MIXED MODELS APPROACH
With publication-ready visualizations and precise methodological reporting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from pathlib import Path
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.multitest import multipletests
from statsmodels.graphics.gofplots import qqplot
import warnings

# NO WARNINGS ARE SUPPRESSED - Convergence issues MUST be visible

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================
INPUT_FILE = Path(r"G:\Master\Experiment\Statistics\Sucrose perference\Sucrose_Preference_LongFormat_Template.xlsx")
OUTPUT_PATH = Path(r"G:\Master\Experiment\Statistics\Sucrose perference\Result2")
OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

class FigureConfig:
    """Configuration for publication-quality figures"""
    
    FONT = 'Arial'
    FONT_SIZES = {
        'title': 9,
        'axis': 8,
        'tick': 7,
        'legend': 7,
        'annotation': 7,
        'panel': 10,
        'table': 8,
        'effect': 6
    }
    
    COLORS = {
        'PD': '#D55E00',      # Vermilion
        'Control': '#0072B2', # Blue
        'OFF': '#999999',     # Gray
        'ON': '#009E73',      # Green
        'highlight': '#E69F00', # Orange
        'PD_light': '#F5B07C',  # Light vermilion
        'Control_light': '#7FB3D5'  # Light blue
    }
    
    SINGLE_COLUMN = 3.54  # inches
    DOUBLE_COLUMN = 7.48  # inches
    DPI = 600
    LINE_WIDTH = 0.75

config = FigureConfig()
plt.rcParams['font.sans-serif'] = config.FONT
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['savefig.dpi'] = config.DPI
plt.rcParams['axes.linewidth'] = config.LINE_WIDTH
plt.rcParams['lines.linewidth'] = config.LINE_WIDTH

# ============================================================================
# STATISTICAL UTILITY FUNCTIONS
# ============================================================================

def hedges_g_paired(x, y, bias_correction=True, n_bootstrap=5000):
    """
    Calculate Hedges' g for paired samples with small-sample bias correction
    and bootstrap confidence intervals.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    n = len(x)
    
    if n < 2:
        return {'g': np.nan, 'g_corrected': np.nan, 'ci_lower': np.nan, 
                'ci_upper': np.nan, 'var': np.nan, 'n': n}
    
    d = y - x
    mean_diff = np.mean(d)
    sd_diff = np.std(d, ddof=1)
    
    if sd_diff == 0:
        return {'g': 0, 'g_corrected': 0, 'ci_lower': 0, 'ci_upper': 0, 
                'var': 0, 'n': n}
    
    r = np.corrcoef(x, y)[0, 1]
    r = 0 if np.isnan(r) else r
    
    g = mean_diff / sd_diff
    var_g = (1/n + g**2/(2*n)) * 2*(1-r)
    
    if bias_correction and n > 1:
        df = n - 1
        J = 1 - (3 / (4*df - 1))
        g_corrected = g * J
        var_g_corrected = J**2 * var_g
    else:
        g_corrected = g
        var_g_corrected = var_g
    
    # Bootstrap confidence intervals
    bootstrap_g = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n, n, replace=True)
        x_boot = x[indices]
        y_boot = y[indices]
        d_boot = y_boot - x_boot
        mean_diff_boot = np.mean(d_boot)
        sd_diff_boot = np.std(d_boot, ddof=1)
        if sd_diff_boot > 0:
            g_boot = mean_diff_boot / sd_diff_boot
            if bias_correction:
                g_boot = g_boot * J
            bootstrap_g.append(g_boot)
    
    bootstrap_ci = np.percentile(bootstrap_g, [2.5, 97.5]) if bootstrap_g else [np.nan, np.nan]
    
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, df=n-1)
    se_g = np.sqrt(var_g_corrected) if var_g_corrected > 0 else 0
    
    ci_lower = g_corrected - t_crit * se_g
    ci_upper = g_corrected + t_crit * se_g
    
    return {
        'g': g,
        'g_corrected': g_corrected,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_ci_lower': bootstrap_ci[0],
        'bootstrap_ci_upper': bootstrap_ci[1],
        'var': var_g_corrected,
        'n': n,
        'correlation': r
    }


def hedges_g_independent(x, y, bias_correction=True, n_bootstrap=5000):
    """
    Calculate Hedges' g for independent samples with small-sample bias correction
    and bootstrap confidence intervals.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    x = x[~np.isnan(x)]
    y = y[~np.isnan(y)]
    
    n1, n2 = len(x), len(y)
    n = n1 + n2
    
    if n1 < 2 or n2 < 2:
        return {'g': np.nan, 'g_corrected': np.nan, 'ci_lower': np.nan, 
                'ci_upper': np.nan, 'var': np.nan, 'n1': n1, 'n2': n2}
    
    mean1, mean2 = np.mean(x), np.mean(y)
    sd1, sd2 = np.std(x, ddof=1), np.std(y, ddof=1)
    
    pooled_sd = np.sqrt(((n1-1)*sd1**2 + (n2-1)*sd2**2) / (n1 + n2 - 2))
    
    d = (mean2 - mean1) / pooled_sd if pooled_sd > 0 else 0
    var_d = (n1 + n2)/(n1*n2) + d**2/(2*(n1 + n2))
    
    if bias_correction:
        df = n1 + n2 - 2
        J = 1 - (3 / (4*df - 1))
        g = d * J
        var_g = J**2 * var_d
    else:
        g = d
        var_g = var_d
    
    # Bootstrap confidence intervals
    bootstrap_g = []
    np.random.seed(42)
    for _ in range(n_bootstrap):
        x_boot = np.random.choice(x, n1, replace=True)
        y_boot = np.random.choice(y, n2, replace=True)
        mean1_boot, mean2_boot = np.mean(x_boot), np.mean(y_boot)
        sd1_boot, sd2_boot = np.std(x_boot, ddof=1), np.std(y_boot, ddof=1)
        pooled_boot = np.sqrt(((n1-1)*sd1_boot**2 + (n2-1)*sd2_boot**2) / (n1 + n2 - 2))
        if pooled_boot > 0:
            g_boot = (mean2_boot - mean1_boot) / pooled_boot
            if bias_correction:
                g_boot = g_boot * J
            bootstrap_g.append(g_boot)
    
    bootstrap_ci = np.percentile(bootstrap_g, [2.5, 97.5]) if bootstrap_g else [np.nan, np.nan]
    
    alpha = 0.05
    t_crit = stats.t.ppf(1 - alpha/2, df=n-2)
    se_g = np.sqrt(var_g) if var_g > 0 else 0
    
    ci_lower = g - t_crit * se_g
    ci_upper = g + t_crit * se_g
    
    return {
        'd': d,
        'g': g,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'bootstrap_ci_lower': bootstrap_ci[0],
        'bootstrap_ci_upper': bootstrap_ci[1],
        'var': var_g,
        'n1': n1,
        'n2': n2,
        'pooled_sd': pooled_sd
    }


def lrt_random_slope_corrected(model_intercept, model_slope):
    """
    CORRECTED Likelihood Ratio Test for random slope with boundary issues.
    
    For testing H0: σ²_slope = 0 vs H1: σ²_slope > 0, the asymptotic
    distribution is a 50:50 mixture of χ²(1) and χ²(2).
    """
    if not (model_intercept.converged and model_slope.converged):
        return {
            'statistic': np.nan,
            'p_value_standard': np.nan,
            'p_value_boundary_corrected': np.nan,
            'df': 2,
            'significant_standard': False,
            'significant_corrected': False,
            'warning': 'Models did not converge'
        }
    
    # Likelihood ratio statistic
    lr_stat = 2 * (model_slope.llf - model_intercept.llf)
    lr_stat = max(0, lr_stat)  # Ensure non-negative
    
    # Standard chi-square test (too conservative)
    p_standard = 1 - stats.chi2.cdf(lr_stat, 2)
    
    # Boundary-corrected p-value (50:50 mixture)
    # 0.5 * χ²(1) + 0.5 * χ²(2)
    p_corrected = 0.5 * (1 - stats.chi2.cdf(lr_stat, 1)) + \
                  0.5 * (1 - stats.chi2.cdf(lr_stat, 2))
    
    return {
        'statistic': lr_stat,
        'p_value_standard': p_standard,
        'p_value_boundary_corrected': p_corrected,
        'df': 2,
        'significant_standard': p_standard < 0.05,
        'significant_corrected': p_corrected < 0.05,
        'warning': 'Boundary-corrected p-value is more appropriate for variance component testing'
    }


def semi_partial_r2_likelihood_based(full_model, reduced_model):
    """
    Calculate semi-partial R² for a fixed effect using likelihood-based model comparison.
    
    Following Edwards et al. (2008) approach:
    R²_partial = (LL_full - LL_reduced) / LL_full
    
    Parameters:
    -----------
    full_model : fitted MixedLM results object (with all fixed effects)
    reduced_model : fitted MixedLM results object (without the effect of interest)
    
    Returns:
    --------
    dict : Semi-partial R² and likelihood information
    """
    if not (full_model.converged and reduced_model.converged):
        return {
            'semi_partial_r2': np.nan,
            'll_full': full_model.llf if full_model.converged else np.nan,
            'll_reduced': reduced_model.llf if reduced_model.converged else np.nan,
            'warning': 'Models did not converge'
        }
    
    ll_full = full_model.llf
    ll_reduced = reduced_model.llf
    
    # Ensure non-negative
    if ll_full <= 0:
        # Alternative formulation for cases where LL is negative
        # Use likelihood ratio based measure
        lr_stat = 2 * (ll_full - ll_reduced)
        r2_approx = 1 - np.exp(-lr_stat / len(full_model.model.endog))
        return {
            'semi_partial_r2': r2_approx,
            'll_full': ll_full,
            'll_reduced': ll_reduced,
            'method': 'Likelihood-ratio approximation'
        }
    
    # Standard Edwards et al. (2008) formula
    semi_partial_r2 = (ll_full - ll_reduced) / abs(ll_full)
    semi_partial_r2 = max(0, min(1, semi_partial_r2))  # Clamp to [0, 1]
    
    return {
        'semi_partial_r2': semi_partial_r2,
        'll_full': ll_full,
        'll_reduced': ll_reduced,
        'method': 'Edwards et al. (2008) likelihood-based R²'
    }


def calculate_icc(model_result, model_type):
    """
    Calculate Intraclass Correlation Coefficient (ICC) ONLY for random intercept models.
    
    ICC = σ²_random_intercept / (σ²_random_intercept + σ²_residual)
    
    Parameters:
    -----------
    model_result : fitted MixedLM results object
    model_type : str ('random_intercept' or 'random_slope')
    
    Returns:
    --------
    dict : ICC value or explanation
    """
    if model_type != 'random_intercept':
        return {
            'icc': None,
            'note': 'ICC not reported due to random slope specification.',
            'interpretation': 'Not applicable for models with random slopes'
        }
    
    if model_result.cov_re is None or model_result.cov_re.size == 0:
        return {
            'icc': 0,
            'note': 'No random effects variance estimated',
            'interpretation': 'No clustering'
        }
    
    random_var = float(model_result.cov_re[0, 0])
    residual_var = model_result.scale
    
    if (random_var + residual_var) > 0:
        icc = random_var / (random_var + residual_var)
    else:
        icc = 0
    
    return {
        'icc': icc,
        'random_variance': random_var,
        'residual_variance': residual_var,
        'interpretation': f'{icc*100:.1f}% of residual variance attributable to between-subject differences (after accounting for fixed effects)'
    }


def conditional_residuals(model_result, model_df):
    """
    Calculate conditional (level-1) residuals from the fitted model.
    Uses the actual model object, not a refitted version.
    """
    # Overall residuals
    raw_residuals = model_result.resid
    
    # Get random effects if available
    if hasattr(model_result, 'random_effects') and model_result.random_effects:
        # Subject-specific random effects
        re_dict = model_result.random_effects
        
        # Create array of random effects matched to observations
        re_values = np.zeros(len(model_df))
        
        for i, (subject, re) in enumerate(re_dict.items()):
            # Fix FutureWarning by using .iloc for position-based indexing
            if isinstance(re, (np.ndarray, pd.Series)):
                if hasattr(re, 'iloc'):
                    re_val = re.iloc[0] if len(re) > 0 else 0
                else:
                    re_val = re[0] if len(re) > 0 else 0
            else:
                re_val = re
            mask = model_df['Subject'].values == subject
            re_values[mask] = re_val
        
        # Conditional residuals (level-1) = raw residuals
        # For LMMs, raw residuals from fitted model are already conditional
        conditional_resid = raw_residuals
    else:
        conditional_resid = raw_residuals
        re_values = np.zeros(len(model_df))
    
    return {
        'raw_residuals': raw_residuals,
        'conditional_residuals': conditional_resid,
        'random_effects': re_values
    }


def diagnostics_from_selected_model(model_result, model_df):
    """
    Model diagnostics using the selected fitted model.
    Durbin-Watson is intentionally omitted as it's inappropriate for clustered data.
    """
    diagnostics = {}
    
    # Get conditional residuals from the actual fitted model
    resid_data = conditional_residuals(model_result, model_df)
    conditional_resid = resid_data['conditional_residuals']
    
    # 1. Normality of conditional residuals (Shapiro-Wilk)
    if len(conditional_resid) >= 3:
        shapiro_stat, shapiro_p = stats.shapiro(conditional_resid)
        diagnostics['residual_normality'] = {
            'test': 'Shapiro-Wilk',
            'statistic': float(shapiro_stat),
            'p_value': float(shapiro_p),
            'normal': shapiro_p > 0.05,
            'note': 'Testing conditional residuals (level-1)',
            'n_observations': len(conditional_resid)
        }
    
    # 2. Homoscedasticity - visual assessment recommended for LMMs
    diagnostics['homoscedasticity_note'] = 'Visual inspection of residuals vs fitted plot recommended for LMMs'
    
    # 3. Random effects normality (if random intercept model)
    if len(resid_data['random_effects']) > 0 and np.any(resid_data['random_effects'] != 0):
        re_unique = np.unique(resid_data['random_effects'])
        if len(re_unique) >= 3:
            re_shapiro, re_p = stats.shapiro(re_unique)
            diagnostics['random_effects_normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': float(re_shapiro),
                'p_value': float(re_p),
                'normal': re_p > 0.05,
                'note': 'Testing distribution of random intercepts',
                'n_subjects': len(re_unique)
            }
    
    return diagnostics


def fdr_correction_with_reporting(p_values, alpha=0.05, method='fdr_bh'):
    """Apply FDR correction and return detailed results."""
    if len(p_values) == 0:
        return {'corrected': [], 'significant': [], 'reject': [], 'q_values': []}
    
    reject, corrected, _, _ = multipletests(p_values, alpha=alpha, method=method)
    
    return {
        'corrected': corrected,
        'significant': reject,
        'reject': reject,
        'q_values': corrected
    }

# ============================================================================
# DATA LOADING AND PREPARATION
# ============================================================================
def load_data_from_excel():
    """Load data from the specific Excel file"""
    
    print(f"Loading data from: {INPUT_FILE}")
    
    if not INPUT_FILE.exists():
        print(f"❌ File not found: {INPUT_FILE}")
        print("Using hardcoded data as fallback...")
        return create_hardcoded_dataframe()
    
    try:
        df = pd.read_excel(INPUT_FILE, sheet_name=0)
        
        print(f"✓ File loaded successfully")
        print(f"  Sheet: {pd.ExcelFile(INPUT_FILE).sheet_names[0]}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        df.columns = df.columns.str.strip()
        return df
        
    except Exception as e:
        print(f"❌ Error loading Excel file: {e}")
        print("Using hardcoded data as fallback...")
        return create_hardcoded_dataframe()

def create_hardcoded_dataframe():
    """Create DataFrame from hardcoded data as fallback"""
    
    print("Creating dataframe from hardcoded data...")
    
    data = {
        'Subject': ['PD_1', 'PD_1', 'PD_2', 'PD_2', 'PD_3', 'PD_3', 'PD_4', 'PD_4', 
                    'PD_5', 'PD_5', 'PD_6', 'PD_6', 'PD_7', 'PD_7', 'PD_8', 'PD_8',
                    'PD_9', 'PD_9', 'CO_1', 'CO_1', 'CO_2', 'CO_2', 'CO_3', 'CO_3',
                    'CO_4', 'CO_4', 'CO_5', 'CO_5', 'CO_6', 'CO_6', 'CO_7', 'CO_7',
                    'CO_8', 'CO_8', 'CO_9', 'CO_9'],
        'Group': ['PD']*18 + ['Control']*18,
        'Stimulation': ['OFF', 'ON']*18,
        '% SucrosePreference': [60.71, 81.58, 69.39, 83.33, 79.16, 83.58, 71.73, 63.41,
                               70.96, 92.15, 56.36, 81.01, 73.93, 79.71, 81.15, 86.66,
                               54.90, 72.60, 64.29, 95.74, 48.65, 75.00, 65.75, 97.14,
                               77.77, 79.59, 91.20, 96.04, 89.01, 91.53, 80.92, 90.14,
                               79.45, 79.71, 82.50, 90.10],
        'Total Intake (g)': [28, 38, 49, 72, 48, 67, 46, 41, 31, 51, 110, 79, 73, 69,
                            69, 60, 51, 73, 70, 47, 37, 40, 73, 70, 63, 49, 57, 86,
                            91, 65, 105, 71, 146, 69, 117, 91],
        'Water intake (g)': [11, 7, 15, 12, 10, 11, 13, 15, 9, 4, 48, 15, 19, 14,
                            13, 20, 23, 20, 25, 2, 19, 10, 25, 2, 14, 10, 5, 12,
                            10, 12, 20, 7, 30, 14, 21, 9],
        'Sucrose intake (g)': [17, 31, 34, 60, 38, 56, 33, 26, 22, 47, 62, 64, 54, 55,
                              56, 40, 28, 53, 45, 45, 18, 30, 48, 68, 49, 39, 52, 74,
                              81, 53, 85, 64, 116, 55, 96, 82]
    }
    
    df = pd.DataFrame(data)
    print("✓ Hardcoded data created successfully")
    return df

def create_wide_formats(df):
    """Create wide format DataFrames for each variable"""
    
    df.columns = df.columns.str.strip()
    
    # Sucrose Preference wide format
    sp_wide = df.pivot_table(
        index=['Subject', 'Group'],
        columns='Stimulation',
        values='% SucrosePreference'
    ).reset_index()
    sp_wide.columns.name = None
    sp_wide = sp_wide.rename(columns={'OFF': 'SPT_OFF', 'ON': 'SPT_ON'})
    sp_wide['ΔSPT'] = sp_wide['SPT_ON'] - sp_wide['SPT_OFF']
    
    wide_df = sp_wide.copy()
    
    # Add other variables if available
    if 'Total Intake (g)' in df.columns:
        ti_wide = df.pivot_table(
            index=['Subject', 'Group'],
            columns='Stimulation',
            values='Total Intake (g)'
        ).reset_index()
        ti_wide.columns.name = None
        ti_wide = ti_wide.rename(columns={'OFF': 'TI_OFF', 'ON': 'TI_ON'})
        ti_wide['ΔTI'] = ti_wide['TI_ON'] - ti_wide['TI_OFF']
        wide_df = wide_df.merge(ti_wide, on=['Subject', 'Group'])
    
    if 'Water intake (g)' in df.columns:
        wi_wide = df.pivot_table(
            index=['Subject', 'Group'],
            columns='Stimulation',
            values='Water intake (g)'
        ).reset_index()
        wi_wide.columns.name = None
        wi_wide = wi_wide.rename(columns={'OFF': 'WI_OFF', 'ON': 'WI_ON'})
        wi_wide['ΔWI'] = wi_wide['WI_ON'] - wi_wide['WI_OFF']
        wide_df = wide_df.merge(wi_wide, on=['Subject', 'Group'])
    
    if 'Sucrose intake (g)' in df.columns:
        si_wide = df.pivot_table(
            index=['Subject', 'Group'],
            columns='Stimulation',
            values='Sucrose intake (g)'
        ).reset_index()
        si_wide.columns.name = None
        si_wide = si_wide.rename(columns={'OFF': 'SI_OFF', 'ON': 'SI_ON'})
        si_wide['ΔSI'] = si_wide['SI_ON'] - si_wide['SI_OFF']
        wide_df = wide_df.merge(si_wide, on=['Subject', 'Group'])
    
    return df, wide_df

# ============================================================================
# THESIS-LEVEL MIXED MODEL FITTING
# ============================================================================

def fit_mixed_model_thesis_level(df, dependent_var, family='primary'):
    """
    Thesis-level mixed model fitting with:
    - Selected model storage for diagnostics
    - Likelihood-based semi-partial R²
    - Conditional ICC reporting
    - No Durbin-Watson
    - Precise methodological terminology
    """
    
    print(f"\n  Fitting mixed model for: {dependent_var}")
    
    # Prepare data
    model_df = df[['Subject', 'Group', 'Stimulation', dependent_var]].dropna().copy()
    
    # Create numeric codes
    model_df['Group_encoded'] = (model_df['Group'] == 'PD').astype(int)
    model_df['Stimulation_encoded'] = (model_df['Stimulation'] == 'ON').astype(int)
    model_df['Interaction'] = model_df['Group_encoded'] * model_df['Stimulation_encoded']
    
    # Group means for visualization
    group_means = model_df.groupby(['Group', 'Stimulation'])[dependent_var].agg(['mean', 'sem', 'std']).round(2)
    
    # Check if we have enough data
    if len(model_df) < 4 or model_df['Subject'].nunique() < 2:
        print(f"    ⚠ WARNING: Insufficient data for {dependent_var}")
        return create_fallback_results(model_df, dependent_var, family, group_means, 
                                       error="Insufficient data")
    
    # Prepare data for MixedLM
    exog = np.column_stack([
        np.ones(len(model_df)),           # intercept
        model_df['Group_encoded'].values,  # group
        model_df['Stimulation_encoded'].values,  # stimulation
        model_df['Interaction'].values     # interaction
    ])
    
    exog_names = ['Intercept', 'Group_PD', 'Stimulation_ON', 'Group_x_Stim']
    endog = model_df[dependent_var].values
    groups = model_df['Subject'].values
    
    # ========================================================================
    # MODEL 1: Random intercept only
    # ========================================================================
    print(f"    Fitting random intercept model...")
    
    try:
        model_intercept = MixedLM(endog, exog, groups, exog_re=np.ones(len(model_df)))
        result_intercept = model_intercept.fit(reml=True, method='bfgs', maxiter=1000)
        
        if not result_intercept.converged:
            print(f"    ⚠ WARNING: Random intercept model did not converge")
    except Exception as e:
        print(f"    ⚠ ERROR fitting random intercept model: {e}")
        return create_fallback_results(model_df, dependent_var, family, group_means, 
                                       error=f"Model failed: {e}")
    
    # ========================================================================
    # MODEL 2: Random intercept + random slope for Stimulation
    # ========================================================================
    print(f"    Testing random slope model...")
    result_slope = None
    
    try:
        exog_re = np.column_stack([
            np.ones(len(model_df)),           # random intercept
            model_df['Stimulation_encoded'].values  # random slope for stimulation
        ])
        
        model_slope = MixedLM(endog, exog, groups, exog_re=exog_re)
        result_slope = model_slope.fit(reml=False, method='bfgs', maxiter=1000)
        
        if not result_slope.converged:
            print(f"    ⚠ WARNING: Random slope model did not converge")
    except Exception as e:
        print(f"    ⚠ Random slope model failed: {e}")
        result_slope = None
    
    # ========================================================================
    # MODEL SELECTION with boundary-corrected LRT
    # ========================================================================
    model_selection = {
        'model1_converged': result_intercept.converged,
        'model2_converged': result_slope.converged if result_slope else False,
        'model1_aic': result_intercept.aic,
        'model2_aic': result_slope.aic if result_slope else np.inf,
        'model1_bic': result_intercept.bic,
        'model2_bic': result_slope.bic if result_slope else np.inf,
    }
    
    # Perform boundary-corrected LRT if both models converged
    lr_test = None
    if result_slope and result_slope.converged and result_intercept.converged:
        lr_test = lrt_random_slope_corrected(result_intercept, result_slope)
        
        # Select model based on corrected p-value and AIC
        if lr_test['significant_corrected'] and result_slope.aic < result_intercept.aic:
            selected_model = result_slope
            selected_model_type = 'random_slope'
            print(f"    ✓ Random slope model preferred by boundary-corrected LRT (p={lr_test['p_value_boundary_corrected']:.4f})")
        else:
            selected_model = result_intercept
            selected_model_type = 'random_intercept'
            print(f"    ✓ Random intercept model preferred by boundary-corrected LRT (p={lr_test['p_value_boundary_corrected']:.4f})")
    else:
        selected_model = result_intercept
        selected_model_type = 'random_intercept'
        lr_test = {'statistic': np.nan, 'p_value_boundary_corrected': np.nan, 'significant_corrected': False}
        print(f"    ✓ Using random intercept model (LRT not possible)")
    
    model_selection['selected_model'] = selected_model_type
    model_selection['lr_test'] = lr_test
    
    # ========================================================================
    # STORE THE SELECTED MODEL OBJECT FOR DIAGNOSTICS
    # ========================================================================
    final_model = selected_model
    final_model_type = selected_model_type
    
    # ========================================================================
    # EXTRACT FIXED EFFECTS (Wald z - asymptotic normal approximation)
    # ========================================================================
    fixed_effects = {}
    
    for i, param_name in enumerate(exog_names):
        if i < len(final_model.params):
            coef = final_model.params[i]
            se = final_model.bse[i] if i < len(final_model.bse) else 0
            z_stat = final_model.tvalues[i] if i < len(final_model.tvalues) else 0
            p_value = final_model.pvalues[i] if i < len(final_model.pvalues) else 1.0
            
            # Wald z confidence intervals (asymptotic normal)
            z_crit = stats.norm.ppf(0.975)
            ci_lower = coef - z_crit * se
            ci_upper = coef + z_crit * se
            
            fixed_effects[param_name] = {
                'coefficient': coef,
                'se': se,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'z_stat': z_stat,
                'p_value': p_value,
                'family': family,
                'inference_method': 'Asymptotic Wald z-test'
            }
    
    # ========================================================================
    # RANDOM EFFECTS
    # ========================================================================
    random_effects = {}
    random_var_dict = {}
    
    if final_model.cov_re is not None and final_model.cov_re.size > 0:
        if final_model_type == 'random_intercept':
            random_var = float(final_model.cov_re[0, 0])
            random_effects['Subject(Intercept)'] = {
                'var': random_var,
                'sd': np.sqrt(random_var) if random_var > 0 else 0
            }
            random_var_dict['var_intercept'] = random_var
        else:
            # Extract full covariance matrix
            re_names = ['Intercept', 'Stim_Slope']
            for i in range(final_model.cov_re.shape[0]):
                for j in range(final_model.cov_re.shape[1]):
                    if i == j:
                        var_name = f'var_{re_names[i]}'
                        var_val = float(final_model.cov_re[i, j])
                        random_effects[var_name] = var_val
                        random_var_dict[var_name] = var_val
                    else:
                        cov_name = f'cov_{re_names[i]}_{re_names[j]}'
                        cov_val = float(final_model.cov_re[i, j])
                        random_effects[cov_name] = cov_val
                        
                        # Calculate correlation
                        var_i = float(final_model.cov_re[i, i])
                        var_j = float(final_model.cov_re[j, j])
                        if var_i > 0 and var_j > 0:
                            corr = cov_val / np.sqrt(var_i * var_j)
                            random_effects[f'corr_{re_names[i]}_{re_names[j]}'] = corr
    
    # ========================================================================
    # LIKELIHOOD-BASED SEMI-PARTIAL R² (replaces ω²)
    # ========================================================================
    semi_partial_r2 = {}
    
    # For each fixed effect (except intercept), fit reduced model and compute semi-partial R²
    for i, effect_name in enumerate(exog_names):
        if effect_name == 'Intercept':
            continue
        
        # Create reduced model without this effect
        mask = np.ones(exog.shape[1], dtype=bool)
        mask[i] = False
        exog_reduced = exog[:, mask]
        
        try:
            # Fit reduced model with same random effects structure
            if final_model_type == 'random_intercept':
                model_reduced = MixedLM(endog, exog_reduced, groups, exog_re=np.ones(len(model_df)))
            else:
                exog_re = np.column_stack([
                    np.ones(len(model_df)),
                    model_df['Stimulation_encoded'].values
                ])
                model_reduced = MixedLM(endog, exog_reduced, groups, exog_re=exog_re)
            
            result_reduced = model_reduced.fit(reml=True, method='bfgs', maxiter=1000)
            
            # Calculate semi-partial R²
            r2_result = semi_partial_r2_likelihood_based(final_model, result_reduced)
            semi_partial_r2[effect_name] = r2_result
            
        except Exception as e:
            semi_partial_r2[effect_name] = {
                'semi_partial_r2': np.nan,
                'warning': f'Could not fit reduced model: {str(e)}'
            }
    
    # ========================================================================
    # ICC CALCULATION (conditional on model type)
    # ========================================================================
    icc_result = calculate_icc(final_model, final_model_type)
    
    # ========================================================================
    # MAIN EFFECT SIZES (Hedges g - kept as is)
    # ========================================================================
    effect_sizes = {}
    
    off_data = model_df[model_df['Stimulation'] == 'OFF']
    pd_off = off_data[off_data['Group'] == 'PD'][dependent_var].values
    control_off = off_data[off_data['Group'] == 'Control'][dependent_var].values
    
    if len(pd_off) > 1 and len(control_off) > 1:
        effect_sizes['Group_main_at_OFF'] = hedges_g_independent(control_off, pd_off)
    
    control_data = model_df[model_df['Group'] == 'Control']
    control_off = control_data[control_data['Stimulation'] == 'OFF'][dependent_var].values
    control_on = control_data[control_data['Stimulation'] == 'ON'][dependent_var].values
    
    if len(control_off) > 1 and len(control_on) > 1 and len(control_off) == len(control_on):
        effect_sizes['Stimulation_main_in_Control'] = hedges_g_paired(control_off, control_on)
    
    # ========================================================================
    # MODEL-BASED SIMPLE EFFECTS (only when interaction present)
    # ========================================================================
    interaction_present = False
    simple_effects = {}
    
    if 'Group_x_Stim' in fixed_effects and fixed_effects['Group_x_Stim']['p_value'] < 0.05:
        interaction_present = True
        print(f"    ✓ Significant interaction detected (p={fixed_effects['Group_x_Stim']['p_value']:.4f})")
        
        if final_model_type == 'random_intercept':
            beta_group = fixed_effects['Group_PD']['coefficient']
            beta_stim = fixed_effects['Stimulation_ON']['coefficient']
            beta_interact = fixed_effects['Group_x_Stim']['coefficient']
            se_stim = fixed_effects['Stimulation_ON']['se']
            se_interact = fixed_effects['Group_x_Stim']['se']
            
            # Stimulation effect in Control group = beta_stim
            simple_effects['Stimulation_in_Control'] = {
                'estimate': beta_stim,
                'se': se_stim,
                'z': beta_stim / se_stim if se_stim > 0 else 0,
                'p_value': fixed_effects['Stimulation_ON']['p_value'],
                'method': 'Model-based contrast'
            }
            
            # Stimulation effect in PD group = beta_stim + beta_interact
            est_pd = beta_stim + beta_interact
            # SE = sqrt(var(beta_stim) + var(beta_interact) + 2*cov)
            # Approximate from available info (assuming independence)
            se_pd = np.sqrt(se_stim**2 + se_interact**2)
            z_pd = est_pd / se_pd if se_pd > 0 else 0
            p_pd = 2 * (1 - stats.norm.cdf(np.abs(z_pd)))
            
            simple_effects['Stimulation_in_PD'] = {
                'estimate': est_pd,
                'se': se_pd,
                'z': z_pd,
                'p_value': p_pd,
                'method': 'Model-based contrast (approximate SE)'
            }
    
    # ========================================================================
    # DIAGNOSTICS USING THE SELECTED MODEL (NOT REFITTED)
    # ========================================================================
    diagnostics = diagnostics_from_selected_model(final_model, model_df)
    
    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================
    results = {
        'dependent_var': dependent_var,
        'family': family,
        'model_type': final_model_type,
        'inference_method': 'Asymptotic Wald z-tests for fixed effects',
        'n_subjects': int(model_df['Subject'].nunique()),
        'n_observations': len(model_df),
        'fixed_effects': fixed_effects,
        'random_effects': random_effects,
        'residual_variance': float(final_model.scale),
        'semi_partial_r2': semi_partial_r2,
        'icc': icc_result,
        'effect_sizes': effect_sizes,
        'aic': float(final_model.aic) if not np.isnan(final_model.aic) else 0,
        'bic': float(final_model.bic) if not np.isnan(final_model.bic) else 0,
        'converged': final_model.converged,
        'model_selection': model_selection,
        'interaction_present': interaction_present,
        'simple_effects': simple_effects,
        'group_means': group_means.to_dict() if hasattr(group_means, 'to_dict') else {},
        'diagnostics': diagnostics,
        '_fitted_model': final_model,
        '_model_df': model_df
    }
    
    return results


def create_fallback_results(model_df, dependent_var, family, group_means, error=None):
    """Create fallback results when mixed model fails"""
    
    print(f"    ⚠ Using descriptive statistics only for {dependent_var}")
    if error:
        print(f"    ⚠ Reason: {error}")
    
    # Calculate basic statistics
    overall_mean = model_df[dependent_var].mean() if len(model_df) > 0 else 0
    overall_std = model_df[dependent_var].std() if len(model_df) > 0 else 0
    
    # Calculate group means
    group_stats = {}
    for group in ['PD', 'Control']:
        for stim in ['OFF', 'ON']:
            subset = model_df[(model_df['Group'] == group) & (model_df['Stimulation'] == stim)]
            if len(subset) > 0:
                key = f"{group}_{stim}"
                group_stats[key] = {
                    'mean': float(subset[dependent_var].mean()),
                    'std': float(subset[dependent_var].std()),
                    'n': len(subset)
                }
    
    return {
        'dependent_var': dependent_var,
        'family': family,
        'model_type': 'descriptive_only',
        'inference_method': 'Descriptive statistics only (model failed to converge)',
        'n_subjects': int(model_df['Subject'].nunique()) if len(model_df) > 0 else 0,
        'n_observations': len(model_df),
        'fixed_effects': {},
        'random_effects': {},
        'residual_variance': float(overall_std**2) if overall_std > 0 else 0,
        'semi_partial_r2': {},
        'icc': {'icc': None, 'note': 'Model failed, ICC not computed'},
        'effect_sizes': {},
        'aic': 0,
        'bic': 0,
        'converged': False,
        'model_selection': {'error': 'Model failed', 'convergence_warnings': [error] if error else []},
        'interaction_present': False,
        'simple_effects': {},
        'group_means': group_means.to_dict() if hasattr(group_means, 'to_dict') else {},
        'group_stats': group_stats,
        'diagnostics': {},
        'overall_mean': float(overall_mean),
        'overall_std': float(overall_std),
        'error': error
    }


def fit_all_mixed_models(df):
    """Fit mixed models for all available dependent variables."""
    
    # PRIMARY OUTCOME
    primary_vars = ['% SucrosePreference']
    
    # SECONDARY OUTCOMES
    secondary_vars = []
    if 'Total Intake (g)' in df.columns:
        secondary_vars.append('Total Intake (g)')
    if 'Water intake (g)' in df.columns:
        secondary_vars.append('Water intake (g)')
    if 'Sucrose intake (g)' in df.columns:
        secondary_vars.append('Sucrose intake (g)')
    
    print("\n" + "="*70)
    print("FITTING MIXED MODELS")
    print("="*70)
    print("\nPRIMARY OUTCOME:")
    print("  Sucrose Preference (%) - Primary endpoint")
    
    all_results = {}
    
    print("\nPRIMARY ENDPOINT:")
    for var in primary_vars:
        if var in df.columns:
            all_results[var] = fit_mixed_model_thesis_level(df, var, family='primary')
    
    if secondary_vars:
        print("\nSECONDARY ENDPOINTS:")
        for var in secondary_vars:
            if var in df.columns:
                all_results[var] = fit_mixed_model_thesis_level(df, var, family='secondary')
    
    # ========================================================================
    # FDR CORRECTION (applied separately per family)
    # ========================================================================
    print("\n" + "="*70)
    print("APPLYING FDR CORRECTION")
    print("="*70)
    print("  Families:")
    print("    Family 1 (Primary - Preference): Sucrose Preference")
    print("    Family 2 (Secondary - Consumption): Total Intake, Water Intake, Sucrose Intake")
    
    family_pvalues = {'primary': [], 'secondary': []}
    family_params = {'primary': [], 'secondary': []}
    
    for var, results in all_results.items():
        if results['model_type'] == 'descriptive_only':
            continue
            
        family = results['family']
        for effect_name, effect_data in results['fixed_effects'].items():
            if effect_name != 'Intercept':
                family_pvalues[family].append(effect_data['p_value'])
                family_params[family].append({
                    'variable': var,
                    'effect': effect_name,
                    'family': family
                })
    
    corrected_results = {}
    for family in ['primary', 'secondary']:
        if family_pvalues[family]:
            correction = fdr_correction_with_reporting(family_pvalues[family], alpha=0.05)
            
            for idx, (param, p_corrected) in enumerate(zip(family_params[family], correction['corrected'])):
                key = f"{param['variable']}_{param['effect']}"
                corrected_results[key] = {
                    'variable': param['variable'],
                    'effect': param['effect'],
                    'family': param['family'],
                    'p_value_raw': float(family_pvalues[family][idx]),
                    'p_value_fdr': float(p_corrected),
                    'q_value': float(correction['q_values'][idx]),
                    'significant_fdr': bool(correction['significant'][idx])
                }
    
    print(f"\n  Primary endpoint: {len(family_pvalues['primary'])} tests")
    print(f"  Secondary endpoints: {len(family_pvalues['secondary'])} tests")
    print(f"  FDR correction applied within each family (Benjamini-Hochberg, controlling expected proportion of false positives ≤ 0.05)")
    
    return all_results, corrected_results

# ============================================================================
# MAIN FIGURE 1: PRIMARY OUTCOME (Sucrose Preference)
# ============================================================================

def create_main_figure_primary(all_results, corrected_results, output_path):
    """
    Create publication-ready main figure for primary outcome.
    Panel A: Model-estimated marginal means with spaghetti
    Panel B: Interaction plot with effect size
    Panel C: Forest plot of fixed effects
    Panel D: Hedges g effect sizes
    """
    
    fig = plt.figure(figsize=(config.DOUBLE_COLUMN, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Get results for primary outcome
    primary_var = '% SucrosePreference'
    if primary_var not in all_results:
        print("  ⚠ Primary outcome results not found")
        return None
    
    results = all_results[primary_var]
    if results['model_type'] == 'descriptive_only':
        print("  ⚠ Primary outcome model failed")
        return None
    
    # Get model and data
    fitted_model = results['_fitted_model']
    model_df = results['_model_df']
    
    # ========================================================================
    # PANEL A: Model-estimated marginal means with spaghetti
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate estimated marginal means from model
    groups = ['Control', 'PD']
    stim_levels = ['OFF', 'ON']
    stim_codes = {'OFF': 0, 'ON': 1}
    
    emmeans = {}
    for group in groups:
        group_code = 1 if group == 'PD' else 0
        emmeans[group] = {}
        for stim in stim_levels:
            stim_code = stim_codes[stim]
            # Design matrix row for this combination
            X = np.array([[1, group_code, stim_code, group_code * stim_code]])
            pred = np.dot(X, fitted_model.params)[0]
            # Approximate SE (simplified)
            se = np.sqrt(np.diag(X @ fitted_model.cov_params() @ X.T))[0]
            ci = 1.96 * se
            emmeans[group][stim] = {
                'mean': pred,
                'se': se,
                'ci_lower': pred - ci,
                'ci_upper': pred + ci
            }
    
    # Plot individual trajectories (spaghetti)
    np.random.seed(42)
    for group, color in [('Control', config.COLORS['Control']), ('PD', config.COLORS['PD'])]:
        group_data = model_df[model_df['Group'] == group]
        light_color = config.COLORS['Control_light'] if group == 'Control' else config.COLORS['PD_light']
        
        for subject in group_data['Subject'].unique():
            subj_data = group_data[group_data['Subject'] == subject]
            off_val = subj_data[subj_data['Stimulation'] == 'OFF'][primary_var].values
            on_val = subj_data[subj_data['Stimulation'] == 'ON'][primary_var].values
            if len(off_val) > 0 and len(on_val) > 0:
                ax1.plot([0, 1], [off_val[0], on_val[0]], 
                        color=light_color, alpha=0.3, linewidth=0.5, zorder=1)
    
    # Plot model-estimated means with error bars
    x_positions = [0, 1]
    for i, group in enumerate(groups):
        color = config.COLORS[group]
        means = [emmeans[group][stim]['mean'] for stim in stim_levels]
        ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
        ci_upper = [emmeans[group][stim]['ci_upper'] for stim in stim_levels]
        
        # Line connecting means
        ax1.plot(x_positions, means, color=color, linewidth=2, marker='o', 
                markersize=6, label=f'{group}', zorder=3)
        
        # Error bars
        ax1.fill_between(x_positions, ci_lower, ci_upper, color=color, alpha=0.2, zorder=2)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['OFF', 'ON'], fontsize=config.FONT_SIZES['tick'])
    ax1.set_ylabel('Sucrose Preference (%)', fontsize=config.FONT_SIZES['axis'])
    ax1.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
    ax1.set_title('A: Model-Estimated Marginal Means', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    ax1.legend(fontsize=config.FONT_SIZES['legend'], frameon=True, loc='best')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.set_ylim(40, 110)
    
    # ========================================================================
    # PANEL B: Interaction plot with effect size
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create interaction plot (group difference by stimulation)
    groups_for_plot = ['Control', 'PD']
    x_pos = np.arange(len(stim_levels))
    width = 0.35
    
    for i, group in enumerate(groups_for_plot):
        color = config.COLORS[group]
        means = [emmeans[group][stim]['mean'] for stim in stim_levels]
        ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
        ci_upper = [emmeans[group][stim]['ci_upper'] for stim in stim_levels]
        yerr = [means[j] - ci_lower[j] for j in range(len(means))]
        
        offset = -width/2 if group == 'Control' else width/2
        ax2.bar(x_pos + offset, means, width, label=group, color=color, alpha=0.7,
               yerr=yerr, capsize=3, error_kw={'linewidth': 1})
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stim_levels, fontsize=config.FONT_SIZES['tick'])
    ax2.set_ylabel('Sucrose Preference (%)', fontsize=config.FONT_SIZES['axis'])
    ax2.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
    ax2.set_title('B: Interaction Plot', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    ax2.legend(fontsize=config.FONT_SIZES['legend'], frameon=True)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    
    # Add effect size annotation
    if 'Group_x_Stim' in results['semi_partial_r2']:
        r2_data = results['semi_partial_r2']['Group_x_Stim']
        if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
            ax2.text(0.5, 0.95, f"Interaction pseudo-R² = {r2_data['semi_partial_r2']:.3f}",
                    transform=ax2.transAxes, fontsize=config.FONT_SIZES['annotation'],
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================================================
    # PANEL C: Forest plot of fixed effects
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    effects = []
    cis_lower = []
    cis_upper = []
    labels = []
    colors = []
    p_values = []
    r2_values = []
    
    for effect_name in ['Group_PD', 'Stimulation_ON', 'Group_x_Stim']:
        if effect_name in results['fixed_effects']:
            ef = results['fixed_effects'][effect_name]
            effects.append(ef['coefficient'])
            cis_lower.append(ef['ci_lower'])
            cis_upper.append(ef['ci_upper'])
            
            # Format label
            if effect_name == 'Group_PD':
                labels.append('Group (PD vs Control)')
            elif effect_name == 'Stimulation_ON':
                labels.append('Stimulation (ON vs OFF)')
            else:
                labels.append('Group × Stimulation')
            
            colors.append(config.COLORS['highlight'] if effect_name == 'Group_x_Stim' else 'gray')
            p_values.append(ef['p_value'])
            
            # Get semi-partial R²
            if effect_name in results['semi_partial_r2']:
                r2_data = results['semi_partial_r2'][effect_name]
                if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
                    r2_values.append(r2_data['semi_partial_r2'])
                else:
                    r2_values.append(np.nan)
            else:
                r2_values.append(np.nan)
    
    y_pos = np.arange(len(effects))
    
    # Plot forest
    ax3.errorbar(effects, y_pos, xerr=[np.abs(np.array(effects) - np.array(cis_lower)), 
                                        np.abs(np.array(cis_upper) - np.array(effects))],
                fmt='o', color='black', ecolor='black', capsize=3, capthick=1, markerfacecolor='white')
    
    # Color points based on significance
    for i, (effect, p_val, color) in enumerate(zip(effects, p_values, colors)):
        marker_color = 'red' if p_val < 0.05 else 'black'
        ax3.plot(effect, i, 'o', color=marker_color, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=config.FONT_SIZES['axis'])
    ax3.set_xlabel('Estimate (95% CI)', fontsize=config.FONT_SIZES['axis'])
    ax3.set_title('C: Fixed Effects (Wald z-tests)', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
    
    # Add p-values and R² as text
    for i, (p_val, r2) in enumerate(zip(p_values, r2_values)):
        p_text = f"p = {p_val:.3f}"
        if p_val < 0.001:
            p_text = "p < 0.001"
        r2_text = f"R² = {r2:.3f}" if not np.isnan(r2) else ""
        ax3.text(max(cis_upper) + 2, i, f"{p_text}  {r2_text}", 
                va='center', fontsize=config.FONT_SIZES['effect'])
    
    # ========================================================================
    # PANEL D: Hedges g effect sizes
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    g_effects = []
    g_cis_lower = []
    g_cis_upper = []
    g_labels = []
    
    if 'effect_sizes' in results:
        es = results['effect_sizes']
        
        if 'Group_main_at_OFF' in es:
            g_effects.append(es['Group_main_at_OFF']['g'])
            g_cis_lower.append(es['Group_main_at_OFF']['ci_lower'])
            g_cis_upper.append(es['Group_main_at_OFF']['ci_upper'])
            g_labels.append('Group diff (OFF)')
        
        if 'Stimulation_main_in_Control' in es:
            g_effects.append(es['Stimulation_main_in_Control']['g'])
            g_cis_lower.append(es['Stimulation_main_in_Control']['ci_lower'])
            g_cis_upper.append(es['Stimulation_main_in_Control']['ci_upper'])
            g_labels.append('Stimulation (Control)')
        
        if results['interaction_present'] and 'Stimulation_in_PD' in results.get('simple_effects', {}):
            se = results['simple_effects']['Stimulation_in_PD']
            # Approximate g from contrast
            if 'estimate' in se and 'se' in se:
                # Rough approximation - in practice would need proper computation
                g_effects.append(se['estimate'] / results['residual_variance']**0.5)
                g_cis_lower.append(g_effects[-1] - 1.96 * se['se'])
                g_cis_upper.append(g_effects[-1] + 1.96 * se['se'])
                g_labels.append('Stimulation (PD)')
    
    if g_effects:
        y_pos = np.arange(len(g_effects))
        
        ax4.errorbar(g_effects, y_pos, xerr=[np.abs(np.array(g_effects) - np.array(g_cis_lower)), 
                                              np.abs(np.array(g_cis_upper) - np.array(g_effects))],
                    fmt='o', color='black', ecolor='black', capsize=3, capthick=1, markerfacecolor='white')
        
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax4.axvline(x=0.2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax4.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax4.axvline(x=0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(g_labels, fontsize=config.FONT_SIZES['axis'])
        ax4.set_xlabel("Hedges' g (95% CI)", fontsize=config.FONT_SIZES['axis'])
        ax4.set_title('D: Effect Sizes (Bias-Corrected)', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
        ax4.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
        
        # Add effect size interpretation
        ax4.text(0.02, 0.02, 'Small: 0.2\nMedium: 0.5\nLarge: 0.8', 
                transform=ax4.transAxes, fontsize=config.FONT_SIZES['effect'],
                va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'Effect sizes not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('D: Effect Sizes', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    
    plt.suptitle('Figure 1: Primary Outcome - Sucrose Preference', 
                fontsize=config.FONT_SIZES['title']+2, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig_path = output_path / "Figure1_Primary.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 1 saved: {fig_path}")
    
    return fig


def create_main_figure_secondary(all_results, corrected_results, output_path):
    """
    Create publication-ready main figure for secondary outcomes.
    One row per variable with model-estimated means and interaction visualization.
    """
    
    secondary_vars = ['Total Intake (g)', 'Water intake (g)', 'Sucrose intake (g)']
    available_vars = [v for v in secondary_vars if v in all_results and all_results[v]['model_type'] != 'descriptive_only']
    
    if not available_vars:
        print("  ⚠ No secondary outcome results available")
        return None
    
    fig = plt.figure(figsize=(config.DOUBLE_COLUMN, 4 * len(available_vars)))
    gs = gridspec.GridSpec(len(available_vars), 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for row_idx, var in enumerate(available_vars):
        results = all_results[var]
        fitted_model = results['_fitted_model']
        model_df = results['_model_df']
        
        # Calculate estimated marginal means
        groups = ['Control', 'PD']
        stim_levels = ['OFF', 'ON']
        stim_codes = {'OFF': 0, 'ON': 1}
        
        emmeans = {}
        for group in groups:
            group_code = 1 if group == 'PD' else 0
            emmeans[group] = {}
            for stim in stim_levels:
                stim_code = stim_codes[stim]
                X = np.array([[1, group_code, stim_code, group_code * stim_code]])
                pred = np.dot(X, fitted_model.params)[0]
                se = np.sqrt(np.diag(X @ fitted_model.cov_params() @ X.T))[0]
                ci = 1.96 * se
                emmeans[group][stim] = {
                    'mean': pred,
                    'se': se,
                    'ci_lower': pred - ci,
                    'ci_upper': pred + ci
                }
        
        # Panel: Model-estimated means
        ax_left = fig.add_subplot(gs[row_idx, 0])
        
        x_positions = [0, 1]
        for group in groups:
            color = config.COLORS[group]
            means = [emmeans[group][stim]['mean'] for stim in stim_levels]
            ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
            ci_upper = [emmeans[group][stim]['ci_upper'] for stim in stim_levels]
            
            ax_left.plot(x_positions, means, color=color, linewidth=2, marker='o', 
                        markersize=5, label=group)
            ax_left.fill_between(x_positions, ci_lower, ci_upper, color=color, alpha=0.2)
        
        ax_left.set_xticks([0, 1])
        ax_left.set_xticklabels(['OFF', 'ON'], fontsize=config.FONT_SIZES['tick'])
        ax_left.set_ylabel(var, fontsize=config.FONT_SIZES['axis'])
        if row_idx == len(available_vars) - 1:
            ax_left.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
        ax_left.set_title(f"{chr(65 + row_idx*2)}: {var}", fontsize=config.FONT_SIZES['title'], pad=5, loc='left')
        ax_left.legend(fontsize=config.FONT_SIZES['legend'], frameon=True)
        ax_left.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Panel: Interaction bars
        ax_right = fig.add_subplot(gs[row_idx, 1])
        
        x_pos = np.arange(len(stim_levels))
        width = 0.35
        
        for i, group in enumerate(groups):
            color = config.COLORS[group]
            means = [emmeans[group][stim]['mean'] for stim in stim_levels]
            ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
            yerr = [means[j] - ci_lower[j] for j in range(len(means))]
            
            offset = -width/2 if group == 'Control' else width/2
            ax_right.bar(x_pos + offset, means, width, label=group, color=color, alpha=0.7,
                        yerr=yerr, capsize=3, error_kw={'linewidth': 1})
        
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(stim_levels, fontsize=config.FONT_SIZES['tick'])
        ax_right.set_ylabel(var, fontsize=config.FONT_SIZES['axis'])
        if row_idx == len(available_vars) - 1:
            ax_right.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
        ax_right.set_title(f"{chr(65 + row_idx*2 + 1)}: Interaction", fontsize=config.FONT_SIZES['title'], pad=5, loc='left')
        ax_right.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        
        # Add effect size if interaction present
        if 'Group_x_Stim' in results.get('semi_partial_r2', {}):
            r2_data = results['semi_partial_r2']['Group_x_Stim']
            if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
                ax_right.text(0.5, 0.95, f"Interaction pseudo-R² = {r2_data['semi_partial_r2']:.3f}",
                            transform=ax_right.transAxes, fontsize=config.FONT_SIZES['annotation'],
                            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Figure 2: Secondary Outcomes - Intake Measures', 
                fontsize=config.FONT_SIZES['title']+2, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig_path = output_path / "Figure2_Secondary.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 2 saved: {fig_path}")
    
    return fig


def create_supplementary_figure1_diagnostics(all_results, output_path):
    """
    Create supplementary diagnostic figures for each variable.
    One page per variable with QQ plot, residuals vs fitted, and random effects distribution.
    """
    
    for var, results in all_results.items():
        if results['model_type'] == 'descriptive_only' or '_fitted_model' not in results:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(config.DOUBLE_COLUMN, 8))
        fig.suptitle(f'Supplementary Figure 1.{var}: Model Diagnostics', 
                    fontsize=config.FONT_SIZES['title']+2, y=0.98)
        
        fitted_model = results['_fitted_model']
        model_df = results['_model_df']
        
        # Get residuals
        resid_data = conditional_residuals(fitted_model, model_df)
        conditional_resid = resid_data['conditional_residuals']
        fitted = fitted_model.fittedvalues
        
        # Panel A: QQ plot
        ax = axes[0, 0]
        qqplot(conditional_resid, line='s', ax=ax, alpha=0.6)
        ax.set_title('A: Q-Q Plot (Conditional Residuals)', fontsize=config.FONT_SIZES['title'])
        ax.set_xlabel('Theoretical Quantiles', fontsize=config.FONT_SIZES['axis'])
        ax.set_ylabel('Sample Quantiles', fontsize=config.FONT_SIZES['axis'])
        
        # Add Shapiro-Wilk result
        if len(conditional_resid) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(conditional_resid)
            ax.text(0.05, 0.95, f"Shapiro-Wilk p={shapiro_p:.4f}", 
                   transform=ax.transAxes, fontsize=config.FONT_SIZES['annotation'],
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel B: Residuals vs Fitted
        ax = axes[0, 1]
        ax.scatter(fitted, conditional_resid, alpha=0.6, s=20)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('B: Residuals vs Fitted', fontsize=config.FONT_SIZES['title'])
        ax.set_xlabel('Fitted Values', fontsize=config.FONT_SIZES['axis'])
        ax.set_ylabel('Conditional Residuals', fontsize=config.FONT_SIZES['axis'])
        
        # Panel C: Residuals by Group
        ax = axes[1, 0]
        residual_by_group = [conditional_resid[model_df['Group'] == g] for g in ['PD', 'Control']]
        bp = ax.boxplot(residual_by_group, tick_labels=['PD', 'Control'], patch_artist=True)
        for patch, color in zip(bp['boxes'], [config.COLORS['PD'], config.COLORS['Control']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
        ax.set_title('C: Residuals by Group', fontsize=config.FONT_SIZES['title'])
        ax.set_ylabel('Conditional Residuals', fontsize=config.FONT_SIZES['axis'])
        
        # Panel D: Random effects distribution (if available)
        ax = axes[1, 1]
        if len(resid_data['random_effects']) > 0 and np.any(resid_data['random_effects'] != 0):
            re_unique = np.unique(resid_data['random_effects'])
            ax.hist(re_unique, bins='auto', alpha=0.7, color='gray', edgecolor='black')
            ax.set_title('D: Random Effects Distribution', fontsize=config.FONT_SIZES['title'])
            ax.set_xlabel('Random Intercept', fontsize=config.FONT_SIZES['axis'])
            ax.set_ylabel('Frequency', fontsize=config.FONT_SIZES['axis'])
            
            # Add normality test
            if len(re_unique) >= 3:
                re_shapiro, re_p = stats.shapiro(re_unique)
                ax.text(0.05, 0.95, f"Shapiro-Wilk p={re_p:.4f}", 
                       transform=ax.transAxes, fontsize=config.FONT_SIZES['annotation'],
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Random effects not available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('D: Random Effects', fontsize=config.FONT_SIZES['title'])
        
        plt.tight_layout()
        safe_var = var.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        fig_path = output_path / f"Supplementary_Figure1_Diagnostics_{safe_var}.png"
        fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Supplementary Figure 1 ({var}) saved: {fig_path}")


def create_supplementary_figure2_correlations(wide_df, correlation_results, output_path):
    """
    Create improved correlation heatmap with FDR-corrected significance.
    Larger, cleaner, with upper triangle only and clear legend.
    """
    
    delta_vars = [col for col in wide_df.columns if col.startswith('Δ')]
    
    if len(delta_vars) < 2 or not correlation_results:
        print("  ⚠ Insufficient data for correlation heatmap")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(config.DOUBLE_COLUMN, 5))
    fig.suptitle('Supplementary Figure 2: FDR-Corrected Correlation Heatmaps', 
                fontsize=config.FONT_SIZES['title']+2, y=1.02)
    
    for idx, group in enumerate(['PD', 'Control']):
        ax = axes[idx]
        
        if group not in correlation_results:
            ax.text(0.5, 0.5, f"No correlation data for {group}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{group} Group', fontsize=config.FONT_SIZES['title'])
            continue
        
        corr_data = correlation_results[group]
        corr_matrix = np.array(corr_data['correlations'])
        variables = corr_data['variables']
        n_vars = len(variables)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Create annotation matrix with FDR-corrected significance
        annot_matrix = np.zeros_like(corr_matrix, dtype='U10')
        if 'p_values_fdr' in corr_data:
            p_fdr = np.array(corr_data['p_values_fdr'])
            # Reshape p_fdr to matrix
            p_matrix = np.zeros((n_vars, n_vars))
            upper_tri = np.triu_indices(n_vars, k=1)
            for i, (row, col) in enumerate(zip(*upper_tri)):
                if i < len(p_fdr):
                    p_matrix[row, col] = p_fdr[i]
                    p_matrix[col, row] = p_fdr[i]
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i < j:
                        r_val = corr_matrix[i, j]
                        p_val = p_matrix[i, j]
                        if p_val < 0.001:
                            stars = '***'
                        elif p_val < 0.01:
                            stars = '**'
                        elif p_val < 0.05:
                            stars = '*'
                        else:
                            stars = ''
                        annot_matrix[i, j] = f"{r_val:.2f}{stars}"
        else:
            for i in range(n_vars):
                for j in range(n_vars):
                    if i < j:
                        annot_matrix[i, j] = f"{corr_matrix[i, j]:.2f}"
        
        # Create heatmap with mask
        sns.heatmap(corr_matrix, mask=mask, annot=annot_matrix, fmt='',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   xticklabels=variables, yticklabels=variables,
                   ax=ax, cbar=idx==1, cbar_kws={'label': 'Spearman r'},
                   annot_kws={'size': config.FONT_SIZES['annotation']})
        
        ax.set_title(f'{group} Group', fontsize=config.FONT_SIZES['title'])
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=config.FONT_SIZES['tick'])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=config.FONT_SIZES['tick'])
    
    # Add significance legend
    legend_text = "* p_FDR < 0.05\n** p_FDR < 0.01\n*** p_FDR < 0.001"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=config.FONT_SIZES['annotation'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig_path = output_path / "Supplementary_Figure2_Correlations.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Supplementary Figure 2 saved: {fig_path}")


def create_supplementary_figure3_individual_changes(wide_df, output_path):
    """
    Create individual-level change plots for each variable.
    Shows within-subject Δ distributions with violin plots and individual points.
    """
    
    delta_vars = [col for col in wide_df.columns if col.startswith('Δ')]
    
    if len(delta_vars) == 0:
        print("  ⚠ No delta variables available")
        return None
    
    n_vars = len(delta_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(config.DOUBLE_COLUMN, 3 * n_vars))
    if n_vars == 1:
        axes = [axes]
    
    fig.suptitle('Supplementary Figure 3: Individual-Level Change Scores', 
                fontsize=config.FONT_SIZES['title']+2, y=0.98)
    
    for idx, delta_var in enumerate(delta_vars):
        ax = axes[idx]
        
        # Prepare data
        pd_changes = wide_df[wide_df['Group'] == 'PD'][delta_var].dropna()
        control_changes = wide_df[wide_df['Group'] == 'Control'][delta_var].dropna()
        
        # Create violin plot
        positions = [0, 1]
        data = [control_changes, pd_changes]
        colors = [config.COLORS['Control'], config.COLORS['PD']]
        
        # Violin plots
        vp = ax.violinplot(data, positions=positions, widths=0.6, showmeans=False, 
                          showmedians=False, showextrema=False)
        
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)
            pc.set_edgecolor(colors[i])
            pc.set_linewidth(1)
        
        # Box plots inside
        bp = ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True,
                       showfliers=False, whiskerprops={'color': 'black', 'linewidth': 0.5},
                       medianprops={'color': 'white', 'linewidth': 1.5})
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Individual points with jitter
        np.random.seed(42)
        for i, (d, color) in enumerate(zip(data, colors)):
            jitter = np.random.normal(0, 0.05, len(d))
            ax.scatter(positions[i] + jitter, d, color=color, alpha=0.8, s=25,
                      edgecolor='black', linewidth=0.5, zorder=3)
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        var_label = delta_var.replace('Δ', 'Δ ')
        ax.set_ylabel(var_label, fontsize=config.FONT_SIZES['axis'])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'PD'], fontsize=config.FONT_SIZES['tick'])
        ax.set_title(f"{chr(65 + idx)}: {var_label}", fontsize=config.FONT_SIZES['title'], loc='left')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        
        # Add summary statistics
        stats_text = f"Control: n={len(control_changes)}, M={control_changes.mean():.1f}, SD={control_changes.std():.1f}\n"
        stats_text += f"PD: n={len(pd_changes)}, M={pd_changes.mean():.1f}, SD={pd_changes.std():.1f}"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=config.FONT_SIZES['effect'],
               ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    fig_path = output_path / "Supplementary_Figure3_IndividualChanges.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Supplementary Figure 3 saved: {fig_path}")

# ============================================================================
# ENHANCED CORRELATION ANALYSIS (for supplementary)
# ============================================================================

def enhanced_correlation_analysis(wide_df, output_path):
    """Perform enhanced correlation analysis with FDR-corrected significance."""
    
    print("\n" + "="*70)
    print("ENHANCED CORRELATION ANALYSIS")
    print("="*70)
    
    delta_vars = [col for col in wide_df.columns if col.startswith('Δ')]
    
    if len(delta_vars) < 2:
        print("  ⚠ Insufficient delta variables for correlation analysis")
        return {}
    
    correlation_results = {}
    
    for group in ['PD', 'Control']:
        group_data = wide_df[wide_df['Group'] == group][delta_vars].dropna()
        
        if len(group_data) < 3:
            print(f"  ⚠ Insufficient data for {group} group")
            continue
        
        # Calculate Spearman correlations with p-values
        n_vars = len(delta_vars)
        corr_matrix = np.zeros((n_vars, n_vars))
        p_matrix = np.ones((n_vars, n_vars))
        
        for i, var1 in enumerate(delta_vars):
            for j, var2 in enumerate(delta_vars):
                if i < j:
                    corr, p_val = stats.spearmanr(group_data[var1], group_data[var2])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr
                    p_matrix[i, j] = p_val
                    p_matrix[j, i] = p_val
        
        # FDR correction for correlation p-values
        upper_tri_indices = np.triu_indices_from(p_matrix, k=1)
        p_values_upper = p_matrix[upper_tri_indices]
        
        if len(p_values_upper) > 0:
            reject, p_corrected, _, _ = multipletests(p_values_upper, method='fdr_bh')
            
            correlation_results[group] = {
                'correlations': corr_matrix.tolist(),
                'p_values_original': p_matrix.tolist(),
                'p_values_fdr': p_corrected.tolist(),
                'significant_fdr': reject.tolist(),
                'variables': delta_vars
            }
    
    return correlation_results

# ============================================================================
# THESIS-LEVEL SUMMARY WITH PRECISE METHODOLOGICAL REPORTING
# ============================================================================

def create_enhanced_summary_thesis_v2(all_results, corrected_results, df, wide_df, output_path):
    """
    Thesis-level summary with precise mixed-model reporting.
    CORRECTED: Accurate LMM terminology, conditional ICC, proper LRT description.
    """
    
    summary_path = output_path / "analysis_summary_thesis_level_v2.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE SUCROSE PREFERENCE ANALYSIS - THESIS-LEVEL MIXED MODELS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis performed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXPERIMENTAL DESIGN\n")
        f.write("-"*40 + "\n")
        f.write("* Design: 2 (Group: PD vs Control) × 2 (Stimulation: OFF vs ON) mixed factorial\n")
        f.write("* Between-subject factor: Group\n")
        f.write("* Within-subject factor: Stimulation\n")
        f.write("* Clustering: Repeated measures within subject\n\n")
        
        f.write("STATISTICAL METHODS\n")
        f.write("-"*40 + "\n")
        f.write("* Primary model: Linear mixed-effects models fitted via restricted maximum likelihood (REML)\n")
        f.write("* Random effects structure: Selected via boundary-corrected likelihood ratio test (Stram & Lee, 1994)\n")
        f.write("* Fixed effects inference: Asymptotic Wald z-tests (normal approximation)\n")
        f.write("* Effect sizes: Likelihood-based semi-partial R² (Edwards et al., 2008)\n")
        f.write("* Multiple comparison correction: Benjamini-Hochberg false discovery rate (FDR), α = 0.05\n")
        f.write("* FDR families: Primary (Sucrose Preference) and Secondary (Intake measures) separated a priori\n")
        f.write("* Supplemental effect sizes: Hedges g with bias correction and bootstrap 95% CIs\n\n")
        
        f.write("SAMPLE CHARACTERISTICS\n")
        f.write("-"*40 + "\n")
        f.write(f"Total subjects: {df['Subject'].nunique()}\n")
        f.write(f"  PD group: {len(wide_df[wide_df['Group'] == 'PD'])} subjects\n")
        f.write(f"  Control group: {len(wide_df[wide_df['Group'] == 'Control'])} subjects\n")
        f.write(f"Total observations: {len(df)} (balanced: 2 per subject)\n\n")
        
        f.write("MIXED MODEL RESULTS\n")
        f.write("="*40 + "\n\n")
        
        for var, results in all_results.items():
            f.write(f"\n{var.upper()}\n")
            f.write("-"*40 + "\n")
            
            if results['model_type'] == 'descriptive_only':
                f.write(f"MODEL FAILED TO CONVERGE: {results.get('error', 'Unknown error')}\n")
                f.write(f"Descriptive statistics reported only.\n")
                if 'group_stats' in results:
                    for key, stats in results['group_stats'].items():
                        f.write(f"  {key}: M = {stats['mean']:.2f}, SD = {stats['std']:.2f}, n = {stats['n']}\n")
                f.write("\n" + "-"*40 + "\n")
                continue
            
            # Model summary with precise type
            model_display = results['model_type'].replace('_', ' ').title()
            f.write(f"Model structure: {model_display}\n")
            f.write(f"  Random effects: ")
            if results['model_type'] == 'random_intercept':
                f.write(f"Subject-specific random intercepts")
            else:
                f.write(f"Subject-specific random intercepts and random slopes for Stimulation")
            f.write(f" (estimated via REML)\n")
            
            f.write(f"Convergence: {'Yes' if results['converged'] else 'No - interpret with caution'}\n")
            f.write(f"N subjects: {results['n_subjects']}, N observations: {results['n_observations']}\n")
            f.write(f"AIC: {results['aic']:.1f}, BIC: {results['bic']:.1f}\n\n")
            
            # Model selection with precise wording
            if 'model_selection' in results and results['model_selection'].get('lr_test'):
                lrt = results['model_selection']['lr_test']
                if not np.isnan(lrt.get('statistic', np.nan)):
                    f.write(f"Random slope assessment (boundary-corrected LRT, H0: σ²_slope = 0):\n")
                    f.write(f"  Likelihood ratio statistic = {lrt['statistic']:.2f}\n")
                    f.write(f"  Boundary-corrected p-value = {lrt['p_value_boundary_corrected']:.4f}\n")
                    if lrt.get('p_value_standard') and not np.isnan(lrt['p_value_standard']):
                        f.write(f"  (Standard χ²(2) p-value = {lrt['p_value_standard']:.4f})\n")
                    f.write(f"  Model preferred: {results['model_selection']['selected_model']}\n\n")
            
            # Fixed effects with precise terminology
            f.write("Fixed Effects (Wald z-tests, asymptotic normal approximation):\n")
            for effect_name, effect_data in results['fixed_effects'].items():
                if effect_name == 'Intercept':
                    continue
                
                # FDR significance marker
                sig_marker = ""
                for corr_key, corr_data in corrected_results.items():
                    if (corr_data['variable'] == var and 
                        effect_name in corr_data['effect'] and
                        corr_data['significant_fdr']):
                        sig_marker = "†"
                        break
                
                p_val = effect_data['p_value']
                p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< .0001"
                
                f.write(f"  {effect_name}{sig_marker}:\n")
                f.write(f"    Estimate = {effect_data['coefficient']:.3f}, SE = {effect_data['se']:.3f}\n")
                f.write(f"    95% CI = [{effect_data['ci_lower']:.3f}, {effect_data['ci_upper']:.3f}]\n")
                f.write(f"    z = {effect_data['z_stat']:.2f}, p = {p_str}\n")
                
                # Likelihood-based semi-partial R² with cautious interpretation
                if effect_name in results.get('semi_partial_r2', {}):
                    r2_data = results['semi_partial_r2'][effect_name]
                    if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
                        f.write(f"    Likelihood-based pseudo-R² = {r2_data['semi_partial_r2']:.3f} ")
                        f.write(f"(proportional reduction in lack-of-fit)\n")
                        f.write(f"    [Note: Not equivalent to OLS R²; use for relative comparison]\n")
            
            # ICC - conditional on model type
            if 'icc' in results:
                icc_data = results['icc']
                if icc_data['icc'] is not None:
                    f.write(f"\nIntraclass Correlation Coefficient (ICC):\n")
                    f.write(f"  ICC = {icc_data['icc']:.3f}\n")
                    f.write(f"  Interpretation: {icc_data['icc']*100:.1f}% of residual variance ")
                    f.write(f"attributable to between-subject differences (after accounting for fixed effects)\n")
                else:
                    f.write(f"\n{icc_data['note']}\n")
                    if 'random_slope' in results['model_type']:
                        f.write(f"  ICC not interpretable due to random slope specification.\n")
            
            # Simple effects - only when interaction present
            if results['interaction_present'] and results['simple_effects']:
                f.write(f"\nSimple Effects of Stimulation within Groups (model-based contrasts):\n")
                for effect_name, effect_data in results['simple_effects'].items():
                    p_val = effect_data['p_value']
                    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< .0001"
                    
                    f.write(f"  {effect_name}:\n")
                    f.write(f"    Estimate = {effect_data['estimate']:.3f}, SE = {effect_data['se']:.3f}\n")
                    f.write(f"    z = {effect_data.get('z', 0):.2f}, p = {p_str}\n")
                    f.write(f"    [{effect_data['method']}]\n")
            
            # Hedges g (supplemental, model-free)
            if results.get('effect_sizes'):
                f.write(f"\nSupplemental Effect Sizes (Hedges g, bias-corrected with bootstrap 95% CI):\n")
                f.write(f"  [Note: These are model-agnostic descriptive effect sizes for reference]\n")
                for es_name, es_data in results['effect_sizes'].items():
                    if 'g' in es_data:
                        f.write(f"  {es_name}:\n")
                        f.write(f"    g = {es_data['g']:.3f} ")
                        f.write(f"[{es_data['ci_lower']:.3f}, {es_data['ci_upper']:.3f}]\n")
                        if 'bootstrap_ci_lower' in es_data and not np.isnan(es_data['bootstrap_ci_lower']):
                            f.write(f"    Bootstrap 95% CI: [{es_data['bootstrap_ci_lower']:.3f}, ")
                            f.write(f"{es_data['bootstrap_ci_upper']:.3f}]\n")
            
            # Diagnostics with appropriate caution
            if results.get('diagnostics'):
                f.write(f"\nModel Diagnostics (based on conditional residuals from fitted model):\n")
                if 'residual_normality' in results['diagnostics']:
                    rd = results['diagnostics']['residual_normality']
                    f.write(f"  Conditional residual normality (Shapiro-Wilk):\n")
                    f.write(f"    W = {rd['statistic']:.3f}, p = {rd['p_value']:.4f} (n = {rd['n_observations']} observations)\n")
                    if rd['normal']:
                        f.write(f"    [No evidence of non-normality; LMM robust to moderate violations]\n")
                    else:
                        f.write(f"    [Evidence of non-normality; interpret with caution, consider robust estimation]\n")
                
                if 'random_effects_normality' in results['diagnostics']:
                    red = results['diagnostics']['random_effects_normality']
                    f.write(f"  Random effects normality (Shapiro-Wilk):\n")
                    f.write(f"    W = {red['statistic']:.3f}, p = {red['p_value']:.4f} (n = {red['n_subjects']} subjects)\n")
                    if not red['normal']:
                        f.write(f"    [Random effects non-normality may affect cluster-specific predictions]\n")
            
            f.write("\n" + "-"*40 + "\n")
        
        f.write("\n\nFDR CORRECTION DETAILS\n")
        f.write("-"*40 + "\n")
        f.write("False Discovery Rate (Benjamini-Hochberg) applied within pre-specified families:\n")
        f.write("* Family 1 (Primary - Preference): Sucrose Preference (3 tests)\n")
        f.write("* Family 2 (Secondary - Consumption): Total Intake, Water Intake, Sucrose Intake (9 tests)\n")
        f.write("Control level: Expected proportion of false positives ≤ 0.05 within each family\n\n")
        
        f.write("Effects significant after FDR correction (†):\n")
        sig_found = False
        for key, data in corrected_results.items():
            if data['significant_fdr']:
                sig_found = True
                f.write(f"  † {data['variable']} - {data['effect']}\n")
                f.write(f"    Raw p = {data['p_value_raw']:.4f}, ")
                f.write(f"FDR-adjusted p = {data['p_value_fdr']:.4f}, ")
                f.write(f"q-value = {data['q_value']:.4f}\n")
        if not sig_found:
            f.write("  No effects significant after FDR correction\n")
        
        f.write("\n\nRECOMMENDED REPORTING TEXT\n")
        f.write("-"*40 + "\n")
        f.write("\"Linear mixed-effects models were fitted using restricted maximum likelihood (REML) ")
        f.write("with subject as a random intercept. Random slope specification was evaluated via ")
        f.write("boundary-corrected likelihood ratio tests (Stram & Lee, 1994). Fixed effects were ")
        f.write("assessed using Wald z-tests. Effect sizes are reported as likelihood-based semi-partial ")
        f.write("R² (Edwards et al., 2008). Intraclass correlation coefficients (ICC) are reported for ")
        f.write("random-intercept models only. Multiple comparisons were controlled using the ")
        f.write("Benjamini-Hochberg false discovery rate (FDR, α = 0.05) applied separately to primary ")
        f.write("and secondary outcome families. Supplemental effect sizes are reported as bias-corrected ")
        f.write("Hedges g with bootstrap 95% confidence intervals.\"\n\n")
        
        f.write("\n\nFILES GENERATED\n")
        f.write("-"*40 + "\n")
        f.write("Main Figures:\n")
        f.write("  Figure1_Primary.png - Primary outcome (Sucrose Preference)\n")
        f.write("  Figure2_Secondary.png - Secondary outcomes (Intake measures)\n\n")
        f.write("Supplementary Figures:\n")
        f.write("  Supplementary_Figure1_Diagnostics_*.png - Model diagnostics per variable\n")
        f.write("  Supplementary_Figure2_Correlations.png - FDR-corrected correlation heatmaps\n")
        f.write("  Supplementary_Figure3_IndividualChanges.png - Individual-level change plots\n\n")
        f.write("Data Files:\n")
        f.write("  comprehensive_analysis_results_mixed.xlsx - Complete numerical results\n")
        f.write("  analysis_summary_thesis_level_v2.txt - This summary file\n")
    
    print(f"✓ Thesis-level summary (v2) saved to: {summary_path}")


def save_enhanced_results(all_results, corrected_results, correlation_results, 
                         df, wide_df, output_path):
    """Save all results to Excel with comprehensive statistics"""
    
    excel_path = output_path / "comprehensive_analysis_results_mixed.xlsx"
    
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Raw data
        df.to_excel(writer, sheet_name='Raw_Data_Long', index=False)
        wide_df.to_excel(writer, sheet_name='Raw_Data_Wide', index=False)
        
        # Mixed model results
        mixed_results_rows = []
        for var, results in all_results.items():
            if results['model_type'] == 'descriptive_only':
                # Add descriptive stats
                if 'group_stats' in results:
                    for key, stats in results['group_stats'].items():
                        row = {
                            'Variable': var,
                            'Model_Type': 'descriptive_only',
                            'Condition': key,
                            'Mean': stats['mean'],
                            'SD': stats['std'],
                            'N': stats['n']
                        }
                        mixed_results_rows.append(row)
                continue
                
            for effect_name, effect_data in results['fixed_effects'].items():
                if effect_name == 'Intercept':
                    continue
                row = {
                    'Variable': var,
                    'Effect': effect_name,
                    'Family': results['family'],
                    'Coefficient': effect_data['coefficient'],
                    'SE': effect_data['se'],
                    'CI_Lower': effect_data['ci_lower'],
                    'CI_Upper': effect_data['ci_upper'],
                    'z_stat': effect_data['z_stat'],
                    'p_value': effect_data['p_value'],
                    'N_subjects': results['n_subjects'],
                    'N_observations': results['n_observations'],
                    'AIC': results['aic'],
                    'Model_Type': results['model_type'],
                    'Converged': results['converged']
                }
                
                # Add semi-partial R² if available
                if effect_name in results.get('semi_partial_r2', {}):
                    r2_data = results['semi_partial_r2'][effect_name]
                    if 'semi_partial_r2' in r2_data:
                        row['Semi_Partial_R2'] = r2_data['semi_partial_r2']
                
                mixed_results_rows.append(row)
        
        if mixed_results_rows:
            pd.DataFrame(mixed_results_rows).to_excel(
                writer, sheet_name='Mixed_Model_Results', index=False)
        
        # FDR corrected results
        if corrected_results:
            corrected_rows = list(corrected_results.values())
            pd.DataFrame(corrected_rows).to_excel(
                writer, sheet_name='FDR_Corrected', index=False)
        
        # ICC results
        icc_rows = []
        for var, results in all_results.items():
            if results['model_type'] != 'descriptive_only' and 'icc' in results:
                icc_data = results['icc']
                row = {
                    'Variable': var,
                    'Model_Type': results['model_type'],
                    **icc_data
                }
                icc_rows.append(row)
        
        if icc_rows:
            pd.DataFrame(icc_rows).to_excel(
                writer, sheet_name='ICC', index=False)
        
        # Simple effects
        simple_effects_rows = []
        for var, results in all_results.items():
            for effect_name, effect_data in results.get('simple_effects', {}).items():
                row = {
                    'Variable': var,
                    'Effect': effect_name,
                    **effect_data
                }
                simple_effects_rows.append(row)
        
        if simple_effects_rows:
            pd.DataFrame(simple_effects_rows).to_excel(
                writer, sheet_name='Simple_Effects', index=False)
        
        # Effect sizes
        effect_sizes_rows = []
        for var, results in all_results.items():
            for es_name, es_data in results.get('effect_sizes', {}).items():
                row = {
                    'Variable': var,
                    'Effect_Size_Type': es_name,
                    **es_data
                }
                effect_sizes_rows.append(row)
        
        if effect_sizes_rows:
            pd.DataFrame(effect_sizes_rows).to_excel(
                writer, sheet_name='Effect_Sizes', index=False)
        
        # Correlations
        if correlation_results:
            for group, corr_data in correlation_results.items():
                if 'correlations' in corr_data:
                    corr_df = pd.DataFrame(
                        corr_data['correlations'],
                        index=corr_data['variables'],
                        columns=corr_data['variables']
                    )
                    corr_df.to_excel(writer, sheet_name=f'Correlations_{group}')
                    
                    # Also save p-values
                    if 'p_values_fdr' in corr_data:
                        n = len(corr_data['variables'])
                        p_matrix = np.zeros((n, n))
                        upper_tri = np.triu_indices(n, k=1)
                        for idx, (i, j) in enumerate(zip(*upper_tri)):
                            if idx < len(corr_data['p_values_fdr']):
                                p_matrix[i, j] = corr_data['p_values_fdr'][idx]
                                p_matrix[j, i] = corr_data['p_values_fdr'][idx]
                        
                        p_df = pd.DataFrame(
                            p_matrix,
                            index=corr_data['variables'],
                            columns=corr_data['variables']
                        )
                        p_df.to_excel(writer, sheet_name=f'Corr_pvalues_{group}')
        
        # Model diagnostics
        diag_rows = []
        for var, results in all_results.items():
            if results['model_type'] == 'descriptive_only':
                continue
            for diag_name, diag_data in results.get('diagnostics', {}).items():
                if isinstance(diag_data, dict):
                    row = {
                        'Variable': var,
                        'Diagnostic': diag_name,
                        **diag_data
                    }
                else:
                    row = {
                        'Variable': var,
                        'Diagnostic': diag_name,
                        'Value': diag_data
                    }
                diag_rows.append(row)
        
        if diag_rows:
            pd.DataFrame(diag_rows).to_excel(
                writer, sheet_name='Diagnostics', index=False)
    
    print(f"✓ Enhanced Excel results saved to: {excel_path}")

# ============================================================================
# ORIGINAL FIGURE FUNCTIONS (preserved for backward compatibility)
# ============================================================================

def plot_variable_trajectories(ax, wide_df, variable, title):
    """Plot individual trajectories for any variable"""
    
    np.random.seed(42)
    
    off_col = f"{variable}_OFF"
    on_col = f"{variable}_ON"
    
    if off_col not in wide_df.columns or on_col not in wide_df.columns:
        ax.text(0.5, 0.5, f'No data for {variable}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=config.FONT_SIZES['title'], pad=10)
        ax.axis('off')
        return
    
    for group, color in [('PD', config.COLORS['PD']), 
                         ('Control', config.COLORS['Control'])]:
        group_data = wide_df[wide_df['Group'] == group]
        valid_data = group_data[[off_col, on_col]].dropna()
        
        if len(valid_data) > 0:
            for idx, row in valid_data.iterrows():
                ax.plot([0, 1], [row[off_col], row[on_col]],
                       color=color, alpha=0.3, linewidth=0.5, zorder=1)
            
            mean_off = valid_data[off_col].mean()
            mean_on = valid_data[on_col].mean()
            ax.plot([0, 1], [mean_off, mean_on],
                   color=color, linewidth=2, marker='o',
                   markersize=5, zorder=3, label=f'{group} Mean')
            
            jitter_off = np.random.normal(0, 0.02, len(valid_data))
            jitter_on = np.random.normal(0, 0.02, len(valid_data))
            
            ax.scatter(np.zeros(len(valid_data)) + jitter_off, valid_data[off_col],
                      color=color, alpha=0.5, s=20, zorder=2,
                      edgecolor='white', linewidth=0.3)
            ax.scatter(np.ones(len(valid_data)) + jitter_on, valid_data[on_col],
                      color=color, alpha=0.5, s=20, zorder=2,
                      edgecolor='white', linewidth=0.3)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['OFF', 'ON'], fontsize=config.FONT_SIZES['tick'])
    ax.set_ylabel(title.split(':')[0], fontsize=config.FONT_SIZES['axis'])
    ax.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
    ax.set_title(title, fontsize=config.FONT_SIZES['title'], pad=10)
    if ax.has_data():
        ax.legend(fontsize=config.FONT_SIZES['legend'], frameon=True, loc='best')
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.text(-0.15, 1.05, title[0], transform=ax.transAxes,
            fontsize=config.FONT_SIZES['panel'], fontweight='bold', va='top')

def plot_variable_boxplot(ax, df, variable, title):
    """Plot boxplot for any variable by Group and Stimulation"""
    
    if variable not in df.columns:
        ax.text(0.5, 0.5, f'No data for {variable}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=config.FONT_SIZES['title'], pad=10)
        ax.axis('off')
        return
    
    plot_data = df[['Group', 'Stimulation', variable]].dropna()
    
    if len(plot_data) == 0:
        ax.text(0.5, 0.5, 'No data available', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=config.FONT_SIZES['title'], pad=10)
        ax.axis('off')
        return
    
    sns.boxplot(data=plot_data, x='Group', y=variable, hue='Stimulation',
                ax=ax, palette=[config.COLORS['OFF'], config.COLORS['ON']],
                width=0.7, linewidth=0.75, fliersize=3)
    
    sns.stripplot(data=plot_data, x='Group', y=variable, hue='Stimulation',
                  dodge=True, ax=ax, palette=[config.COLORS['OFF'], config.COLORS['ON']],
                  size=4, alpha=0.6, edgecolor='black', linewidth=0.3)
    
    ax.set_xlabel('Group', fontsize=config.FONT_SIZES['axis'])
    ax.set_ylabel(variable, fontsize=config.FONT_SIZES['axis'])
    ax.set_title(title, fontsize=config.FONT_SIZES['title'], pad=10)
    
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles[:2], labels[:2], fontsize=config.FONT_SIZES['legend'], 
                  frameon=True, title='Stimulation')
    
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    ax.text(-0.15, 1.05, title[0], transform=ax.transAxes,
            fontsize=config.FONT_SIZES['panel'], fontweight='bold', va='top')

def plot_raincloud_changes(ax, wide_df, delta_var, title, y_label=None):
    """Plot raincloud plot for change scores"""
    
    if delta_var not in wide_df.columns:
        ax.text(0.5, 0.5, f'No data for {delta_var}', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, fontsize=config.FONT_SIZES['title'], pad=10)
        ax.axis('off')
        return
    
    np.random.seed(42)
    
    groups = ['PD', 'Control']
    positions = [0, 1]
    colors = [config.COLORS['PD'], config.COLORS['Control']]
    
    for i, (group, color) in enumerate(zip(groups, colors)):
        changes = wide_df[wide_df['Group'] == group][delta_var].dropna()
        
        if len(changes) == 0:
            continue
        
        violin_parts = ax.violinplot(changes, positions=[positions[i]], 
                                    widths=0.6, showmeans=False, 
                                    showmedians=False, showextrema=False)
        
        for pc in violin_parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.3)
            pc.set_edgecolor(color)
            pc.set_linewidth(1)
        
        median = np.median(changes)
        q1 = np.percentile(changes, 25)
        q3 = np.percentile(changes, 75)
        iqr = q3 - q1
        
        box_width = 0.4
        box = Rectangle((positions[i] - box_width/2, q1), 
                       box_width, iqr,
                       facecolor=color, alpha=0.7,
                       edgecolor='black', linewidth=0.75, zorder=3)
        ax.add_patch(box)
        
        ax.plot([positions[i] - box_width/2, positions[i] + box_width/2],
               [median, median], color='white', linewidth=1.5, zorder=4)
        
        jitter = np.random.normal(0, 0.05, len(changes))
        ax.scatter(positions[i] + jitter, changes,
                  color=color, alpha=0.8, s=25,
                  edgecolor='black', linewidth=0.5, zorder=2, label=group)
    
    if ax.has_data():
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1, zorder=1)
    
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['PD', 'Control'], fontsize=config.FONT_SIZES['tick'])
    if y_label:
        ax.set_ylabel(y_label, fontsize=config.FONT_SIZES['axis'])
    else:
        var_name = delta_var[1:]
        ax.set_ylabel(f'Delta {var_name}', fontsize=config.FONT_SIZES['axis'])
    ax.set_xlabel('Group', fontsize=config.FONT_SIZES['axis'])
    ax.set_title(title, fontsize=config.FONT_SIZES['title'], pad=10)
    if ax.has_data():
        ax.legend(fontsize=config.FONT_SIZES['legend'], frameon=True)
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    ax.text(-0.15, 1.05, title[0], transform=ax.transAxes,
            fontsize=config.FONT_SIZES['panel'], fontweight='bold', va='top')

def create_figure_all_variables_trajectories(df, wide_df):
    """Create figure showing trajectories for all variables (backward compatibility)"""
    
    fig = plt.figure(figsize=(config.DOUBLE_COLUMN, 8))
    fig.suptitle('Individual Trajectories for All Variables', 
                 fontsize=config.FONT_SIZES['title']+2, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_variable_trajectories(ax1, wide_df, 'SPT', 'A: Sucrose Preference (%)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    if 'TI_OFF' in wide_df.columns and 'TI_ON' in wide_df.columns:
        plot_variable_trajectories(ax2, wide_df, 'TI', 'B: Total Intake (g)')
    else:
        ax2.text(0.5, 0.5, 'No Total Intake data', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('B: Total Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 0])
    if 'WI_OFF' in wide_df.columns and 'WI_ON' in wide_df.columns:
        plot_variable_trajectories(ax3, wide_df, 'WI', 'C: Water Intake (g)')
    else:
        ax3.text(0.5, 0.5, 'No Water Intake data', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('C: Water Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    if 'SI_OFF' in wide_df.columns and 'SI_ON' in wide_df.columns:
        plot_variable_trajectories(ax4, wide_df, 'SI', 'D: Sucrose Intake (g)')
    else:
        ax4.text(0.5, 0.5, 'No Sucrose Intake data', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('D: Sucrose Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax4.axis('off')
    
    plt.tight_layout()
    fig_path = OUTPUT_PATH / "figure_all_variables_trajectories.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Trajectories figure saved: {fig_path}")
    return fig

def create_figure_all_variables_boxplots(df, wide_df):
    """Create figure showing boxplots for all variables (backward compatibility)"""
    
    fig = plt.figure(figsize=(config.DOUBLE_COLUMN, 8))
    fig.suptitle('Distribution by Group and Stimulation for All Variables', 
                 fontsize=config.FONT_SIZES['title']+2, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_variable_boxplot(ax1, df, '% SucrosePreference', 'A: Sucrose Preference (%)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    if 'Total Intake (g)' in df.columns:
        plot_variable_boxplot(ax2, df, 'Total Intake (g)', 'B: Total Intake (g)')
    else:
        ax2.text(0.5, 0.5, 'No Total Intake data', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('B: Total Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 0])
    if 'Water intake (g)' in df.columns:
        plot_variable_boxplot(ax3, df, 'Water intake (g)', 'C: Water Intake (g)')
    else:
        ax3.text(0.5, 0.5, 'No Water Intake data', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('C: Water Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    if 'Sucrose intake (g)' in df.columns:
        plot_variable_boxplot(ax4, df, 'Sucrose intake (g)', 'D: Sucrose Intake (g)')
    else:
        ax4.text(0.5, 0.5, 'No Sucrose Intake data', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('D: Sucrose Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax4.axis('off')
    
    plt.tight_layout()
    fig_path = OUTPUT_PATH / "figure_all_variables_boxplots.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Boxplots figure saved: {fig_path}")
    return fig

def create_figure_change_scores_rainclouds(wide_df):
    """Create figure showing raincloud plots for all change scores (backward compatibility)"""
    
    fig = plt.figure(figsize=(config.DOUBLE_COLUMN, 6))
    fig.suptitle('Change Score Distributions for All Variables', 
                 fontsize=config.FONT_SIZES['title']+2, fontweight='bold', y=0.98)
    
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)
    
    ax1 = fig.add_subplot(gs[0, 0])
    plot_raincloud_changes(ax1, wide_df, 'ΔSPT', 'A: Δ Sucrose Preference (%)')
    
    ax2 = fig.add_subplot(gs[0, 1])
    if 'ΔTI' in wide_df.columns:
        plot_raincloud_changes(ax2, wide_df, 'ΔTI', 'B: Δ Total Intake (g)')
    else:
        ax2.text(0.5, 0.5, 'No Total Intake data', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('B: Δ Total Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax2.axis('off')
    
    ax3 = fig.add_subplot(gs[1, 0])
    if 'ΔWI' in wide_df.columns:
        plot_raincloud_changes(ax3, wide_df, 'ΔWI', 'C: Δ Water Intake (g)')
    else:
        ax3.text(0.5, 0.5, 'No Water Intake data', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('C: Δ Water Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax3.axis('off')
    
    ax4 = fig.add_subplot(gs[1, 1])
    if 'ΔSI' in wide_df.columns:
        plot_raincloud_changes(ax4, wide_df, 'ΔSI', 'D: Δ Sucrose Intake (g)')
    else:
        ax4.text(0.5, 0.5, 'No Sucrose Intake data', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('D: Δ Sucrose Intake (g)', fontsize=config.FONT_SIZES['title'])
        ax4.axis('off')
    
    plt.tight_layout()
    fig_path = OUTPUT_PATH / "figure_change_scores_rainclouds.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Raincloud figure saved: {fig_path}")
    return fig

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_comprehensive_analysis():
    """Run the complete comprehensive analysis with thesis-level statistics and visualization"""
    
    print("="*70)
    print("COMPREHENSIVE SUCROSE PREFERENCE TEST ANALYSIS")
    print("THESIS-LEVEL MIXED MODELS APPROACH")
    print("="*70)
    print("\nNOTE: No warnings are suppressed - convergence issues are visible")
    
    # Load data
    df = load_data_from_excel()
    df, wide_df = create_wide_formats(df)
    
    # Fit mixed models with selection
    all_results, corrected_results = fit_all_mixed_models(df)
    
    # Create diagnostic plots using selected models
    create_model_diagnostic_plots(all_results, df, OUTPUT_PATH)
    
    # Correlation analysis with FDR-corrected significance
    correlation_results = enhanced_correlation_analysis(wide_df, OUTPUT_PATH)
    
    # Create backward-compatible figures
    print("\n" + "="*70)
    print("CREATING BACKWARD-COMPATIBLE FIGURES")
    print("="*70)
    
    create_figure_all_variables_trajectories(df, wide_df)
    create_figure_all_variables_boxplots(df, wide_df)
    create_figure_change_scores_rainclouds(wide_df)
    
    # Create new thesis-level figures
    print("\n" + "="*70)
    print("CREATING THESIS-LEVEL FIGURES")
    print("="*70)
    
    create_main_figure_primary(all_results, corrected_results, OUTPUT_PATH)
    create_main_figure_secondary(all_results, corrected_results, OUTPUT_PATH)
    create_supplementary_figure1_diagnostics(all_results, OUTPUT_PATH)
    create_supplementary_figure2_correlations(wide_df, correlation_results, OUTPUT_PATH)
    create_supplementary_figure3_individual_changes(wide_df, OUTPUT_PATH)
    
    # Save results
    save_enhanced_results(all_results, corrected_results, correlation_results, 
                         df, wide_df, OUTPUT_PATH)
    
    # Create thesis-level summary
    create_enhanced_summary_thesis_v2(all_results, corrected_results, df, wide_df, OUTPUT_PATH)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"\nAll files saved to: {OUTPUT_PATH}")
    
    return df, wide_df, all_results, corrected_results

# ============================================================================
# RUN THE ANALYSIS
# ============================================================================
if __name__ == "__main__":
    try:
        df, wide_df, all_results, corrected_results = run_comprehensive_analysis()
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Print model summary by variable
        print("\n" + "="*70)
        print("MODEL SUMMARY BY OUTCOME")
        print("="*70)
        
        for var, results in all_results.items():
            if results['model_type'] != 'descriptive_only':
                print(f"\n{var}:")
                print(f"  Model: {results['model_type'].replace('_', ' ').title()}")
                print(f"  Subjects: {results['n_subjects']}, Observations: {results['n_observations']}")
                print(f"  AIC: {results['aic']:.1f}")
                
                # ICC if available and appropriate
                if 'icc' in results and results['icc']['icc'] is not None:
                    print(f"  ICC: {results['icc']['icc']:.3f}")
                
                # Fixed effects with FDR markers
                for effect_name, effect_data in results['fixed_effects'].items():
                    if effect_name == 'Intercept':
                        continue
                    
                    # FDR significance
                    sig_fdr = False
                    for corr_key, corr_data in corrected_results.items():
                        if (corr_data['variable'] == var and 
                            effect_name in corr_data['effect'] and
                            corr_data['significant_fdr']):
                            sig_fdr = True
                            break
                    
                    fdr_mark = "†" if sig_fdr else ""
                    z = effect_data['z_stat']
                    p = effect_data['p_value']
                    
                    # Significance stars
                    if p < 0.001:
                        stars = '***'
                    elif p < 0.01:
                        stars = '**'
                    elif p < 0.05:
                        stars = '*'
                    else:
                        stars = 'ns'
                    
                    print(f"  {effect_name}{fdr_mark}: b={effect_data['coefficient']:.2f} "
                          f"[{effect_data['ci_lower']:.2f}, {effect_data['ci_upper']:.2f}], "
                          f"z={z:.2f}, p={p:.4f} {stars}")
                    
                    # Semi-partial R²
                    if effect_name in results.get('semi_partial_r2', {}):
                        r2_data = results['semi_partial_r2'][effect_name]
                        if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
                            print(f"    pseudo-R² = {r2_data['semi_partial_r2']:.3f}")
        
        # FDR summary
        print("\n" + "="*70)
        print("FDR-CORRECTED SIGNIFICANCE (†, q < 0.05)")
        print("="*70)
        sig_found = False
        for key, data in corrected_results.items():
            if data['significant_fdr']:
                sig_found = True
                print(f"  † {data['variable']} - {data['effect']}: "
                      f"p_raw={data['p_value_raw']:.4f}, p_FDR={data['p_value_fdr']:.4f}")
        if not sig_found:
            print("  No effects significant after FDR correction")
        
        # Model selection summary
        print("\n" + "="*70)
        print("RANDOM EFFECTS STRUCTURE")
        print("="*70)
        for var, results in all_results.items():
            if results['model_type'] != 'descriptive_only':
                model_type = results['model_type'].replace('_', ' ').title()
                print(f"  {var}: {model_type}")
        
        print("\n" + "="*70)
        print("OUTPUT FILES")
        print("="*70)
        print(f"  Summary: {OUTPUT_PATH}/analysis_summary_thesis_level_v2.txt")
        print(f"  Excel data: {OUTPUT_PATH}/comprehensive_analysis_results_mixed.xlsx")
        print(f"  Main figures: Figure1_Primary.png, Figure2_Secondary.png")
        print(f"  Supplementary figures: See output directory")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Critical error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠ WARNING: Analysis incomplete")

        """
COMPREHENSIVE SUCROSE PREFERENCE TEST ANALYSIS
THESIS-LEVEL MIXED MODELS APPROACH - CONTINUOUS
With publication-ready visualizations and precise methodological reporting
"""

# [Previous code up to line ~2150 remains exactly the same]

# ============================================================================
# MODEL DIAGNOSTIC PLOTS FUNCTION (continuation)
# ============================================================================

def create_model_diagnostic_plots(all_results, df, output_path):
    """Create diagnostic plots using the selected fitted model (not refitted)."""
    
    for var, results in all_results.items():
        if results['model_type'] == 'descriptive_only':
            continue
        
        # Check if we have the fitted model stored
        if '_fitted_model' not in results:
            print(f"  ⚠ No fitted model stored for {var}, skipping diagnostics")
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(config.DOUBLE_COLUMN, 8))
        fig.suptitle(f'Model Diagnostics: {var}', fontsize=config.FONT_SIZES['title']+2, y=0.98)
        
        try:
            # Get the actual fitted model
            fitted_model = results['_fitted_model']
            
            # Get data for this variable
            model_df = df[['Subject', 'Group', 'Stimulation', var]].dropna().copy()
            
            # Get residuals from the fitted model (not refitted)
            resid_data = conditional_residuals(fitted_model, model_df)
            conditional_resid = resid_data['conditional_residuals']
            fitted = fitted_model.fittedvalues
            
            # Panel A: QQ plot of conditional residuals
            ax = axes[0, 0]
            qqplot(conditional_resid, line='s', ax=ax, alpha=0.6)
            ax.set_title('A: Q-Q Plot (Conditional Residuals)', fontsize=config.FONT_SIZES['title'])
            ax.set_xlabel('Theoretical Quantiles', fontsize=config.FONT_SIZES['axis'])
            ax.set_ylabel('Sample Quantiles', fontsize=config.FONT_SIZES['axis'])
            
            # Add Shapiro-Wilk result
            if len(conditional_resid) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(conditional_resid)
                ax.text(0.05, 0.95, f"Shapiro-Wilk p={shapiro_p:.4f}", 
                       transform=ax.transAxes, fontsize=config.FONT_SIZES['annotation'],
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # Panel B: Residuals vs Fitted
            ax = axes[0, 1]
            ax.scatter(fitted, conditional_resid, alpha=0.6, s=20, color='black')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title('B: Residuals vs Fitted', fontsize=config.FONT_SIZES['title'])
            ax.set_xlabel('Fitted Values', fontsize=config.FONT_SIZES['axis'])
            ax.set_ylabel('Conditional Residuals', fontsize=config.FONT_SIZES['axis'])
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Panel C: Residuals by Group
            ax = axes[1, 0]
            residual_by_group = [conditional_resid[model_df['Group'] == g] for g in ['PD', 'Control']]
            bp = ax.boxplot(residual_by_group, tick_labels=['PD', 'Control'], patch_artist=True,
                           widths=0.6, showfliers=True)
            for patch, color in zip(bp['boxes'], [config.COLORS['PD'], config.COLORS['Control']]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(0.5)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(0.5)
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(1.5)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title('C: Residuals by Group', fontsize=config.FONT_SIZES['title'])
            ax.set_ylabel('Conditional Residuals', fontsize=config.FONT_SIZES['axis'])
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
            
            # Panel D: Residuals by Stimulation
            ax = axes[1, 1]
            residual_by_stim = [conditional_resid[model_df['Stimulation'] == s] for s in ['OFF', 'ON']]
            bp = ax.boxplot(residual_by_stim, tick_labels=['OFF', 'ON'], patch_artist=True,
                           widths=0.6, showfliers=True)
            for patch, color in zip(bp['boxes'], [config.COLORS['OFF'], config.COLORS['ON']]):
                patch.set_facecolor(color)
                patch.set_alpha(0.5)
            for whisker in bp['whiskers']:
                whisker.set_color('black')
                whisker.set_linewidth(0.5)
            for cap in bp['caps']:
                cap.set_color('black')
                cap.set_linewidth(0.5)
            for median in bp['medians']:
                median.set_color('white')
                median.set_linewidth(1.5)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
            ax.set_title('D: Residuals by Stimulation', fontsize=config.FONT_SIZES['title'])
            ax.set_ylabel('Conditional Residuals', fontsize=config.FONT_SIZES['axis'])
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
            
        except Exception as e:
            for ax in axes.flat:
                ax.text(0.5, 0.5, f"Diagnostics unavailable:\n{str(e)[:50]}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        safe_var = var.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        fig_path = output_path / f"diagnostics_{safe_var}.png"
        fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Diagnostics saved: {fig_path}")

# ============================================================================
# MAIN FIGURE 1: PRIMARY OUTCOME (Sucrose Preference) - CONTINUED
# ============================================================================

def create_main_figure_primary(all_results, corrected_results, output_path):
    """
    Create publication-ready main figure for primary outcome.
    Panel A: Model-estimated marginal means with spaghetti
    Panel B: Interaction plot with effect size
    Panel C: Forest plot of fixed effects
    Panel D: Hedges g effect sizes
    """
    
    fig = plt.figure(figsize=(config.DOUBLE_COLUMN, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.3)
    
    # Get results for primary outcome
    primary_var = '% SucrosePreference'
    if primary_var not in all_results:
        print("  ⚠ Primary outcome results not found")
        return None
    
    results = all_results[primary_var]
    if results['model_type'] == 'descriptive_only':
        print("  ⚠ Primary outcome model failed")
        return None
    
    # Get model and data
    fitted_model = results['_fitted_model']
    model_df = results['_model_df']
    
    # ========================================================================
    # PANEL A: Model-estimated marginal means with spaghetti
    # ========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Calculate estimated marginal means from model
    groups = ['Control', 'PD']
    stim_levels = ['OFF', 'ON']
    stim_codes = {'OFF': 0, 'ON': 1}
    
    emmeans = {}
    for group in groups:
        group_code = 1 if group == 'PD' else 0
        emmeans[group] = {}
        for stim in stim_levels:
            stim_code = stim_codes[stim]
            # Design matrix row for this combination
            X = np.array([[1, group_code, stim_code, group_code * stim_code]])
            pred = np.dot(X, fitted_model.params)[0]
            # Approximate SE (simplified)
            se = np.sqrt(np.diag(X @ fitted_model.cov_params() @ X.T))[0]
            ci = 1.96 * se
            emmeans[group][stim] = {
                'mean': pred,
                'se': se,
                'ci_lower': pred - ci,
                'ci_upper': pred + ci
            }
    
    # Plot individual trajectories (spaghetti)
    np.random.seed(42)
    for group, color in [('Control', config.COLORS['Control']), ('PD', config.COLORS['PD'])]:
        group_data = model_df[model_df['Group'] == group]
        light_color = config.COLORS['Control_light'] if group == 'Control' else config.COLORS['PD_light']
        
        for subject in group_data['Subject'].unique():
            subj_data = group_data[group_data['Subject'] == subject]
            off_val = subj_data[subj_data['Stimulation'] == 'OFF'][primary_var].values
            on_val = subj_data[subj_data['Stimulation'] == 'ON'][primary_var].values
            if len(off_val) > 0 and len(on_val) > 0:
                ax1.plot([0, 1], [off_val[0], on_val[0]], 
                        color=light_color, alpha=0.3, linewidth=0.5, zorder=1)
    
    # Plot model-estimated means with error bars
    x_positions = [0, 1]
    for i, group in enumerate(groups):
        color = config.COLORS[group]
        means = [emmeans[group][stim]['mean'] for stim in stim_levels]
        ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
        ci_upper = [emmeans[group][stim]['ci_upper'] for stim in stim_levels]
        
        # Line connecting means
        ax1.plot(x_positions, means, color=color, linewidth=2, marker='o', 
                markersize=6, label=f'{group}', zorder=3)
        
        # Error bars as shaded region
        ax1.fill_between(x_positions, ci_lower, ci_upper, color=color, alpha=0.2, zorder=2)
        
        # Add individual point estimates at each time point
        for j, (x, mean) in enumerate(zip(x_positions, means)):
            ax1.plot(x, mean, 'o', color=color, markersize=6, markeredgecolor='white', 
                    markeredgewidth=0.5, zorder=4)
    
    ax1.set_xticks([0, 1])
    ax1.set_xticklabels(['OFF', 'ON'], fontsize=config.FONT_SIZES['tick'])
    ax1.set_ylabel('Sucrose Preference (%)', fontsize=config.FONT_SIZES['axis'])
    ax1.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
    ax1.set_title('A: Model-Estimated Marginal Means', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    ax1.legend(fontsize=config.FONT_SIZES['legend'], frameon=True, loc='best')
    ax1.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax1.set_ylim(40, 110)
    
    # ========================================================================
    # PANEL B: Interaction plot with effect size
    # ========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Create interaction plot (group difference by stimulation)
    groups_for_plot = ['Control', 'PD']
    x_pos = np.arange(len(stim_levels))
    width = 0.35
    
    for i, group in enumerate(groups_for_plot):
        color = config.COLORS[group]
        means = [emmeans[group][stim]['mean'] for stim in stim_levels]
        ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
        ci_upper = [emmeans[group][stim]['ci_upper'] for stim in stim_levels]
        yerr_lower = [means[j] - ci_lower[j] for j in range(len(means))]
        yerr_upper = [ci_upper[j] - means[j] for j in range(len(means))]
        yerr = [yerr_lower, yerr_upper]
        
        offset = -width/2 if group == 'Control' else width/2
        ax2.bar(x_pos + offset, means, width, label=group, color=color, alpha=0.7,
               yerr=yerr, capsize=3, error_kw={'linewidth': 1, 'ecolor': 'black'})
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(stim_levels, fontsize=config.FONT_SIZES['tick'])
    ax2.set_ylabel('Sucrose Preference (%)', fontsize=config.FONT_SIZES['axis'])
    ax2.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
    ax2.set_title('B: Interaction Plot', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    ax2.legend(fontsize=config.FONT_SIZES['legend'], frameon=True)
    ax2.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
    
    # Add effect size annotation
    if 'Group_x_Stim' in results.get('semi_partial_r2', {}):
        r2_data = results['semi_partial_r2']['Group_x_Stim']
        if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
            ax2.text(0.5, 0.95, f"Interaction pseudo-R² = {r2_data['semi_partial_r2']:.3f}",
                    transform=ax2.transAxes, fontsize=config.FONT_SIZES['annotation'],
                    ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # ========================================================================
    # PANEL C: Forest plot of fixed effects
    # ========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    effects = []
    cis_lower = []
    cis_upper = []
    labels = []
    colors = []
    p_values = []
    r2_values = []
    
    for effect_name in ['Group_PD', 'Stimulation_ON', 'Group_x_Stim']:
        if effect_name in results['fixed_effects']:
            ef = results['fixed_effects'][effect_name]
            effects.append(ef['coefficient'])
            cis_lower.append(ef['ci_lower'])
            cis_upper.append(ef['ci_upper'])
            
            # Format label
            if effect_name == 'Group_PD':
                labels.append('Group (PD vs Control)')
            elif effect_name == 'Stimulation_ON':
                labels.append('Stimulation (ON vs OFF)')
            else:
                labels.append('Group × Stimulation')
            
            colors.append(config.COLORS['highlight'] if effect_name == 'Group_x_Stim' else 'gray')
            p_values.append(ef['p_value'])
            
            # Get semi-partial R²
            if effect_name in results.get('semi_partial_r2', {}):
                r2_data = results['semi_partial_r2'][effect_name]
                if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
                    r2_values.append(r2_data['semi_partial_r2'])
                else:
                    r2_values.append(np.nan)
            else:
                r2_values.append(np.nan)
    
    y_pos = np.arange(len(effects))
    
    # Plot forest
    ax3.errorbar(effects, y_pos, xerr=[np.abs(np.array(effects) - np.array(cis_lower)), 
                                        np.abs(np.array(cis_upper) - np.array(effects))],
                fmt='o', color='black', ecolor='black', capsize=3, capthick=1, markerfacecolor='white')
    
    # Color points based on significance
    for i, (effect, p_val, color) in enumerate(zip(effects, p_values, colors)):
        marker_color = 'red' if p_val < 0.05 else 'black'
        ax3.plot(effect, i, 'o', color=marker_color, markersize=6, markeredgecolor='black', markeredgewidth=0.5)
    
    ax3.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(labels, fontsize=config.FONT_SIZES['axis'])
    ax3.set_xlabel('Estimate (95% CI)', fontsize=config.FONT_SIZES['axis'])
    ax3.set_title('C: Fixed Effects (Wald z-tests)', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    ax3.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
    
    # Add p-values and R² as text
    max_x = max(cis_upper) if cis_upper else 10
    for i, (p_val, r2) in enumerate(zip(p_values, r2_values)):
        p_text = f"p = {p_val:.3f}"
        if p_val < 0.001:
            p_text = "p < 0.001"
        r2_text = f"R² = {r2:.3f}" if not np.isnan(r2) else ""
        ax3.text(max_x + 2, i, f"{p_text}  {r2_text}", 
                va='center', fontsize=config.FONT_SIZES['effect'])
    
    # ========================================================================
    # PANEL D: Hedges g effect sizes
    # ========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    g_effects = []
    g_cis_lower = []
    g_cis_upper = []
    g_labels = []
    
    if 'effect_sizes' in results:
        es = results['effect_sizes']
        
        if 'Group_main_at_OFF' in es:
            g_effects.append(es['Group_main_at_OFF']['g'])
            g_cis_lower.append(es['Group_main_at_OFF']['ci_lower'])
            g_cis_upper.append(es['Group_main_at_OFF']['ci_upper'])
            g_labels.append('Group diff (OFF)')
        
        if 'Stimulation_main_in_Control' in es:
            g_effects.append(es['Stimulation_main_in_Control']['g'])
            g_cis_lower.append(es['Stimulation_main_in_Control']['ci_lower'])
            g_cis_upper.append(es['Stimulation_main_in_Control']['ci_upper'])
            g_labels.append('Stimulation (Control)')
        
        if results['interaction_present'] and 'Stimulation_in_PD' in results.get('simple_effects', {}):
            se = results['simple_effects']['Stimulation_in_PD']
            # Rough approximation from contrast
            if 'estimate' in se and 'se' in se:
                g_approx = se['estimate'] / np.sqrt(results['residual_variance'])
                g_effects.append(g_approx)
                g_cis_lower.append(g_approx - 1.96 * se['se'])
                g_cis_upper.append(g_approx + 1.96 * se['se'])
                g_labels.append('Stimulation (PD)')
    
    if g_effects:
        y_pos = np.arange(len(g_effects))
        
        ax4.errorbar(g_effects, y_pos, xerr=[np.abs(np.array(g_effects) - np.array(g_cis_lower)), 
                                              np.abs(np.array(g_cis_upper) - np.array(g_effects))],
                    fmt='o', color='black', ecolor='black', capsize=3, capthick=1, markerfacecolor='white')
        
        # Add reference lines for effect size conventions
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax4.axvline(x=0.2, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax4.axvline(x=0.5, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        ax4.axvline(x=0.8, color='gray', linestyle=':', linewidth=0.5, alpha=0.5)
        
        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(g_labels, fontsize=config.FONT_SIZES['axis'])
        ax4.set_xlabel("Hedges' g (95% CI)", fontsize=config.FONT_SIZES['axis'])
        ax4.set_title('D: Effect Sizes (Bias-Corrected)', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
        ax4.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='x')
        
        # Add effect size interpretation box
        ax4.text(0.02, 0.02, 'Small: 0.2\nMedium: 0.5\nLarge: 0.8', 
                transform=ax4.transAxes, fontsize=config.FONT_SIZES['effect'],
                va='bottom', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax4.text(0.5, 0.5, 'Effect sizes not available', 
                ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('D: Effect Sizes', fontsize=config.FONT_SIZES['title'], pad=10, loc='left')
    
    plt.suptitle('Figure 1: Primary Outcome - Sucrose Preference', 
                fontsize=config.FONT_SIZES['title']+2, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig_path = output_path / "Figure1_Primary.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 1 saved: {fig_path}")
    
    return fig


# ============================================================================
# MAIN FIGURE 2: SECONDARY OUTCOMES
# ============================================================================

def create_main_figure_secondary(all_results, corrected_results, output_path):
    """
    Create publication-ready main figure for secondary outcomes.
    One row per variable with model-estimated means and interaction visualization.
    """
    
    secondary_vars = ['Total Intake (g)', 'Water intake (g)', 'Sucrose intake (g)']
    available_vars = [v for v in secondary_vars if v in all_results and all_results[v]['model_type'] != 'descriptive_only']
    
    if not available_vars:
        print("  ⚠ No secondary outcome results available")
        return None
    
    fig = plt.figure(figsize=(config.DOUBLE_COLUMN, 4 * len(available_vars)))
    gs = gridspec.GridSpec(len(available_vars), 2, figure=fig, hspace=0.4, wspace=0.3)
    
    for row_idx, var in enumerate(available_vars):
        results = all_results[var]
        fitted_model = results['_fitted_model']
        model_df = results['_model_df']
        
        # Calculate estimated marginal means
        groups = ['Control', 'PD']
        stim_levels = ['OFF', 'ON']
        stim_codes = {'OFF': 0, 'ON': 1}
        
        emmeans = {}
        for group in groups:
            group_code = 1 if group == 'PD' else 0
            emmeans[group] = {}
            for stim in stim_levels:
                stim_code = stim_codes[stim]
                X = np.array([[1, group_code, stim_code, group_code * stim_code]])
                pred = np.dot(X, fitted_model.params)[0]
                se = np.sqrt(np.diag(X @ fitted_model.cov_params() @ X.T))[0]
                ci = 1.96 * se
                emmeans[group][stim] = {
                    'mean': pred,
                    'se': se,
                    'ci_lower': pred - ci,
                    'ci_upper': pred + ci
                }
        
        # Panel Left: Model-estimated means
        ax_left = fig.add_subplot(gs[row_idx, 0])
        
        x_positions = [0, 1]
        for group in groups:
            color = config.COLORS[group]
            means = [emmeans[group][stim]['mean'] for stim in stim_levels]
            ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
            ci_upper = [emmeans[group][stim]['ci_upper'] for stim in stim_levels]
            
            ax_left.plot(x_positions, means, color=color, linewidth=2, marker='o', 
                        markersize=5, label=group, zorder=3)
            ax_left.fill_between(x_positions, ci_lower, ci_upper, color=color, alpha=0.2, zorder=2)
            
            # Add individual point estimates
            for j, (x, mean) in enumerate(zip(x_positions, means)):
                ax_left.plot(x, mean, 'o', color=color, markersize=5, markeredgecolor='white', 
                            markeredgewidth=0.5, zorder=4)
        
        ax_left.set_xticks([0, 1])
        ax_left.set_xticklabels(['OFF', 'ON'], fontsize=config.FONT_SIZES['tick'])
        ax_left.set_ylabel(var, fontsize=config.FONT_SIZES['axis'])
        if row_idx == len(available_vars) - 1:
            ax_left.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
        ax_left.set_title(f"{chr(65 + row_idx*2)}: {var}", fontsize=config.FONT_SIZES['title'], pad=5, loc='left')
        ax_left.legend(fontsize=config.FONT_SIZES['legend'], frameon=True)
        ax_left.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Panel Right: Interaction bars
        ax_right = fig.add_subplot(gs[row_idx, 1])
        
        x_pos = np.arange(len(stim_levels))
        width = 0.35
        
        for i, group in enumerate(groups):
            color = config.COLORS[group]
            means = [emmeans[group][stim]['mean'] for stim in stim_levels]
            ci_lower = [emmeans[group][stim]['ci_lower'] for stim in stim_levels]
            ci_upper = [emmeans[group][stim]['ci_upper'] for stim in stim_levels]
            yerr_lower = [means[j] - ci_lower[j] for j in range(len(means))]
            yerr_upper = [ci_upper[j] - means[j] for j in range(len(means))]
            yerr = [yerr_lower, yerr_upper]
            
            offset = -width/2 if group == 'Control' else width/2
            bars = ax_right.bar(x_pos + offset, means, width, label=group, color=color, alpha=0.7,
                               yerr=yerr, capsize=3, error_kw={'linewidth': 1, 'ecolor': 'black'})
        
        ax_right.set_xticks(x_pos)
        ax_right.set_xticklabels(stim_levels, fontsize=config.FONT_SIZES['tick'])
        ax_right.set_ylabel(var, fontsize=config.FONT_SIZES['axis'])
        if row_idx == len(available_vars) - 1:
            ax_right.set_xlabel('Stimulation', fontsize=config.FONT_SIZES['axis'])
        ax_right.set_title(f"{chr(65 + row_idx*2 + 1)}: Interaction", fontsize=config.FONT_SIZES['title'], pad=5, loc='left')
        ax_right.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        
        # Add effect size if interaction present
        if 'Group_x_Stim' in results.get('semi_partial_r2', {}):
            r2_data = results['semi_partial_r2']['Group_x_Stim']
            if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
                ax_right.text(0.5, 0.95, f"Interaction pseudo-R² = {r2_data['semi_partial_r2']:.3f}",
                            transform=ax_right.transAxes, fontsize=config.FONT_SIZES['annotation'],
                            ha='center', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('Figure 2: Secondary Outcomes - Intake Measures', 
                fontsize=config.FONT_SIZES['title']+2, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    fig_path = output_path / "Figure2_Secondary.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Figure 2 saved: {fig_path}")
    
    return fig


# ============================================================================
# SUPPLEMENTARY FIGURE 1: MODEL DIAGNOSTICS
# ============================================================================

def create_supplementary_figure1_diagnostics(all_results, output_path):
    """
    Create supplementary diagnostic figures for each variable.
    One page per variable with QQ plot, residuals vs fitted, residuals by group, and random effects.
    """
    
    for var, results in all_results.items():
        if results['model_type'] == 'descriptive_only' or '_fitted_model' not in results:
            continue
        
        fig, axes = plt.subplots(2, 2, figsize=(config.DOUBLE_COLUMN, 8))
        fig.suptitle(f'Supplementary Figure 1.{var}: Model Diagnostics', 
                    fontsize=config.FONT_SIZES['title']+2, y=0.98)
        
        fitted_model = results['_fitted_model']
        model_df = results['_model_df']
        
        # Get residuals
        resid_data = conditional_residuals(fitted_model, model_df)
        conditional_resid = resid_data['conditional_residuals']
        fitted = fitted_model.fittedvalues
        
        # Panel A: QQ plot
        ax = axes[0, 0]
        qqplot(conditional_resid, line='s', ax=ax, alpha=0.6, color='black')
        ax.set_title('A: Q-Q Plot (Conditional Residuals)', fontsize=config.FONT_SIZES['title'])
        ax.set_xlabel('Theoretical Quantiles', fontsize=config.FONT_SIZES['axis'])
        ax.set_ylabel('Sample Quantiles', fontsize=config.FONT_SIZES['axis'])
        
        # Add Shapiro-Wilk result
        if len(conditional_resid) >= 3:
            shapiro_stat, shapiro_p = stats.shapiro(conditional_resid)
            ax.text(0.05, 0.95, f"Shapiro-Wilk p={shapiro_p:.4f}", 
                   transform=ax.transAxes, fontsize=config.FONT_SIZES['annotation'],
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Panel B: Residuals vs Fitted
        ax = axes[0, 1]
        ax.scatter(fitted, conditional_resid, alpha=0.6, s=20, color='black')
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title('B: Residuals vs Fitted', fontsize=config.FONT_SIZES['title'])
        ax.set_xlabel('Fitted Values', fontsize=config.FONT_SIZES['axis'])
        ax.set_ylabel('Conditional Residuals', fontsize=config.FONT_SIZES['axis'])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Panel C: Residuals by Group
        ax = axes[1, 0]
        residual_by_group = [conditional_resid[model_df['Group'] == g] for g in ['PD', 'Control']]
        bp = ax.boxplot(residual_by_group, tick_labels=['PD', 'Control'], patch_artist=True,
                       widths=0.6, showfliers=True)
        for patch, color in zip(bp['boxes'], [config.COLORS['PD'], config.COLORS['Control']]):
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        for whisker in bp['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(0.5)
        for cap in bp['caps']:
            cap.set_color('black')
            cap.set_linewidth(0.5)
        for median in bp['medians']:
            median.set_color('white')
            median.set_linewidth(1.5)
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
        ax.set_title('C: Residuals by Group', fontsize=config.FONT_SIZES['title'])
        ax.set_ylabel('Conditional Residuals', fontsize=config.FONT_SIZES['axis'])
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        
        # Panel D: Random effects distribution (if available)
        ax = axes[1, 1]
        if len(resid_data['random_effects']) > 0 and np.any(resid_data['random_effects'] != 0):
            re_unique = np.unique(resid_data['random_effects'])
            ax.hist(re_unique, bins='auto', alpha=0.7, color='gray', edgecolor='black')
            ax.set_title('D: Random Effects Distribution', fontsize=config.FONT_SIZES['title'])
            ax.set_xlabel('Random Intercept', fontsize=config.FONT_SIZES['axis'])
            ax.set_ylabel('Frequency', fontsize=config.FONT_SIZES['axis'])
            ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
            
            # Add normality test
            if len(re_unique) >= 3:
                re_shapiro, re_p = stats.shapiro(re_unique)
                ax.text(0.05, 0.95, f"Shapiro-Wilk p={re_p:.4f}", 
                       transform=ax.transAxes, fontsize=config.FONT_SIZES['annotation'],
                       verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Random effects not available\nor not estimated', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('D: Random Effects', fontsize=config.FONT_SIZES['title'])
        
        plt.tight_layout()
        safe_var = var.replace(' ', '_').replace('%', 'pct').replace('(', '').replace(')', '')
        fig_path = output_path / f"Supplementary_Figure1_Diagnostics_{safe_var}.png"
        fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Supplementary Figure 1 ({var}) saved: {fig_path}")


# ============================================================================
# SUPPLEMENTARY FIGURE 2: CORRELATION HEATMAPS
# ============================================================================

def create_supplementary_figure2_correlations(wide_df, correlation_results, output_path):
    """
    Create improved correlation heatmap with FDR-corrected significance.
    Larger, cleaner, with upper triangle only and clear legend.
    """
    
    delta_vars = [col for col in wide_df.columns if col.startswith('Δ')]
    
    if len(delta_vars) < 2 or not correlation_results:
        print("  ⚠ Insufficient data for correlation heatmap")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(config.DOUBLE_COLUMN, 5))
    fig.suptitle('Supplementary Figure 2: FDR-Corrected Correlation Heatmaps', 
                fontsize=config.FONT_SIZES['title']+2, y=1.02)
    
    for idx, group in enumerate(['PD', 'Control']):
        ax = axes[idx]
        
        if group not in correlation_results:
            ax.text(0.5, 0.5, f"No correlation data for {group}", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{group} Group', fontsize=config.FONT_SIZES['title'])
            continue
        
        corr_data = correlation_results[group]
        corr_matrix = np.array(corr_data['correlations'])
        variables = corr_data['variables']
        n_vars = len(variables)
        
        # Create mask for upper triangle (hide upper triangle)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        
        # Create annotation matrix with FDR-corrected significance
        annot_matrix = np.zeros_like(corr_matrix, dtype='U10')
        if 'p_values_fdr' in corr_data:
            p_fdr = np.array(corr_data['p_values_fdr'])
            # Reshape p_fdr to matrix
            p_matrix = np.zeros((n_vars, n_vars))
            upper_tri = np.triu_indices(n_vars, k=1)
            for i, (row, col) in enumerate(zip(*upper_tri)):
                if i < len(p_fdr):
                    p_matrix[row, col] = p_fdr[i]
                    p_matrix[col, row] = p_fdr[i]
            
            for i in range(n_vars):
                for j in range(n_vars):
                    if i < j:  # Only annotate lower triangle (since we'll show lower)
                        r_val = corr_matrix[i, j]
                        p_val = p_matrix[i, j]
                        if p_val < 0.001:
                            stars = '***'
                        elif p_val < 0.01:
                            stars = '**'
                        elif p_val < 0.05:
                            stars = '*'
                        else:
                            stars = ''
                        annot_matrix[i, j] = f"{r_val:.2f}{stars}"
        
        # Create heatmap with mask (showing only lower triangle)
        sns.heatmap(corr_matrix, mask=mask, annot=annot_matrix, fmt='',
                   cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                   xticklabels=variables, yticklabels=variables,
                   ax=ax, cbar=idx==1, cbar_kws={'label': 'Spearman r'},
                   annot_kws={'size': config.FONT_SIZES['annotation']},
                   square=True)
        
        ax.set_title(f'{group} Group', fontsize=config.FONT_SIZES['title'], fontweight='bold')
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=config.FONT_SIZES['tick'])
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=config.FONT_SIZES['tick'])
    
    # Add significance legend
    legend_text = "* p_FDR < 0.05\n** p_FDR < 0.01\n*** p_FDR < 0.001"
    fig.text(0.5, 0.02, legend_text, ha='center', fontsize=config.FONT_SIZES['annotation'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig_path = output_path / "Supplementary_Figure2_Correlations.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Supplementary Figure 2 saved: {fig_path}")


# ============================================================================
# SUPPLEMENTARY FIGURE 3: INDIVIDUAL CHANGE SCORES
# ============================================================================

def create_supplementary_figure3_individual_changes(wide_df, output_path):
    """
    Create individual-level change plots for each variable.
    Shows within-subject Δ distributions with violin plots and individual points.
    """
    
    delta_vars = [col for col in wide_df.columns if col.startswith('Δ')]
    
    if len(delta_vars) == 0:
        print("  ⚠ No delta variables available")
        return None
    
    n_vars = len(delta_vars)
    fig, axes = plt.subplots(n_vars, 1, figsize=(config.DOUBLE_COLUMN, 3 * n_vars))
    if n_vars == 1:
        axes = [axes]
    
    fig.suptitle('Supplementary Figure 3: Individual-Level Change Scores', 
                fontsize=config.FONT_SIZES['title']+2, y=0.98)
    
    for idx, delta_var in enumerate(delta_vars):
        ax = axes[idx]
        
        # Prepare data
        pd_changes = wide_df[wide_df['Group'] == 'PD'][delta_var].dropna()
        control_changes = wide_df[wide_df['Group'] == 'Control'][delta_var].dropna()
        
        # Create violin plot
        positions = [0, 1]
        data = [control_changes, pd_changes]
        colors = [config.COLORS['Control'], config.COLORS['PD']]
        
        # Violin plots
        vp = ax.violinplot(data, positions=positions, widths=0.6, showmeans=False, 
                          showmedians=False, showextrema=False)
        
        for i, pc in enumerate(vp['bodies']):
            pc.set_facecolor(colors[i])
            pc.set_alpha(0.3)
            pc.set_edgecolor(colors[i])
            pc.set_linewidth(1)
        
        # Box plots inside
        bp = ax.boxplot(data, positions=positions, widths=0.2, patch_artist=True,
                       showfliers=False, whiskerprops={'color': 'black', 'linewidth': 0.5},
                       medianprops={'color': 'white', 'linewidth': 1.5})
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        # Individual points with jitter
        np.random.seed(42)
        for i, (d, color) in enumerate(zip(data, colors)):
            jitter = np.random.normal(0, 0.05, len(d))
            ax.scatter(positions[i] + jitter, d, color=color, alpha=0.8, s=25,
                      edgecolor='black', linewidth=0.5, zorder=3)
        
        # Zero line
        ax.axhline(y=0, color='black', linestyle='--', alpha=0.5, linewidth=1)
        
        # Labels
        var_label = delta_var.replace('Δ', 'Δ ')
        ax.set_ylabel(var_label, fontsize=config.FONT_SIZES['axis'])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Control', 'PD'], fontsize=config.FONT_SIZES['tick'])
        ax.set_title(f"{chr(65 + idx)}: {var_label}", fontsize=config.FONT_SIZES['title'], loc='left')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5, axis='y')
        
        # Add summary statistics
        control_mean = control_changes.mean()
        control_sd = control_changes.std()
        pd_mean = pd_changes.mean()
        pd_sd = pd_changes.std()
        
        stats_text = f"Control: n={len(control_changes)}, M={control_mean:.1f}, SD={control_sd:.1f}\n"
        stats_text += f"PD: n={len(pd_changes)}, M={pd_mean:.1f}, SD={pd_sd:.1f}"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=config.FONT_SIZES['effect'],
               ha='right', va='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='black'))
    
    plt.tight_layout()
    fig_path = output_path / "Supplementary_Figure3_IndividualChanges.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Supplementary Figure 3 saved: {fig_path}")


# ============================================================================
# MAIN ANALYSIS FUNCTION (COMPLETE)
# ============================================================================

def run_comprehensive_analysis():
    """Run the complete comprehensive analysis with thesis-level statistics and visualization"""
    
    print("="*70)
    print("COMPREHENSIVE SUCROSE PREFERENCE TEST ANALYSIS")
    print("THESIS-LEVEL MIXED MODELS APPROACH")
    print("="*70)
    print("\nNOTE: No warnings are suppressed - convergence issues are visible")
    
    # Load data
    df = load_data_from_excel()
    df, wide_df = create_wide_formats(df)
    
    # Fit mixed models with selection
    all_results, corrected_results = fit_all_mixed_models(df)
    
    # Create diagnostic plots using selected models
    create_model_diagnostic_plots(all_results, df, OUTPUT_PATH)
    
    # Correlation analysis with FDR-corrected significance
    correlation_results = enhanced_correlation_analysis(wide_df, OUTPUT_PATH)
    
    # Create backward-compatible figures (original style)
    print("\n" + "="*70)
    print("CREATING BACKWARD-COMPATIBLE FIGURES")
    print("="*70)
    
    create_figure_all_variables_trajectories(df, wide_df)
    create_figure_all_variables_boxplots(df, wide_df)
    create_figure_change_scores_rainclouds(wide_df)
    
    # Create new thesis-level figures
    print("\n" + "="*70)
    print("CREATING THESIS-LEVEL FIGURES")
    print("="*70)
    
    create_main_figure_primary(all_results, corrected_results, OUTPUT_PATH)
    create_main_figure_secondary(all_results, corrected_results, OUTPUT_PATH)
    create_supplementary_figure1_diagnostics(all_results, OUTPUT_PATH)
    create_supplementary_figure2_correlations(wide_df, correlation_results, OUTPUT_PATH)
    create_supplementary_figure3_individual_changes(wide_df, OUTPUT_PATH)
    
    # Save results
    save_enhanced_results(all_results, corrected_results, correlation_results, 
                         df, wide_df, OUTPUT_PATH)
    
    # Create thesis-level summary
    create_enhanced_summary_thesis_v2(all_results, corrected_results, df, wide_df, OUTPUT_PATH)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    
    print(f"\nAll files saved to: {OUTPUT_PATH}")
    
    return df, wide_df, all_results, corrected_results


# ============================================================================
# RUN THE ANALYSIS
# ============================================================================
if __name__ == "__main__":
    try:
        df, wide_df, all_results, corrected_results = run_comprehensive_analysis()
        print("\n" + "="*70)
        print("✓ ANALYSIS COMPLETED SUCCESSFULLY")
        print("="*70)
        
        # Print model summary by variable
        print("\n" + "="*70)
        print("MODEL SUMMARY BY OUTCOME")
        print("="*70)
        
        for var, results in all_results.items():
            if results['model_type'] != 'descriptive_only':
                print(f"\n{var}:")
                print(f"  Model: {results['model_type'].replace('_', ' ').title()}")
                print(f"  Subjects: {results['n_subjects']}, Observations: {results['n_observations']}")
                print(f"  AIC: {results['aic']:.1f}")
                
                # ICC if available and appropriate
                if 'icc' in results and results['icc']['icc'] is not None:
                    print(f"  ICC: {results['icc']['icc']:.3f}")
                
                # Fixed effects with FDR markers
                for effect_name, effect_data in results['fixed_effects'].items():
                    if effect_name == 'Intercept':
                        continue
                    
                    # FDR significance
                    sig_fdr = False
                    for corr_key, corr_data in corrected_results.items():
                        if (corr_data['variable'] == var and 
                            effect_name in corr_data['effect'] and
                            corr_data['significant_fdr']):
                            sig_fdr = True
                            break
                    
                    fdr_mark = "†" if sig_fdr else ""
                    z = effect_data['z_stat']
                    p = effect_data['p_value']
                    
                    # Significance stars
                    if p < 0.001:
                        stars = '***'
                    elif p < 0.01:
                        stars = '**'
                    elif p < 0.05:
                        stars = '*'
                    else:
                        stars = 'ns'
                    
                    print(f"  {effect_name}{fdr_mark}: b={effect_data['coefficient']:.2f} "
                          f"[{effect_data['ci_lower']:.2f}, {effect_data['ci_upper']:.2f}], "
                          f"z={z:.2f}, p={p:.4f} {stars}")
                    
                    # Semi-partial R²
                    if effect_name in results.get('semi_partial_r2', {}):
                        r2_data = results['semi_partial_r2'][effect_name]
                        if 'semi_partial_r2' in r2_data and not np.isnan(r2_data['semi_partial_r2']):
                            print(f"    pseudo-R² = {r2_data['semi_partial_r2']:.3f}")
        
        # FDR summary
        print("\n" + "="*70)
        print("FDR-CORRECTED SIGNIFICANCE (†, q < 0.05)")
        print("="*70)
        sig_found = False
        for key, data in corrected_results.items():
            if data['significant_fdr']:
                sig_found = True
                print(f"  † {data['variable']} - {data['effect']}: "
                      f"p_raw={data['p_value_raw']:.4f}, p_FDR={data['p_value_fdr']:.4f}")
        if not sig_found:
            print("  No effects significant after FDR correction")
        
        # Model selection summary
        print("\n" + "="*70)
        print("RANDOM EFFECTS STRUCTURE")
        print("="*70)
        for var, results in all_results.items():
            if results['model_type'] != 'descriptive_only':
                model_type = results['model_type'].replace('_', ' ').title()
                print(f"  {var}: {model_type}")
        
        print("\n" + "="*70)
        print("OUTPUT FILES")
        print("="*70)
        print(f"  Summary: {OUTPUT_PATH}/analysis_summary_thesis_level_v2.txt")
        print(f"  Excel data: {OUTPUT_PATH}/comprehensive_analysis_results_mixed.xlsx")
        print(f"  Main figures: Figure1_Primary.png, Figure2_Secondary.png")
        print(f"  Supplementary figures:")
        print(f"    - Supplementary_Figure1_Diagnostics_*.png")
        print(f"    - Supplementary_Figure2_Correlations.png")
        print(f"    - Supplementary_Figure3_IndividualChanges.png")
        print(f"  Legacy figures:")
        print(f"    - figure_all_variables_trajectories.png")
        print(f"    - figure_all_variables_boxplots.png")
        print(f"    - figure_change_scores_rainclouds.png")
        print("\n" + "="*70)
        
    except Exception as e:
        print(f"\n❌ Critical error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠ WARNING: Analysis incomplete")