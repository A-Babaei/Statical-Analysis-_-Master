"""
COMPREHENSIVE SUCROSE PREFERENCE TEST ANALYSIS
MIXED MODELS APPROACH - STATISTICALLY CORRECT VERSION
With omega squared, boundary-corrected LRT, model-based contrasts, and proper diagnostics
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
from statsmodels.stats.diagnostic import het_breuschpagan
import warnings

# NO WARNINGS ARE SUPPRESSED - Convergence issues MUST be visible

# ============================================================================
# PATHS AND CONFIGURATION
# ============================================================================
INPUT_FILE = Path(r"G:\Master\Experiment\Statistics\Sucrose perference\Sucrose_Preference_LongFormat_Template.xlsx")
OUTPUT_PATH = Path(r"G:\Master\Experiment\Statistics\Sucrose perference\ComprehensiveResult")
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
        'table': 8
    }
    
    COLORS = {
        'PD': '#D55E00',      # Vermilion
        'Control': '#0072B2', # Blue
        'OFF': '#999999',     # Gray
        'ON': '#009E73',      # Green
        'highlight': '#E69F00' # Orange
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
# STATISTICAL UTILITY FUNCTIONS - CORRECTED VERSIONS
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


def omega_squared_from_model(model_result, exog, endog, random_var_dict):
    """
    Calculate omega squared (ω²) for fixed effects in mixed models.
    More appropriate than partial eta squared for LMMs.
    
    ω² = (SS_effect - df_effect * MS_error) / (SS_total + MS_error)
    """
    n = len(endog)
    k_fixed = len(model_result.params)
    
    # Total sum of squares
    SS_total = np.sum((endog - np.mean(endog))**2)
    
    # Residual variance and degrees of freedom
    MS_error = model_result.scale
    df_error = n - k_fixed - sum(1 for v in random_var_dict.values() if v > 0)
    
    omega_sq = {}
    
    # Get parameter names
    if hasattr(model_result, 'data') and hasattr(model_result.data, 'xnames'):
        param_names = model_result.data.xnames
    else:
        param_names = [f'Param_{i}' for i in range(k_fixed)]
    
    for i, param_name in enumerate(param_names):
        if i == 0:  # Skip intercept
            continue
            
        if i >= len(model_result.params):
            continue
            
        coef = model_result.params[i]
        se = model_result.bse[i] if i < len(model_result.bse) else 0
        
        # Approximate sum of squares for this effect using Wald statistic
        if se > 0:
            F_approx = (coef / se)**2
            SS_effect = F_approx * MS_error
            
            # Omega squared calculation
            omega = (SS_effect - 1 * MS_error) / (SS_total + MS_error)
            omega = max(0, min(1, omega))  # Clamp to [0, 1]
        else:
            omega = 0
            F_approx = 0
        
        omega_sq[param_name] = {
            'omega_squared': omega,
            'coefficient': coef,
            'se': se,
            'F_approx': F_approx
        }
    
    return omega_sq


def nakagawa_r2_corrected(model_result, exog, groups, random_var_dict):
    """
    CORRECTED Nakagawa R² for mixed models with proper handling of random effects.
    
    Marginal R² = σ²_fixed / (σ²_fixed + σ²_random + σ²_residual)
    Conditional R² = (σ²_fixed + σ²_random) / (σ²_fixed + σ²_random + σ²_residual)
    """
    # Get fixed effects parameters
    if hasattr(model_result, 'fe_params'):
        fixed_params = model_result.fe_params
    else:
        fixed_params = model_result.params
    
    # Ensure exog matches fixed_params length
    if exog.shape[1] > len(fixed_params):
        exog_fixed = exog[:, :len(fixed_params)]
    else:
        exog_fixed = exog
    
    # Fixed effects variance (using only fixed effects)
    fixed_pred = np.dot(exog_fixed, fixed_params[:exog_fixed.shape[1]])
    var_fixed = np.var(fixed_pred, ddof=1)
    
    # Random effects variance
    var_random = 0
    if model_result.cov_re is not None and model_result.cov_re.size > 0:
        cov_re = model_result.cov_re
        if hasattr(cov_re, 'values'):
            cov_re_values = cov_re.values
        else:
            cov_re_values = cov_re
        
        # Sum all elements of covariance matrix (includes variances and covariances)
        var_random = np.sum(np.abs(cov_re_values))  # Use absolute to handle negative covariances
    
    # Residual variance
    var_residual = model_result.scale
    
    # Total variance
    total_var = var_fixed + var_random + var_residual
    
    # R² calculations
    marginal_r2 = var_fixed / total_var if total_var > 0 else 0
    conditional_r2 = (var_fixed + var_random) / total_var if total_var > 0 else 0
    
    # Intraclass Correlation Coefficient (ICC)
    icc = var_random / (var_random + var_residual) if (var_random + var_residual) > 0 else 0
    
    return {
        'marginal_r2': marginal_r2,
        'conditional_r2': conditional_r2,
        'icc': icc,
        'var_fixed': var_fixed,
        'var_random': var_random,
        'var_residual': var_residual,
        'total_variance': total_var
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


def standardized_coefficients(model_result, exog, endog):
    """
    Calculate standardized coefficients (β) for mixed models.
    
    Standardized β = β * (σ_x / σ_y)
    where σ_x is SD of predictor, σ_y is SD of outcome
    """
    std_coef = {}
    
    beta = model_result.params
    sd_y = np.std(endog, ddof=1)
    
    # Get parameter names
    if hasattr(model_result, 'data') and hasattr(model_result.data, 'xnames'):
        param_names = model_result.data.xnames
    else:
        param_names = [f'Param_{i}' for i in range(len(beta))]
    
    for i, (name, coef) in enumerate(zip(param_names, beta)):
        if i == 0:  # Skip intercept
            continue
            
        # Get SD of predictor
        if i < exog.shape[1]:
            sd_x = np.std(exog[:, i], ddof=1)
            if sd_x > 0 and sd_y > 0:
                beta_std = coef * (sd_x / sd_y)
                std_coef[name] = beta_std
    
    return std_coef


def conditional_residuals(model_result, model_df):
    """
    Calculate conditional (level-1) and marginal (level-2) residuals
    for better diagnostic plots.
    """
    # Overall residuals
    raw_residuals = model_result.resid
    
    # Get random effects
    if hasattr(model_result, 'random_effects') and model_result.random_effects:
        # Subject-specific random effects
        re_dict = model_result.random_effects
        
        # Create array of random effects matched to observations
        re_values = np.zeros(len(model_df))
        subject_col = model_df.columns.get_loc('Subject') if 'Subject' in model_df.columns else 0
        
        for i, (subject, re) in enumerate(re_dict.items()):
            if isinstance(re, (np.ndarray, pd.Series)):
                re_val = re[0] if len(re) > 0 else 0
            else:
                re_val = re
            mask = model_df['Subject'].values == subject
            re_values[mask] = re_val
        
        # Conditional residuals (level-1)
        conditional_resid = raw_residuals
        
        # Marginal residuals (level-2) = random effects + conditional residuals
        marginal_resid = re_values + conditional_resid
    else:
        conditional_resid = raw_residuals
        marginal_resid = raw_residuals
        re_values = np.zeros(len(model_df))
    
    return {
        'raw_residuals': raw_residuals,
        'conditional_residuals': conditional_resid,
        'marginal_residuals': marginal_resid,
        'random_effects': re_values
    }


def improved_diagnostics(model_result, model_df):
    """
    Improved model diagnostics appropriate for LMMs.
    """
    diagnostics = {}
    
    # Get conditional residuals
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
            'note': 'Testing conditional residuals (level-1)'
        }
    
    # 2. Homoscedasticity - Breusch-Pagan on conditional residuals
    try:
        # Need exog for BP test - extract from model if available
        if hasattr(model_result, 'model') and hasattr(model_result.model, 'exog'):
            exog = model_result.model.exog
            # Use only first few columns to avoid singularity
            if exog.shape[1] > 1:
                exog_bp = exog[:, 1:min(4, exog.shape[1])]  # Skip intercept, limit to 3 predictors
                bp_stat, bp_p, f_stat, f_p = het_breuschpagan(conditional_resid, exog_bp)
                diagnostics['homoscedasticity'] = {
                    'test': 'Breusch-Pagan',
                    'statistic': float(bp_stat),
                    'p_value': float(bp_p),
                    'homogeneous': bp_p > 0.05,
                    'note': 'Testing conditional residuals'
                }
    except (ImportError, Exception) as e:
        diagnostics['homoscedasticity_note'] = 'Visual inspection recommended for LMMs'
    
    # 3. Durbin-Watson for autocorrelation
    if len(conditional_resid) > 1:
        # Sort by subject and stimulation to check autocorrelation
        dw = np.sum(np.diff(conditional_resid)**2) / np.sum(conditional_resid**2)
        diagnostics['durbin_watson'] = {
            'value': float(dw),
            'interpretation': 'Values near 2 indicate no autocorrelation',
            'note': 'Use with caution for clustered data'
        }
    
    # 4. Random effects normality (if random slope model)
    if len(resid_data['random_effects']) > 0 and np.any(resid_data['random_effects'] != 0):
        re_unique = np.unique(resid_data['random_effects'])
        if len(re_unique) >= 3:
            re_shapiro, re_p = stats.shapiro(re_unique)
            diagnostics['random_effects_normality'] = {
                'test': 'Shapiro-Wilk',
                'statistic': float(re_shapiro),
                'p_value': float(re_p),
                'normal': re_p > 0.05,
                'note': 'Testing distribution of random intercepts'
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
# IMPROVED MIXED MODEL FITTING FUNCTION
# ============================================================================

def fit_mixed_model_improved(df, dependent_var, family='primary'):
    """
    IMPROVED mixed model fitting with correct statistics.
    Uses omega squared, boundary-corrected LRT, and model-based contrasts.
    """
    
    print(f"\n  Fitting mixed model for: {dependent_var}")
    
    # Prepare data
    model_df = df[['Subject', 'Group', 'Stimulation', dependent_var]].dropna().copy()
    
    # Create numeric codes
    model_df['Group_encoded'] = (model_df['Group'] == 'PD').astype(int)
    model_df['Stimulation_encoded'] = (model_df['Stimulation'] == 'ON').astype(int)
    model_df['Interaction'] = model_df['Group_encoded'] * model_df['Stimulation_encoded']
    
    # Center predictors for standardized coefficients
    model_df['Group_centered'] = model_df['Group_encoded'] - model_df['Group_encoded'].mean()
    model_df['Stim_centered'] = model_df['Stimulation_encoded'] - model_df['Stimulation_encoded'].mean()
    
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
            print(f"    ✓ Random slope model selected (boundary-corrected p={lr_test['p_value_boundary_corrected']:.4f})")
        else:
            selected_model = result_intercept
            selected_model_type = 'random_intercept'
            print(f"    ✓ Random intercept model selected (boundary-corrected p={lr_test['p_value_boundary_corrected']:.4f})")
    else:
        selected_model = result_intercept
        selected_model_type = 'random_intercept'
        lr_test = {'statistic': np.nan, 'p_value_boundary_corrected': np.nan, 'significant_corrected': False}
        print(f"    ✓ Using random intercept model (LRT not possible)")
    
    model_selection['selected_model'] = selected_model_type
    model_selection['lr_test'] = lr_test
    
    # ========================================================================
    # EXTRACT FIXED EFFECTS (Wald z)
    # ========================================================================
    fixed_effects = {}
    
    for i, param_name in enumerate(exog_names):
        if i < len(selected_model.params):
            coef = selected_model.params[i]
            se = selected_model.bse[i] if i < len(selected_model.bse) else 0
            z_stat = selected_model.tvalues[i] if i < len(selected_model.tvalues) else 0
            p_value = selected_model.pvalues[i] if i < len(selected_model.pvalues) else 1.0
            
            # Wald z confidence intervals
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
                'inference_method': 'Wald z-test'
            }
    
    # ========================================================================
    # RANDOM EFFECTS
    # ========================================================================
    random_effects = {}
    random_var_dict = {}
    
    if selected_model.cov_re is not None and selected_model.cov_re.size > 0:
        if selected_model_type == 'random_intercept':
            random_var = float(selected_model.cov_re[0, 0])
            random_effects['Subject(Intercept)'] = {
                'var': random_var,
                'sd': np.sqrt(random_var) if random_var > 0 else 0
            }
            random_var_dict['var_intercept'] = random_var
        else:
            # Extract full covariance matrix
            re_names = ['Intercept', 'Stim_Slope']
            for i in range(selected_model.cov_re.shape[0]):
                for j in range(selected_model.cov_re.shape[1]):
                    if i == j:
                        var_name = f'var_{re_names[i]}'
                        var_val = float(selected_model.cov_re[i, j])
                        random_effects[var_name] = var_val
                        random_var_dict[var_name] = var_val
                    else:
                        cov_name = f'cov_{re_names[i]}_{re_names[j]}'
                        cov_val = float(selected_model.cov_re[i, j])
                        random_effects[cov_name] = cov_val
                        
                        # Calculate correlation
                        var_i = float(selected_model.cov_re[i, i])
                        var_j = float(selected_model.cov_re[j, j])
                        if var_i > 0 and var_j > 0:
                            corr = cov_val / np.sqrt(var_i * var_j)
                            random_effects[f'corr_{re_names[i]}_{re_names[j]}'] = corr
    
    # ========================================================================
    # CORRECTED R² AND EFFECT SIZES
    # ========================================================================
    r2_nakagawa = nakagawa_r2_corrected(selected_model, exog, groups, random_var_dict)
    
    # Omega squared instead of partial eta squared
    omega_sq = omega_squared_from_model(selected_model, exog, endog, random_var_dict)
    
    # Standardized coefficients
    std_coef = standardized_coefficients(selected_model, exog, endog)
    
    # ========================================================================
    # MAIN EFFECT SIZES (Hedges g - keep as is)
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
    # MODEL-BASED SIMPLE EFFECTS (instead of t-tests)
    # ========================================================================
    interaction_present = False
    simple_effects = {}
    
    if 'Group_x_Stim' in fixed_effects and fixed_effects['Group_x_Stim']['p_value'] < 0.05:
        interaction_present = True
        print(f"    ✓ Significant interaction detected (p={fixed_effects['Group_x_Stim']['p_value']:.4f})")
        
        # For random intercept model, use fixed effects to derive simple effects
        if selected_model_type == 'random_intercept':
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
                'method': 'Model-based estimate'
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
                'method': 'Model-based estimate (approximate SE)'
            }
    
    # ========================================================================
    # IMPROVED DIAGNOSTICS
    # ========================================================================
    diagnostics = improved_diagnostics(selected_model, model_df)
    
    # ========================================================================
    # COMPILE RESULTS
    # ========================================================================
    results = {
        'dependent_var': dependent_var,
        'family': family,
        'model_type': selected_model_type,
        'inference_method': 'Wald z-tests from linear mixed-effects model',
        'n_subjects': int(model_df['Subject'].nunique()),
        'n_observations': len(model_df),
        'fixed_effects': fixed_effects,
        'random_effects': random_effects,
        'residual_variance': float(selected_model.scale),
        'r_squared_nakagawa': r2_nakagawa,
        'omega_squared': omega_sq,
        'standardized_coefficients': std_coef,
        'effect_sizes': effect_sizes,
        'aic': float(selected_model.aic) if not np.isnan(selected_model.aic) else 0,
        'bic': float(selected_model.bic) if not np.isnan(selected_model.bic) else 0,
        'converged': selected_model.converged,
        'model_selection': model_selection,
        'interaction_present': interaction_present,
        'simple_effects': simple_effects,
        'group_means': group_means.to_dict() if hasattr(group_means, 'to_dict') else {},
        'diagnostics': diagnostics
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
        'inference_method': 'Descriptive statistics only (model failed)',
        'n_subjects': int(model_df['Subject'].nunique()) if len(model_df) > 0 else 0,
        'n_observations': len(model_df),
        'fixed_effects': {},
        'random_effects': {},
        'residual_variance': float(overall_std**2) if overall_std > 0 else 0,
        'r_squared_nakagawa': {'marginal_r2': 0, 'conditional_r2': 0, 'icc': 0},
        'omega_squared': {},
        'standardized_coefficients': {},
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
    
    # PRIMARY OUTCOME (explicitly defined)
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
            all_results[var] = fit_mixed_model_improved(df, var, family='primary')
    
    if secondary_vars:
        print("\nSECONDARY ENDPOINTS:")
        for var in secondary_vars:
            if var in df.columns:
                all_results[var] = fit_mixed_model_improved(df, var, family='secondary')
    
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
    print(f"  FDR correction applied within each family (Benjamini-Hochberg)")
    
    return all_results, corrected_results

# ============================================================================
# ENHANCED CORRELATION ANALYSIS WITH FDR-CORRECTED SIGNIFICANCE
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
    
    fig, axes = plt.subplots(1, 2, figsize=(config.DOUBLE_COLUMN, 4))
    correlation_results = {}
    
    for idx, group in enumerate(['PD', 'Control']):
        ax = axes[idx]
        group_data = wide_df[wide_df['Group'] == group][delta_vars].dropna()
        
        if len(group_data) < 3:
            ax.text(0.5, 0.5, f"Insufficient data\n(n={len(group_data)})", 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{group} Group', fontsize=config.FONT_SIZES['title'])
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
            
            # Create mask (True = hide, False = show)
            mask = np.ones_like(corr_matrix, dtype=bool)
            for idx_pair, (i, j) in enumerate(zip(*upper_tri_indices)):
                if reject[idx_pair]:  # Significant after FDR
                    mask[i, j] = False
                    mask[j, i] = False
            
            # Create annotation with FDR-corrected significance
            annot_matrix = np.zeros_like(corr_matrix, dtype='U15')
            for i in range(n_vars):
                for j in range(n_vars):
                    if i < j:
                        # Find the index in the upper triangle list
                        pair_idx = list(zip(*upper_tri_indices)).index((i, j))
                        p_fdr = p_corrected[pair_idx]
                        
                        # Significance stars based on FDR-corrected p-value
                        if p_fdr < 0.001:
                            stars = '***'
                        elif p_fdr < 0.01:
                            stars = '**'
                        elif p_fdr < 0.05:
                            stars = '*'
                        else:
                            stars = ''
                        
                        annot_matrix[i, j] = f"{corr_matrix[i, j]:.2f}{stars}"
                        annot_matrix[j, i] = f"{corr_matrix[i, j]:.2f}{stars}"
            
            # Create heatmap
            sns.heatmap(corr_matrix, mask=mask, annot=annot_matrix, 
                       fmt='', cmap='RdBu_r', center=0, vmin=-1, vmax=1,
                       xticklabels=delta_vars, yticklabels=delta_vars,
                       ax=ax, cbar=idx==1,
                       annot_kws={'size': config.FONT_SIZES['annotation']-2})
            
            correlation_results[group] = {
                'correlations': corr_matrix.tolist(),
                'p_values_original': p_matrix.tolist(),
                'p_values_fdr': p_corrected.tolist(),
                'significant_fdr': reject.tolist(),
                'variables': delta_vars
            }
        
        ax.set_title(f'{group} Group (Spearman r, FDR-corrected *)', 
                    fontsize=config.FONT_SIZES['title']-1)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    
    plt.suptitle('Delta Score Correlations with FDR Correction', 
                fontsize=config.FONT_SIZES['title']+2, y=1.02)
    plt.tight_layout()
    
    fig_path = output_path / "correlation_heatmaps_fdr.png"
    fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Correlation heatmaps saved: {fig_path}")
    
    return correlation_results

# ============================================================================
# MODEL DIAGNOSTIC PLOTS (using correct residuals)
# ============================================================================

def create_model_diagnostic_plots(all_results, df, output_path):
    """Create diagnostic plots for models using correct residuals"""
    
    for var, results in all_results.items():
        if results['model_type'] == 'descriptive_only':
            continue
            
        fig, axes = plt.subplots(2, 2, figsize=(config.DOUBLE_COLUMN, 6))
        fig.suptitle(f'Model Diagnostics: {var}', fontsize=config.FONT_SIZES['title']+2)
        
        try:
            # Get model residuals by refitting (simplified for diagnostics)
            model_df = df[['Subject', 'Group', 'Stimulation', var]].dropna().copy()
            model_df['Group_encoded'] = (model_df['Group'] == 'PD').astype(int)
            model_df['Stimulation_encoded'] = (model_df['Stimulation'] == 'ON').astype(int)
            model_df['Interaction'] = model_df['Group_encoded'] * model_df['Stimulation_encoded']
            
            exog = np.column_stack([
                np.ones(len(model_df)),
                model_df['Group_encoded'].values,
                model_df['Stimulation_encoded'].values,
                model_df['Interaction'].values
            ])
            endog = model_df[var].values
            groups = model_df['Subject'].values
            
            # Fit random intercept model for diagnostics
            model = MixedLM(endog, exog, groups, exog_re=np.ones(len(model_df)))
            result = model.fit(reml=True, method='bfgs', maxiter=1000)
            
            # Get conditional residuals
            resid_data = conditional_residuals(result, model_df)
            conditional_resid = resid_data['conditional_residuals']
            fitted = result.fittedvalues
            
            # QQ plot of conditional residuals
            ax = axes[0, 0]
            qqplot(conditional_resid, line='s', ax=ax, alpha=0.6)
            ax.set_title('Q-Q Plot (Conditional Residuals)', fontsize=config.FONT_SIZES['title'])
            
            # Residuals vs Fitted
            ax = axes[0, 1]
            ax.scatter(fitted, conditional_resid, alpha=0.6, s=20)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Residuals vs Fitted', fontsize=config.FONT_SIZES['title'])
            ax.set_xlabel('Fitted Values')
            ax.set_ylabel('Conditional Residuals')
            
            # Residuals by Group
            ax = axes[1, 0]
            residual_by_group = [conditional_resid[model_df['Group'] == g] for g in ['PD', 'Control']]
            ax.boxplot(residual_by_group, tick_labels=['PD', 'Control'], patch_artist=True)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Residuals by Group', fontsize=config.FONT_SIZES['title'])
            
            # Residuals by Stimulation
            ax = axes[1, 1]
            residual_by_stim = [conditional_resid[model_df['Stimulation'] == s] for s in ['OFF', 'ON']]
            ax.boxplot(residual_by_stim, tick_labels=['OFF', 'ON'], patch_artist=True)
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            ax.set_title('Residuals by Stimulation', fontsize=config.FONT_SIZES['title'])
            
            # Add Shapiro-Wilk result
            if len(conditional_resid) >= 3:
                shapiro_stat, shapiro_p = stats.shapiro(conditional_resid)
                axes[0, 0].text(0.05, 0.95, f"Shapiro-Wilk p={shapiro_p:.4f}", 
                               transform=axes[0, 0].transAxes, 
                               fontsize=config.FONT_SIZES['annotation'],
                               verticalalignment='top')
            
        except Exception as e:
            for ax in axes.flat:
                ax.text(0.5, 0.5, f"Diagnostics unavailable:\n{str(e)[:50]}", 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        safe_var = var.replace(' ', '_').replace('%', 'pct')
        fig_path = output_path / f"diagnostics_{safe_var}.png"
        fig.savefig(fig_path, dpi=config.DPI, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✓ Diagnostics saved: {fig_path}")

# ============================================================================
# ENHANCED RESULTS SUMMARY (APA-STYLE)
# ============================================================================

def create_enhanced_summary_v2(all_results, corrected_results, df, wide_df, output_path):
    """REVISED summary with correct mixed-model reporting standards."""
    
    summary_path = output_path / "analysis_summary_mixed_models_v2.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("COMPREHENSIVE SUCROSE PREFERENCE ANALYSIS - MIXED MODELS APPROACH\n")
        f.write("="*80 + "\n\n")
        f.write(f"Analysis performed on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXPERIMENTAL DESIGN\n")
        f.write("-"*40 + "\n")
        f.write("* Design: 2 (Group: PD vs Control) × 2 (Stimulation: OFF vs ON) mixed factorial\n")
        f.write("* Between-subject: Group\n")
        f.write("* Within-subject: Stimulation\n")
        f.write("* Statistical model: Linear mixed-effects model with random intercept by subject\n")
        f.write("* Inference: Wald z-tests (Type III SS approximation)\n")
        f.write("* Effect sizes: ω² (omega squared) for fixed effects, Nakagawa R² for model fit\n")
        f.write("* Multiple comparison correction: Benjamini-Hochberg FDR (α = 0.05) within families\n")
        f.write("* Primary outcome: Sucrose Preference (%)\n")
        f.write("* Secondary outcomes: Total Intake, Water Intake, Sucrose Intake\n\n")
        
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
                f.write(f"MODEL FAILED: {results.get('error', 'Unknown error')}\n")
                if 'group_stats' in results:
                    for key, stats in results['group_stats'].items():
                        f.write(f"  {key}: M = {stats['mean']:.2f}, SD = {stats['std']:.2f}, n = {stats['n']}\n")
                f.write("\n" + "-"*40 + "\n")
                continue
            
            # Model summary
            f.write(f"Model: {results['model_type'].replace('_', ' ').title()}\n")
            f.write(f"Converged: {results['converged']}\n")
            f.write(f"N subjects: {results['n_subjects']}, N observations: {results['n_observations']}\n")
            f.write(f"AIC: {results['aic']:.1f}, BIC: {results['bic']:.1f}\n\n")
            
            # Model selection info
            if 'model_selection' in results and results['model_selection'].get('lr_test'):
                lrt = results['model_selection']['lr_test']
                if not np.isnan(lrt.get('statistic', np.nan)):
                    f.write(f"Random slope test (boundary-corrected LRT):\n")
                    f.write(f"  χ²(2) = {lrt['statistic']:.2f}, ")
                    f.write(f"p_boundary-corrected = {lrt['p_value_boundary_corrected']:.4f}\n")
                    if lrt.get('p_value_standard') and not np.isnan(lrt['p_value_standard']):
                        f.write(f"  (Standard χ² p = {lrt['p_value_standard']:.4f})\n")
                    f.write(f"  Selected: {results['model_selection']['selected_model']}\n\n")
            
            # Fixed effects with APA-style reporting
            f.write("Fixed Effects (Wald z-tests):\n")
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
                
                # APA format: b = X.XX, SE = X.XX, z = X.XX, p = .XXX
                p_val = effect_data['p_value']
                p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< .0001"
                
                f.write(f"  {effect_name}{sig_marker}:\n")
                f.write(f"    b = {effect_data['coefficient']:.3f}, SE = {effect_data['se']:.3f}, ")
                f.write(f"95% CI [{effect_data['ci_lower']:.3f}, {effect_data['ci_upper']:.3f}]\n")
                f.write(f"    z = {effect_data['z_stat']:.2f}, p = {p_str}\n")
                
                # Omega squared
                if effect_name in results.get('omega_squared', {}):
                    omega = results['omega_squared'][effect_name]['omega_squared']
                    f.write(f"    ω² = {omega:.3f}")
                    
                    # Interpret omega squared
                    if omega < 0.01:
                        f.write(" (negligible effect)")
                    elif omega < 0.06:
                        f.write(" (small effect)")
                    elif omega < 0.14:
                        f.write(" (medium effect)")
                    else:
                        f.write(" (large effect)")
                    f.write("\n")
                
                # Standardized coefficient
                if effect_name in results.get('standardized_coefficients', {}):
                    beta_std = results['standardized_coefficients'][effect_name]
                    f.write(f"    β_std = {beta_std:.3f}\n")
            
            # Nakagawa R²
            if 'r_squared_nakagawa' in results:
                r2 = results['r_squared_nakagawa']
                f.write(f"\nModel Fit (Nakagawa R²):\n")
                f.write(f"  Marginal R² = {r2['marginal_r2']:.3f} (fixed effects only)\n")
                f.write(f"  Conditional R² = {r2['conditional_r2']:.3f} (full model)\n")
                f.write(f"  ICC = {r2['icc']:.3f} (proportion of variance at subject level)\n")
                f.write(f"  Variance components: σ²_fixed = {r2['var_fixed']:.2f}, ")
                f.write(f"σ²_random = {r2['var_random']:.2f}, ")
                f.write(f"σ²_residual = {r2['var_residual']:.2f}\n")
            
            # Simple effects (if interaction significant)
            if results['interaction_present'] and results['simple_effects']:
                f.write(f"\nSimple Effects of Stimulation within Groups (model-based estimates):\n")
                for effect_name, effect_data in results['simple_effects'].items():
                    p_val = effect_data['p_value']
                    p_str = f"{p_val:.4f}" if p_val >= 0.0001 else "< .0001"
                    
                    f.write(f"  {effect_name}:\n")
                    f.write(f"    estimate = {effect_data['estimate']:.3f}, SE = {effect_data['se']:.3f}, ")
                    f.write(f"z = {effect_data.get('z', 0):.2f}, p = {p_str}\n")
                    f.write(f"    [{effect_data['method']}]\n")
            
            # Hedges g effect sizes (supplemental)
            if results.get('effect_sizes'):
                f.write(f"\nSupplemental Effect Sizes (Hedges g with 95% CI):\n")
                for es_name, es_data in results['effect_sizes'].items():
                    if 'g' in es_data:
                        f.write(f"  {es_name}:\n")
                        f.write(f"    g = {es_data['g']:.3f} ")
                        f.write(f"[{es_data['ci_lower']:.3f}, {es_data['ci_upper']:.3f}]\n")
                        if 'bootstrap_ci_lower' in es_data and not np.isnan(es_data['bootstrap_ci_lower']):
                            f.write(f"    Bootstrap 95% CI: [{es_data['bootstrap_ci_lower']:.3f}, ")
                            f.write(f"{es_data['bootstrap_ci_upper']:.3f}]\n")
            
            # Diagnostics summary
            if results.get('diagnostics'):
                f.write(f"\nModel Diagnostics:\n")
                if 'residual_normality' in results['diagnostics']:
                    rd = results['diagnostics']['residual_normality']
                    f.write(f"  Residual normality (conditional): ")
                    f.write(f"W = {rd['statistic']:.3f}, p = {rd['p_value']:.4f} - ")
                    f.write(f"{'met' if rd['normal'] else 'violated'}\n")
                
                if 'homoscedasticity' in results['diagnostics']:
                    hd = results['diagnostics']['homoscedasticity']
                    f.write(f"  Homoscedasticity (Breusch-Pagan): ")
                    f.write(f"χ² = {hd['statistic']:.2f}, p = {hd['p_value']:.4f} - ")
                    f.write(f"{'met' if hd['homogeneous'] else 'violated'}\n")
                
                if 'durbin_watson' in results['diagnostics']:
                    dw = results['diagnostics']['durbin_watson']
                    f.write(f"  Autocorrelation (Durbin-Watson): {dw['value']:.3f} ")
                    f.write(f"({dw['interpretation']})\n")
            
            f.write("\n" + "-"*40 + "\n")
        
        f.write("\n\nFDR CORRECTION\n")
        f.write("-"*40 + "\n")
        f.write("Multiple testing correction applied within families:\n")
        f.write("* Family 1 (Primary - Preference): Sucrose Preference (3 tests)\n")
        f.write("* Family 2 (Secondary - Consumption): Total Intake, Water Intake, Sucrose Intake (9 tests)\n")
        f.write("Method: Benjamini-Hochberg (FDR), α = 0.05\n\n")
        
        f.write("Effects significant after FDR correction (†):\n")
        sig_found = False
        for key, data in corrected_results.items():
            if data['significant_fdr']:
                sig_found = True
                f.write(f"  † {data['variable']} - {data['effect']}\n")
                f.write(f"    p_raw = {data['p_value_raw']:.4f}, ")
                f.write(f"p_FDR = {data['p_value_fdr']:.4f}, ")
                f.write(f"q = {data['q_value']:.4f}\n")
        if not sig_found:
            f.write("  No effects significant after FDR correction\n")
        
        f.write("\n\nREPORTING NOTE\n")
        f.write("-"*40 + "\n")
        f.write("Results should be reported as: 'Linear mixed-effects models with random intercept by subject\n")
        f.write("were fitted using restricted maximum likelihood (REML). Fixed effects were evaluated using\n")
        f.write("Wald z-tests. Effect sizes are reported as omega squared (ω²) for fixed effects and Nakagawa's\n")
        f.write("R² for model fit. Multiple comparisons were controlled using Benjamini-Hochberg FDR correction\n")
        f.write("(α = 0.05) applied separately to primary and secondary outcome families.'\n\n")
        
        f.write("\n\nFILES GENERATED\n")
        f.write("-"*40 + "\n")
        f.write("1. figure_all_variables_trajectories.png\n")
        f.write("2. figure_all_variables_boxplots.png\n")
        f.write("3. figure_change_scores_rainclouds.png\n")
        f.write("4. correlation_heatmaps_fdr.png\n")
        f.write("5. diagnostic_plots_*.png\n")
        f.write("6. comprehensive_analysis_results_mixed.xlsx\n")
        f.write("7. analysis_summary_mixed_models.txt\n")
        f.write("8. analysis_summary_mixed_models_v2.txt (this file)\n")
    
    print(f"✓ Enhanced summary (v2) saved to: {summary_path}")


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
                    'Marginal_R2': results['r_squared_nakagawa']['marginal_r2'],
                    'Conditional_R2': results['r_squared_nakagawa']['conditional_r2'],
                    'ICC': results['r_squared_nakagawa']['icc'],
                    'AIC': results['aic'],
                    'Model_Type': results['model_type'],
                    'Converged': results['converged']
                }
                
                # Add omega squared if available
                if effect_name in results.get('omega_squared', {}):
                    row['Omega_Squared'] = results['omega_squared'][effect_name]['omega_squared']
                
                # Add standardized coefficient
                if effect_name in results.get('standardized_coefficients', {}):
                    row['Beta_Std'] = results['standardized_coefficients'][effect_name]
                
                mixed_results_rows.append(row)
        
        if mixed_results_rows:
            pd.DataFrame(mixed_results_rows).to_excel(
                writer, sheet_name='Mixed_Model_Results', index=False)
        
        # FDR corrected results
        if corrected_results:
            corrected_rows = list(corrected_results.values())
            pd.DataFrame(corrected_rows).to_excel(
                writer, sheet_name='FDR_Corrected', index=False)
        
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
# ORIGINAL FIGURE FUNCTIONS (unchanged - preserve aesthetics)
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
    """Create figure showing trajectories for all variables"""
    
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
    return fig

def create_figure_all_variables_boxplots(df):
    """Create figure showing boxplots for all variables"""
    
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
    return fig

def create_figure_change_scores_rainclouds(wide_df):
    """Create figure showing raincloud plots for all change scores"""
    
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
    return fig

# ============================================================================
# MAIN ANALYSIS FUNCTION
# ============================================================================

def run_comprehensive_analysis():
    """Run the complete comprehensive analysis with correct statistics"""
    
    print("="*70)
    print("COMPREHENSIVE SUCROSE PREFERENCE TEST ANALYSIS")
    print("MIXED MODELS APPROACH - STATISTICALLY CORRECT VERSION")
    print("="*70)
    print("\nNOTE: No warnings are suppressed - convergence issues are visible")
    
    # Load data
    df = load_data_from_excel()
    df, wide_df = create_wide_formats(df)
    
    # Fit mixed models with selection
    all_results, corrected_results = fit_all_mixed_models(df)
    
    # Create diagnostic plots (using correct residuals)
    create_model_diagnostic_plots(all_results, df, OUTPUT_PATH)
    
    # Correlation analysis with FDR-corrected significance
    correlation_results = enhanced_correlation_analysis(wide_df, OUTPUT_PATH)
    
    # Create figures
    print("\n" + "="*70)
    print("CREATING FIGURES")
    print("="*70)
    
    print("\n1. Creating trajectories figure...")
    fig1 = create_figure_all_variables_trajectories(df, wide_df)
    fig1_path = OUTPUT_PATH / "figure_all_variables_trajectories.png"
    fig1.savefig(fig1_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig1)
    print(f"   ✓ Saved as: {fig1_path}")
    
    print("\n2. Creating boxplots figure...")
    fig2 = create_figure_all_variables_boxplots(df)
    fig2_path = OUTPUT_PATH / "figure_all_variables_boxplots.png"
    fig2.savefig(fig2_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig2)
    print(f"   ✓ Saved as: {fig2_path}")
    
    print("\n3. Creating change scores raincloud figure...")
    fig3 = create_figure_change_scores_rainclouds(wide_df)
    fig3_path = OUTPUT_PATH / "figure_change_scores_rainclouds.png"
    fig3.savefig(fig3_path, dpi=config.DPI, bbox_inches='tight')
    plt.close(fig3)
    print(f"   ✓ Saved as: {fig3_path}")
    
    # Save results
    save_enhanced_results(all_results, corrected_results, correlation_results, 
                         df, wide_df, OUTPUT_PATH)
    
    # Create summary
    create_enhanced_summary_v2(all_results, corrected_results, df, wide_df, OUTPUT_PATH)
    
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
        print("\n✓ Analysis completed successfully!")
        
        # Print key findings with correct statistical notation
        print("\n" + "="*70)
        print("KEY FINDINGS (Wald z-tests from mixed models)")
        print("="*70)
        
        for var, results in all_results.items():
            if results['model_type'] != 'descriptive_only':
                print(f"\n{var}:")
                for effect_name, effect_data in results['fixed_effects'].items():
                    if effect_name != 'Intercept':
                        # Check FDR significance
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
                        
                        print(f"  {effect_name}{fdr_mark}: b={effect_data['coefficient']:.2f}, "
                             f"z={z:.2f}, p={p:.4f} {stars}")
                        
                        # Add omega squared if available
                        if effect_name in results.get('omega_squared', {}):
                            omega = results['omega_squared'][effect_name]['omega_squared']
                            print(f"    ω²={omega:.3f}")
        
        # Print FDR summary
        print("\n" + "="*70)
        print("FDR-CORRECTED SIGNIFICANCE (†)")
        print("="*70)
        sig_found = False
        for key, data in corrected_results.items():
            if data['significant_fdr']:
                sig_found = True
                print(f"  † {data['variable']} - {data['effect']}: "
                     f"p_raw={data['p_value_raw']:.4f}, p_FDR={data['p_value_fdr']:.4f}")
        if not sig_found:
            print("  No effects significant after FDR correction")
        
    except Exception as e:
        print(f"\n❌ Critical error during analysis: {e}")
        import traceback
        traceback.print_exc()
        print("\n⚠ WARNING: Analysis incomplete due to error")