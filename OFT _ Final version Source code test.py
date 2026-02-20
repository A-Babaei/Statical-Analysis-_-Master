"""
Open Field Test Analysis Pipeline
Nature Biomedical Engineering Style
Enhanced with Partial Eta Squared, Hedges' g, Symmetric Recovery Axis, and Configurable Thresholds
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, shapiro, mannwhitneyu, wilcoxon, zscore, pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import AnovaRM
from statsmodels.formula.api import mixedlm
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Try importing ptitprince for raincloud plots
try:
    import ptitprince as pt
    PTITPRINCE_AVAILABLE = True
except ImportError:
    PTITPRINCE_AVAILABLE = False
    print("Note: ptitprince not available. Using enhanced boxplots instead.")


class PublicationOFTAnalyzer:
    """
    Open Field Test analyzer for publication-quality results.
    Handles data loading, statistical analysis, and visualization.
    """
    
    # Class constants
    PARAMETERS_OF_INTEREST = [
        'TotalDistance_cm',      # Primary outcome
        'TimeMoving_s', 
        'MeanSpeed_cm_s',
        'Time PercentCentral',
        'PercentMoving'
    ]
    
    SECONDARY_PARAMETERS = [
        'TimeMoving_s',
        'MeanSpeed_cm_s', 
        'Time PercentCentral',
        'PercentMoving'
    ]
    
    PARAMETER_LABELS = {
        'TotalDistance_cm': 'Total Distance (cm)',
        'TimeMoving_s': 'Moving Time (s)',
        'MeanSpeed_cm_s': 'Mean Speed (cm/s)',
        'Time PercentCentral': 'Time in Center (%)',
        'PercentMoving': 'Time Moving (%)'
    }
    
    CONDITIONS = ['Pre', 'Stim', 'Post']
    GROUPS = ['PD', 'CO']
    
    # Publication colors
    COLORS = {
        'PD': '#D55E00',  # Orange
        'CO': '#0072B2',  # Blue
        'Pre': '#009E73',  # Green
        'Stim': '#D55E00',  # Orange  
        'Post': '#CC79A7'   # Pink
    }
    
    # --- SCIENTIFIC UPGRADE 4: CONFIGURABLE RESPONSE THRESHOLDS ---
    RESPONSE_THRESHOLDS = {
        "excellent": 100,
        "good": 50,
        "moderate": 0
    }
    
    def __init__(self, data_path: str, output_dir: str, replace_outliers: bool = False):
        """
        Initialize analyzer with data path and output directory.
        
        Parameters
        ----------
        data_path : str
            Path to Excel file with OFT data
        output_dir : str
            Directory for saving outputs
        replace_outliers : bool
            Whether to replace outliers with group mean (default: False)
        """
        self.data_path = data_path
        self.output_dir = output_dir
        self.replace_outliers = replace_outliers
        
        # Create output directories
        self.dirs = {
            'figures': os.path.join(output_dir, 'figures'),
            'tables': os.path.join(output_dir, 'tables'),
            'panels': os.path.join(output_dir, 'figure_panels'),
            'individual': os.path.join(output_dir, 'individual_plots'),
            'supplementary': os.path.join(output_dir, 'supplementary_tables'),
            'logs': os.path.join(output_dir, 'logs')
        }
        
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Initialize outlier log as empty DataFrame
        self.outlier_log = pd.DataFrame()
        
        # Load and process data
        self._load_and_process_data()
        
        # Detect outliers (replace only if flag is True)
        self._detect_and_replace_outliers(replace=self.replace_outliers)
        
        # Initialize results storage
        self.statistical_results = []
        self.individual_stats = None
        self.anova_results = {}
        self.correlation_results = {}
        self.all_stats_detailed = []
        
        # Store primary axes for panel saving
        self.figure_axes = {}
        
    # --- FILE HANDLING: SUPPORTS CSV AND EXCEL ---
    def _load_and_process_data(self):
        """Load Excel or CSV data and convert to long format."""
        print("\n" + "=" * 60)
        print("LOADING AND PROCESSING DATA")
        print("=" * 60)
        
        # Check file extension
        file_ext = os.path.splitext(self.data_path)[1].lower()
        
        try:
            if file_ext == '.csv':
                df = pd.read_csv(self.data_path)
                print(f"✓ Loaded CSV file: {self.data_path}")
            elif file_ext in ['.xlsx', '.xls']:
                df = pd.read_excel(self.data_path)
                print(f"✓ Loaded Excel file: {self.data_path}")
            else:
                raise ValueError(f"Unsupported file format: {file_ext}. Please use .csv, .xlsx, or .xls")
        except Exception as e:
            print(f"❌ ERROR: Could not load data file: {e}")
            sys.exit(1)
        
        # Melt to long format
        id_vars = ['Parameter']
        value_vars = [col for col in df.columns if col != 'Parameter']
        
        self.long_df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name='Animal_Condition',
            value_name='Value'
        )
        
        # Parse AnimalID, Group, and Condition
        def parse_animal_condition(ac_str):
            """Parse animal-condition string into components."""
            ac_str = str(ac_str)
            
            # Extract condition (Pre, Stim, Post)
            if '_Pre' in ac_str:
                condition = 'Pre'
                animal = ac_str.replace('_Pre', '')
            elif '_Stim' in ac_str:
                condition = 'Stim'
                animal = ac_str.replace('_Stim', '')
            elif '_Post' in ac_str:
                condition = 'Post'
                animal = ac_str.replace('_Post', '')
            else:
                return None, None, None
            
            # Determine group
            if animal.startswith('PD') or animal == 'Rat9':
                group = 'PD'
            elif animal.startswith('CO'):
                group = 'CO'
            else:
                group = 'Unknown'
            
            return animal, group, condition
        
        # Apply parsing
        parsed = self.long_df['Animal_Condition'].apply(parse_animal_condition)
        self.long_df['AnimalID'] = [p[0] for p in parsed]
        self.long_df['Group'] = [p[1] for p in parsed]
        self.long_df['Condition'] = [p[2] for p in parsed]
        
        # Drop rows with parsing errors
        self.long_df = self.long_df.dropna(subset=['AnimalID', 'Group', 'Condition'])
        
        # Convert Value to numeric
        self.long_df['Value'] = pd.to_numeric(self.long_df['Value'], errors='coerce')
        
        # Filter to parameters of interest
        self.long_df = self.long_df[self.long_df['Parameter'].isin(self.PARAMETERS_OF_INTEREST)]
        
        # Create wide format for easier analysis
        self.wide_df = self.long_df.pivot_table(
            index=['AnimalID', 'Group', 'Condition'],
            columns='Parameter',
            values='Value'
        ).reset_index()
        
        # Print summary
        print(f"Total animals: {len(self.wide_df['AnimalID'].unique())}")
        print(f"PD animals: {len(self.wide_df[self.wide_df['Group'] == 'PD']['AnimalID'].unique())}")
        print(f"Control animals: {len(self.wide_df[self.wide_df['Group'] == 'CO']['AnimalID'].unique())}")
        print(f"Parameters: {', '.join(self.PARAMETERS_OF_INTEREST)}")
    
    # --- ROBUST OUTLIER DETECTION WITH SAFE LOGGING ---
    def _detect_and_replace_outliers(self, method: str = 'mad', threshold: float = 3.5, replace: bool = False):
        """
        Detect and optionally replace outliers with group mean.
        
        Parameters
        ----------
        method : str
            Method for outlier detection ('iqr', 'mad', or 'zscore')
        threshold : float
            Threshold for outlier detection
        replace : bool
            Whether to replace outliers (default: False)
        """
        print("\n" + "=" * 60)
        print("DETECTING OUTLIERS")
        print("=" * 60)
        print(f"Method: {method.upper()}, Threshold: {threshold}, Replace: {replace}")
        
        outlier_log = []
        
        # Work on a copy
        self.original_df = self.long_df.copy()
        self.cleaned_df = self.long_df.copy()
        
        # Detect outliers per Group × Condition × Parameter
        for group in self.GROUPS:
            for condition in self.CONDITIONS:
                for param in self.PARAMETERS_OF_INTEREST:
                    # Get subset
                    mask = ((self.cleaned_df['Group'] == group) & 
                           (self.cleaned_df['Condition'] == condition) &
                           (self.cleaned_df['Parameter'] == param))
                    
                    subset = self.cleaned_df[mask].copy()
                    
                    if len(subset) < 4:  # Too few for reliable outlier detection
                        continue
                    
                    values = subset['Value'].values
                    
                    # Detect outliers using selected method
                    outlier_mask = np.zeros(len(values), dtype=bool)
                    
                    if method == 'iqr':
                        q1 = np.percentile(values, 25)
                        q3 = np.percentile(values, 75)
                        iqr = q3 - q1
                        lower_bound = q1 - threshold * iqr
                        upper_bound = q3 + threshold * iqr
                        outlier_mask = (values < lower_bound) | (values > upper_bound)
                        
                    elif method == 'mad':
                        # MAD-based robust Z-score
                        median = np.median(values)
                        mad = np.median(np.abs(values - median))
                        if mad > 0:
                            modified_z_scores = 0.6745 * (values - median) / mad
                            outlier_mask = np.abs(modified_z_scores) > threshold
                        else:
                            # If MAD is zero, use IQR as fallback
                            q1 = np.percentile(values, 25)
                            q3 = np.percentile(values, 75)
                            iqr = q3 - q1
                            lower_bound = q1 - 3 * iqr
                            upper_bound = q3 + 3 * iqr
                            outlier_mask = (values < lower_bound) | (values > upper_bound)
                            
                    elif method == 'zscore':
                        mean = np.mean(values)
                        std = np.std(values)
                        if std > 0:
                            z_scores = np.abs(values - mean) / std
                            outlier_mask = z_scores > threshold
                    
                    # Log outliers regardless of replacement setting
                    if np.any(outlier_mask):
                        group_mean = np.mean(values[~outlier_mask]) if np.any(~outlier_mask) else np.mean(values)
                        
                        outlier_indices = subset.index[outlier_mask]
                        for idx in outlier_indices:
                            original_value = self.long_df.loc[idx, 'Value']
                            
                            # Log the detection
                            log_entry = {
                                'AnimalID': subset.loc[idx, 'AnimalID'],
                                'Group': group,
                                'Condition': condition,
                                'Parameter': param,
                                'Original_Value': original_value,
                                'Detection_Method': method,
                                'Threshold': threshold,
                                'Z_score': modified_z_scores[list(subset.index).index(idx)] if method == 'mad' and 'modified_z_scores' in locals() else np.nan,
                                'Replaced': replace
                            }
                            
                            # Replace if requested
                            if replace:
                                self.cleaned_df.loc[idx, 'Value'] = group_mean
                                log_entry['Replacement_Value'] = group_mean
                                print(f"  ⚠ Outlier replaced: {group} - {param} - {condition} - "
                                      f"{subset.loc[idx, 'AnimalID']}: {original_value:.2f} → {group_mean:.2f}")
                            else:
                                print(f"  ⚠ Outlier detected (NOT replaced): {group} - {param} - {condition} - "
                                      f"{subset.loc[idx, 'AnimalID']}: {original_value:.2f}")
                            
                            outlier_log.append(log_entry)
        
        # Save outlier log
        if outlier_log:
            self.outlier_log = pd.DataFrame(outlier_log)
            log_path = os.path.join(self.dirs['logs'], 'outlier_detection_log.csv')
            self.outlier_log.to_csv(log_path, index=False)
            print(f"\n✓ Outlier log saved to: {log_path}")
            print(f"  Total outliers detected: {len(self.outlier_log)}")
            
            # Safe counting of replaced outliers
            if 'Replaced' in self.outlier_log.columns:
                replaced_count = self.outlier_log['Replaced'].sum()
                print(f"  Outliers replaced: {replaced_count}")
        else:
            self.outlier_log = pd.DataFrame()
            print("\n  No outliers detected.")
        
        # Update wide_df with cleaned values only if replacement was requested
        if replace and len(self.outlier_log) > 0 and 'Replaced' in self.outlier_log.columns:
            self.wide_df_original = self.wide_df.copy()
            
            replaced_mask = self.outlier_log['Replaced'] == True
            if replaced_mask.any():
                for _, row in self.outlier_log[replaced_mask].iterrows():
                    animal = row['AnimalID']
                    condition = row['Condition']
                    param = row['Parameter']
                    new_value = row['Replacement_Value']
                    
                    mask = ((self.wide_df['AnimalID'] == animal) & 
                           (self.wide_df['Condition'] == condition))
                    self.wide_df.loc[mask, param] = new_value
    
    # =========================================================
    # STATISTICAL UTILITIES
    # =========================================================
    
    def compute_95ci(self, data: pd.Series) -> Tuple[float, float]:
        """
        Compute 95% confidence interval for a series.
        
        Parameters
        ----------
        data : pd.Series
            Input data
            
        Returns
        -------
        Tuple[float, float]
            Lower and upper bounds of 95% CI
        """
        data = data.dropna()
        n = len(data)
        if n < 2:
            return (np.nan, np.nan)
        
        mean = data.mean()
        sem = data.std() / np.sqrt(n)
        t_critical = t.ppf(0.975, df=n-1)
        ci = t_critical * sem
        
        return (mean - ci, mean + ci)
    
    # --- SCIENTIFIC UPGRADE 2: HEDGES' G INSTEAD OF COHEN'S D ---
    def compute_cohens_d_independent(self, group1: pd.Series, group2: pd.Series) -> float:
        """
        Compute Cohen's d for independent samples (kept for internal use).
        
        Parameters
        ----------
        group1, group2 : pd.Series
            Data for two groups
            
        Returns
        -------
        float
            Cohen's d effect size
        """
        group1 = group1.dropna()
        group2 = group2.dropna()
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = group1.mean(), group2.mean()
        var1, var2 = group1.var(), group2.var()
        
        # Pooled standard deviation
        pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_sd == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_sd
    
    def compute_hedges_g_independent(self, group1: pd.Series, group2: pd.Series) -> float:
        """
        Compute Hedges' g for independent samples (bias-corrected Cohen's d).
        
        Parameters
        ----------
        group1, group2 : pd.Series
            Data for two groups
            
        Returns
        -------
        float
            Hedges' g effect size
        """
        group1 = group1.dropna()
        group2 = group2.dropna()
        
        n1, n2 = len(group1), len(group2)
        N = n1 + n2
        
        # First compute Cohen's d
        d = self.compute_cohens_d_independent(group1, group2)
        
        # Apply bias correction for Hedges' g
        # Correction factor: J = 1 - 3/(4*df - 1) where df = n1 + n2 - 2
        df = N - 2
        correction_factor = 1 - (3 / (4 * df - 1))
        g = d * correction_factor
        
        return g
    
    def compute_cohens_d_paired(self, pre: pd.Series, post: pd.Series) -> float:
        """
        Compute Cohen's d for paired samples (kept for internal use).
        
        Parameters
        ----------
        pre, post : pd.Series
            Paired measurements
        """
        # Align data
        paired_data = pd.DataFrame({'pre': pre, 'post': post}).dropna()
        if len(paired_data) < 2:
            return np.nan
        
        differences = paired_data['post'] - paired_data['pre']
        mean_diff = differences.mean()
        sd_diff = differences.std()
        
        if sd_diff == 0:
            return 0.0
        
        return mean_diff / sd_diff
    
    def compute_hedges_g_paired(self, pre: pd.Series, post: pd.Series) -> float:
        """
        Compute Hedges' g for paired samples (bias-corrected Cohen's d).
        
        Parameters
        ----------
        pre, post : pd.Series
            Paired measurements
        """
        # Align data
        paired_data = pd.DataFrame({'pre': pre, 'post': post}).dropna()
        if len(paired_data) < 2:
            return np.nan
        
        n = len(paired_data)
        
        # First compute Cohen's d for paired samples
        d = self.compute_cohens_d_paired(pre, post)
        
        # Apply bias correction for Hedges' g
        # For paired designs, df = n - 1
        df = n - 1
        correction_factor = 1 - (3 / (4 * df - 1))
        g = d * correction_factor
        
        return g
    
    def check_normality(self, data: pd.Series, group_name: str, param: str) -> bool:
        """
        Check normality using Shapiro-Wilk test.
        
        Parameters
        ----------
        data : pd.Series
            Data to test
        group_name : str
            Group name for reporting
        param : str
            Parameter name for reporting
            
        Returns
        -------
        bool
            True if normal (p > 0.05), False otherwise
        """
        data = data.dropna()
        if len(data) < 3:
            return True
        
        stat, p = shapiro(data)
        if p < 0.05:
            print(f"  ⚠ Normality violation: {group_name}, {param} (p={p:.4f})")
            return False
        return True
    
    # =========================================================
    # REPEATED MEASURES ANOVA
    # =========================================================
    
    def run_rm_anova(self, parameter: str) -> Dict:
        """
        Run 2×3 repeated measures ANOVA.
        
        Parameters
        ----------
        parameter : str
            Parameter to analyze
            
        Returns
        -------
        Dict
            ANOVA results including partial eta squared
        """
        print(f"\n  Running RM-ANOVA for {parameter}...")
        
        # Prepare data for ANOVA
        anova_data = self.wide_df[['AnimalID', 'Group', 'Condition', parameter]].copy()
        anova_data = anova_data.dropna()
        
        # Pivot to wide format for AnovaRM
        pivot_data = anova_data.pivot_table(
            index=['AnimalID', 'Group'],
            columns='Condition',
            values=parameter
        ).reset_index()
        
        # Ensure we have all conditions
        required_cols = ['AnimalID', 'Group'] + self.CONDITIONS
        if not all(col in pivot_data.columns for col in required_cols):
            return {
                'parameter': parameter,
                'error': 'Missing conditions',
                'n_pd': len(pivot_data[pivot_data['Group'] == 'PD']),
                'n_co': len(pivot_data[pivot_data['Group'] == 'CO'])
            }
        
        # Try AnovaRM first
        try:
            # Reshape for AnovaRM
            anova_long = pd.melt(
                pivot_data,
                id_vars=['AnimalID', 'Group'],
                value_vars=self.CONDITIONS,
                var_name='Condition',
                value_name='Value'
            )
            
            # Run ANOVA
            aov = AnovaRM(anova_long, 'Value', 'AnimalID', within=['Condition'], between=['Group'])
            result = aov.fit()
            
            # Extract results
            anova_table = result.anova_table
            
            results = {
                'parameter': parameter,
                'n_pd': len(pivot_data[pivot_data['Group'] == 'PD']),
                'n_co': len(pivot_data[pivot_data['Group'] == 'CO']),
                'method': 'AnovaRM'
            }
            
            # Extract effects and compute partial eta squared
            for effect in anova_table.index:
                if 'Group' in effect:
                    results['group_F'] = anova_table.loc[effect, 'F Value']
                    results['group_p'] = anova_table.loc[effect, 'Pr > F']
                    results['group_df'] = anova_table.loc[effect, 'Num DF']
                    results['group_den_df'] = anova_table.loc[effect, 'Den DF']
                    
                    # --- SCIENTIFIC UPGRADE 1: PARTIAL ETA SQUARED ---
                    # For between-subjects effects, we need SS_effect and SS_error
                    # Extract from the ANOVA table if available
                    if 'SS' in anova_table.columns:
                        ss_effect = anova_table.loc[effect, 'SS']
                        ss_error = anova_table.loc['Error', 'SS'] if 'Error' in anova_table.index else np.nan
                        if not np.isnan(ss_effect) and not np.isnan(ss_error) and (ss_effect + ss_error) > 0:
                            results['group_eta_sq_partial'] = ss_effect / (ss_effect + ss_error)
                    
                elif 'Condition' in effect and 'Group' not in effect:
                    results['condition_F'] = anova_table.loc[effect, 'F Value']
                    results['condition_p'] = anova_table.loc[effect, 'Pr > F']
                    results['condition_df'] = (anova_table.loc[effect, 'Num DF'], 
                                              anova_table.loc[effect, 'Den DF'])
                    
                    # --- SCIENTIFIC UPGRADE 1: PARTIAL ETA SQUARED ---
                    if 'SS' in anova_table.columns:
                        ss_effect = anova_table.loc[effect, 'SS']
                        ss_error = anova_table.loc['Error', 'SS'] if 'Error' in anova_table.index else np.nan
                        if not np.isnan(ss_effect) and not np.isnan(ss_error) and (ss_effect + ss_error) > 0:
                            results['condition_eta_sq_partial'] = ss_effect / (ss_effect + ss_error)
                    
                elif 'Group:Condition' in effect or 'Condition:Group' in effect:
                    results['interaction_F'] = anova_table.loc[effect, 'F Value']
                    results['interaction_p'] = anova_table.loc[effect, 'Pr > F']
                    results['interaction_df'] = (anova_table.loc[effect, 'Num DF'], 
                                                anova_table.loc[effect, 'Den DF'])
                    
                    # --- SCIENTIFIC UPGRADE 1: PARTIAL ETA SQUARED ---
                    if 'SS' in anova_table.columns:
                        ss_effect = anova_table.loc[effect, 'SS']
                        ss_error = anova_table.loc['Error', 'SS'] if 'Error' in anova_table.index else np.nan
                        if not np.isnan(ss_effect) and not np.isnan(ss_error) and (ss_effect + ss_error) > 0:
                            results['interaction_eta_sq_partial'] = ss_effect / (ss_effect + ss_error)
            
            return results
            
        except Exception as e:
            print(f"    AnovaRM failed: {e}, trying MixedLM...")
            
            # Fallback to MixedLM (partial eta squared not available)
            try:
                mlm_data = anova_long.copy()
                mlm_data['Group_num'] = (mlm_data['Group'] == 'PD').astype(int)
                
                # Create condition dummies
                for cond in self.CONDITIONS[1:]:
                    mlm_data[f'Cond_{cond}'] = (mlm_data['Condition'] == cond).astype(int)
                
                # Fit model
                model = mixedlm("Value ~ Group_num * C(Condition, Treatment(reference='Pre'))", 
                               mlm_data, groups=mlm_data['AnimalID'])
                result = model.fit()
                
                results = {
                    'parameter': parameter,
                    'n_pd': len(pivot_data[pivot_data['Group'] == 'PD']),
                    'n_co': len(pivot_data[pivot_data['Group'] == 'CO']),
                    'method': 'MixedLM'
                }
                
                # Extract effects (partial eta squared not available for MixedLM)
                results['group_coef'] = result.params.get('Group_num', np.nan)
                results['group_p'] = result.pvalues.get('Group_num', np.nan)
                
                # Condition effects
                cond_ps = [result.pvalues.get(f'Cond_{cond}', np.nan) for cond in self.CONDITIONS[1:]]
                results['condition_p'] = np.nanmin(cond_ps) if cond_ps else np.nan
                
                # Interaction effects
                interaction_terms = [f'Group_num:Cond_{cond}' for cond in self.CONDITIONS[1:]]
                interaction_ps = [result.pvalues.get(term, np.nan) for term in interaction_terms]
                results['interaction_p'] = np.nanmin(interaction_ps) if interaction_ps else np.nan
                
                return results
                
            except Exception as e2:
                print(f"    MixedLM also failed: {e2}")
                return {
                    'parameter': parameter,
                    'error': str(e2),
                    'n_pd': len(pivot_data[pivot_data['Group'] == 'PD']),
                    'n_co': len(pivot_data[pivot_data['Group'] == 'CO'])
                }
    
    def run_all_anovas(self):
        """Run RM-ANOVA for all parameters."""
        print("\n" + "=" * 60)
        print("RUNNING REPEATED MEASURES ANOVA")
        print("=" * 60)
        
        for param in self.PARAMETERS_OF_INTEREST:
            result = self.run_rm_anova(param)
            self.anova_results[param] = result
            
            # Print summary with partial eta squared
            if 'error' in result:
                print(f"\n  {param}: ERROR - {result['error']}")
            else:
                print(f"\n  {param}:")
                if 'group_p' in result:
                    print(f"    Group effect: F={result.get('group_F', np.nan):.2f}, "
                          f"p={result['group_p']:.4f}", end='')
                    if 'group_eta_sq_partial' in result:
                        print(f", eta²p={result['group_eta_sq_partial']:.3f}")
                    else:
                        print()
                if 'condition_p' in result:
                    print(f"    Condition effect: p={result['condition_p']:.4f}", end='')
                    if 'condition_eta_sq_partial' in result:
                        print(f", eta²p={result['condition_eta_sq_partial']:.3f}")
                    else:
                        print()
                if 'interaction_p' in result:
                    print(f"    Interaction: p={result['interaction_p']:.4f}", end='')
                    if 'interaction_eta_sq_partial' in result:
                        print(f", eta²p={result['interaction_eta_sq_partial']:.3f}")
                    else:
                        print()
    
    # =========================================================
    # POST HOC COMPARISONS
    # =========================================================
    
    def run_posthoc_comparisons(self, parameter: str) -> pd.DataFrame:
        """
        Run post hoc pairwise comparisons after significant interaction.
        
        Parameters
        ----------
        parameter : str
            Parameter to analyze
            
        Returns
        -------
        pd.DataFrame
            Post hoc results
        """
        comparisons = []
        
        # Within-group comparisons
        for group in self.GROUPS:
            group_data = self.wide_df[self.wide_df['Group'] == group]
            
            # Pre vs Stim
            pre_stim = self.analyze_within_group(parameter, group, 'Pre', 'Stim')
            comparisons.append(pre_stim)
            
            # Pre vs Post
            pre_post = self.analyze_within_group(parameter, group, 'Pre', 'Post')
            comparisons.append(pre_post)
            
            # Stim vs Post
            stim_post = self.analyze_within_group(parameter, group, 'Stim', 'Post')
            comparisons.append(stim_post)
        
        # Between-group comparisons at each condition
        for condition in self.CONDITIONS:
            between = self.analyze_between_groups(parameter, condition)
            comparisons.append(between)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(comparisons)
        
        return results_df
    
    # =========================================================
    # CORRELATION ANALYSIS (SEPARATE BY GROUP)
    # =========================================================
    
    def run_correlation_analysis(self) -> Dict:
        """
        Run correlation analysis separately for each group.
        Now with proper group separation, small sample warnings, and Spearman fallback.
        """
        print("\n" + "=" * 60)
        print("RUNNING CORRELATION ANALYSIS (SEPARATE BY GROUP)")
        print("=" * 60)
        
        correlation_results = {}
        
        for group in self.GROUPS:
            print(f"\n  {group} group:")
            
            # Get stimulation data for this group
            group_data = self.wide_df[
                (self.wide_df['Group'] == group) & 
                (self.wide_df['Condition'] == 'Stim')
            ]
            
            # Parameters for correlation
            params = ['TotalDistance_cm', 'Time PercentCentral', 'MeanSpeed_cm_s', 'TimeMoving_s']
            corr_data = group_data[params].dropna()
            n = len(corr_data)
            
            if n > 1:
                print(f"    N={n} animals")
                
                # Small sample warning
                if n < 10:
                    print(f"    ⚠ WARNING: Small sample size (n={n}) - interpret correlations cautiously")
                
                # Pearson correlation
                corr_matrix_pearson = corr_data.corr(method='pearson')
                
                # Spearman correlation for small samples or non-normal data
                corr_matrix_spearman = corr_data.corr(method='spearman')
                
                # Format for output
                correlations = []
                for i, p1 in enumerate(params):
                    for j, p2 in enumerate(params):
                        if i < j:  # Upper triangle only
                            pearson_r = corr_matrix_pearson.loc[p1, p2]
                            spearman_r = corr_matrix_spearman.loc[p1, p2]
                            
                            # Calculate p-values for Pearson
                            pearson_p = pearsonr(corr_data[p1], corr_data[p2])[1]
                            spearman_p = spearmanr(corr_data[p1], corr_data[p2])[1]
                            
                            correlations.append({
                                'Parameter1': p1,
                                'Parameter2': p2,
                                'Pearson_r': pearson_r,
                                'Pearson_p': pearson_p,
                                'Spearman_r': spearman_r,
                                'Spearman_p': spearman_p,
                                'N': n
                            })
                            
                            sig_marker = " ***" if pearson_p < 0.05 else ""
                            print(f"    {p1} vs {p2}: r={pearson_r:.3f}{sig_marker} (p={pearson_p:.4f})")
                            if n < 10:
                                print(f"      Spearman: ρ={spearman_r:.3f} (p={spearman_p:.4f})")
                
                correlation_results[group] = pd.DataFrame(correlations)
                
                # Save to CSV
                csv_path = os.path.join(
                    self.dirs['supplementary'], 
                    f'Correlations_{group}_Stim.csv'
                )
                correlation_results[group].to_csv(csv_path, index=False)
            else:
                print(f"    Insufficient data for correlation")
                correlation_results[group] = pd.DataFrame()
        
        self.correlation_results = correlation_results
        return correlation_results
    
    # =========================================================
    # STATISTICAL ANALYSES
    # =========================================================
    
    def analyze_between_groups(self, parameter: str, condition: str) -> Dict:
        """
        Compare PD vs Control for a specific parameter and condition.
        Now uses Hedges' g instead of Cohen's d.
        """
        # Get data
        pd_data = self.wide_df[
            (self.wide_df['Group'] == 'PD') & 
            (self.wide_df['Condition'] == condition)
        ][parameter].dropna()
        
        co_data = self.wide_df[
            (self.wide_df['Group'] == 'CO') & 
            (self.wide_df['Condition'] == condition)
        ][parameter].dropna()
        
        result = {
            'parameter': parameter,
            'comparison': f'PD_vs_CO_{condition}',
            'condition': condition,
            'n_pd': len(pd_data),
            'n_co': len(co_data),
            'mean_pd': pd_data.mean() if len(pd_data) > 0 else np.nan,
            'mean_co': co_data.mean() if len(co_data) > 0 else np.nan,
            'sd_pd': pd_data.std() if len(pd_data) > 0 else np.nan,
            'sd_co': co_data.std() if len(co_data) > 0 else np.nan,
            'ci_pd_lower': self.compute_95ci(pd_data)[0] if len(pd_data) > 0 else np.nan,
            'ci_pd_upper': self.compute_95ci(pd_data)[1] if len(pd_data) > 0 else np.nan,
            'ci_co_lower': self.compute_95ci(co_data)[0] if len(co_data) > 0 else np.nan,
            'ci_co_upper': self.compute_95ci(co_data)[1] if len(co_data) > 0 else np.nan
        }
        
        # Check normality
        normal_pd = True
        normal_co = True
        if len(pd_data) >= 3:
            normal_pd = self.check_normality(pd_data, 'PD', parameter)
        if len(co_data) >= 3:
            normal_co = self.check_normality(co_data, 'CO', parameter)
        
        # Run appropriate test
        if len(pd_data) >= 2 and len(co_data) >= 2:
            t_stat, p_value = stats.ttest_ind(pd_data, co_data, equal_var=False)
            result['test_used'] = 't-test (Welch)'
            result['statistic'] = t_stat
            result['p_value'] = p_value
            
            # Also run non-parametric if normality violated
            if not (normal_pd and normal_co):
                u_stat, u_p = mannwhitneyu(pd_data, co_data, alternative='two-sided')
                result['nonparametric_statistic'] = u_stat
                result['nonparametric_p_value'] = u_p
                result['nonparametric_test'] = 'Mann-Whitney U'
            
            # --- SCIENTIFIC UPGRADE 2: HEDGES' G INSTEAD OF COHEN'S D ---
            # Effect size using Hedges' g
            result['effect_size'] = self.compute_hedges_g_independent(pd_data, co_data)
            
        return result
    
    def analyze_within_group(self, parameter: str, group: str, 
                            cond1: str, cond2: str) -> Dict:
        """
        Compare two conditions within a group.
        Now uses Hedges' g instead of Cohen's d.
        """
        # Get data and align by animal
        data1 = self.wide_df[
            (self.wide_df['Group'] == group) & 
            (self.wide_df['Condition'] == cond1)
        ][['AnimalID', parameter]].set_index('AnimalID')
        
        data2 = self.wide_df[
            (self.wide_df['Group'] == group) & 
            (self.wide_df['Condition'] == cond2)
        ][['AnimalID', parameter]].set_index('AnimalID')
        
        # Merge and drop missing
        paired = data1.join(data2, lsuffix='_1', rsuffix='_2').dropna()
        
        result = {
            'parameter': parameter,
            'comparison': f'{group}_{cond1}_vs_{cond2}',
            'group': group,
            'condition1': cond1,
            'condition2': cond2,
            'n': len(paired),
            'mean_1': paired[f'{parameter}_1'].mean() if len(paired) > 0 else np.nan,
            'mean_2': paired[f'{parameter}_2'].mean() if len(paired) > 0 else np.nan,
            'sd_1': paired[f'{parameter}_1'].std() if len(paired) > 0 else np.nan,
            'sd_2': paired[f'{parameter}_2'].std() if len(paired) > 0 else np.nan,
            'ci_1_lower': self.compute_95ci(paired[f'{parameter}_1'])[0] if len(paired) > 0 else np.nan,
            'ci_1_upper': self.compute_95ci(paired[f'{parameter}_1'])[1] if len(paired) > 0 else np.nan,
            'ci_2_lower': self.compute_95ci(paired[f'{parameter}_2'])[0] if len(paired) > 0 else np.nan,
            'ci_2_upper': self.compute_95ci(paired[f'{parameter}_2'])[1] if len(paired) > 0 else np.nan
        }
        
        if len(paired) >= 2:
            # Check normality of differences
            differences = paired[f'{parameter}_2'] - paired[f'{parameter}_1']
            normal_diff = True
            if len(differences) >= 3:
                normal_diff = self.check_normality(differences, f'{group} diff', parameter)
            
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(
                paired[f'{parameter}_1'], 
                paired[f'{parameter}_2']
            )
            result['test_used'] = 'Paired t-test'
            result['statistic'] = t_stat
            result['p_value'] = p_value
            
            # Run non-parametric if normality violated
            if not normal_diff:
                w_stat, w_p = wilcoxon(paired[f'{parameter}_1'], paired[f'{parameter}_2'])
                result['nonparametric_statistic'] = w_stat
                result['nonparametric_p_value'] = w_p
                result['nonparametric_test'] = 'Wilcoxon'
            
            # --- SCIENTIFIC UPGRADE 2: HEDGES' G INSTEAD OF COHEN'S D ---
            # Effect size using Hedges' g
            result['effect_size'] = self.compute_hedges_g_paired(
                paired[f'{parameter}_1'],
                paired[f'{parameter}_2']
            )
        
        return result
    
    # --- PROPER STATISTICAL HIERARCHY ---
    def perform_full_analysis(self) -> pd.DataFrame:
        """
        Perform complete statistical analysis with proper hierarchy.
        Now includes decision tree logging and post-hoc only after significant interaction.
        """
        print("\n" + "=" * 60)
        print("PERFORMING STATISTICAL ANALYSIS")
        print("=" * 60)
        
        # First run ANOVA for all parameters
        self.run_all_anovas()
        
        results = []
        decision_log = []
        
        # For each parameter, apply hierarchical decision tree
        for param in self.PARAMETERS_OF_INTEREST:
            print(f"\n  {'='*40}")
            print(f"  DECISION TREE FOR: {param}")
            print(f"  {'='*40}")
            
            anova_result = self.anova_results.get(param, {})
            
            if 'error' in anova_result:
                print(f"  ⚠ ANOVA failed: {anova_result['error']}")
                decision_log.append({
                    'parameter': param,
                    'interaction_p': np.nan,
                    'interaction_sig': False,
                    'post_hoc_performed': False,
                    'reason': 'ANOVA failed'
                })
                continue
            
            interaction_p = anova_result.get('interaction_p', 1.0)
            group_p = anova_result.get('group_p', 1.0)
            condition_p = anova_result.get('condition_p', 1.0)
            
            interaction_sig = interaction_p < 0.05 if not pd.isna(interaction_p) else False
            
            print(f"  Interaction p = {interaction_p:.4f} {'(significant)' if interaction_sig else '(not significant)'}")
            print(f"  Group main effect p = {group_p:.4f}")
            print(f"  Condition main effect p = {condition_p:.4f}")
            
            # Decision: Only run post-hoc if interaction is significant
            if interaction_sig:
                print(f"  ✅ Interaction significant → running post-hoc comparisons")
                
                # Run post-hoc for this parameter
                posthoc_results = self.run_posthoc_comparisons(param)
                
                # Add to results
                for _, row in posthoc_results.iterrows():
                    results.append(row.to_dict())
                
                decision_log.append({
                    'parameter': param,
                    'interaction_p': interaction_p,
                    'interaction_sig': True,
                    'post_hoc_performed': True,
                    'reason': 'Significant interaction'
                })
            else:
                print(f"  ❌ Interaction not significant → post-hoc skipped")
                print(f"  Interpreting main effects only")
                
                decision_log.append({
                    'parameter': param,
                    'interaction_p': interaction_p,
                    'interaction_sig': False,
                    'post_hoc_performed': False,
                    'reason': 'Non-significant interaction'
                })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Apply FDR correction to secondary parameters only
        if len(results_df) > 0 and 'p_value' in results_df.columns:
            # Separate primary and secondary
            primary_mask = results_df['parameter'] == 'TotalDistance_cm'
            secondary_mask = ~primary_mask
            
            # Apply FDR to secondary outcomes only
            if secondary_mask.any():
                secondary_p = results_df.loc[secondary_mask, 'p_value'].dropna().values
                if len(secondary_p) > 0:
                    rejected, p_corrected, _, _ = multipletests(
                        secondary_p, alpha=0.05, method='fdr_bh'
                    )
                    
                    secondary_indices = results_df[secondary_mask & results_df['p_value'].notna()].index
                    results_df.loc[secondary_indices, 'p_corrected'] = p_corrected
                    results_df.loc[secondary_indices, 'significant'] = rejected
        
        self.statistical_results = results_df
        
        # Save decision log
        decision_df = pd.DataFrame(decision_log)
        decision_path = os.path.join(self.dirs['logs'], 'statistical_decision_tree.csv')
        decision_df.to_csv(decision_path, index=False)
        
        # Print decision tree summary
        print("\n" + "=" * 60)
        print("STATISTICAL DECISION TREE SUMMARY")
        print("=" * 60)
        for entry in decision_log:
            print(f"\n  {entry['parameter']}:")
            print(f"    Interaction p = {entry['interaction_p']:.4f}")
            print(f"    Post-hoc performed: {entry['post_hoc_performed']}")
            print(f"    Reason: {entry['reason']}")
        
        # Save detailed results
        self._save_detailed_statistics()
        
        return results_df
    
    def _save_detailed_statistics(self):
        """Save detailed statistical results for supplementary materials."""
        if len(self.statistical_results) == 0:
            return
        
        # Save full results
        full_path = os.path.join(self.dirs['supplementary'], 'All_Statistical_Results_Detailed.csv')
        self.statistical_results.to_csv(full_path, index=False)
        
        # Create parameter-specific files
        for param in self.PARAMETERS_OF_INTEREST:
            param_results = self.statistical_results[self.statistical_results['parameter'] == param]
            if len(param_results) > 0:
                param_path = os.path.join(self.dirs['supplementary'], f'Statistics_{param}.csv')
                param_results.to_csv(param_path, index=False)
        
        # Save ANOVA results (including partial eta squared)
        anova_df = pd.DataFrame([
            {'Parameter': k, **{f'ANOVA_{key}': value for key, value in v.items() if key != 'parameter'}}
            for k, v in self.anova_results.items()
        ])
        if len(anova_df) > 0:
            anova_path = os.path.join(self.dirs['supplementary'], 'ANOVA_Results.csv')
            anova_df.to_csv(anova_path, index=False)
        
        # Create formatted tables for publication
        self._create_formatted_statistical_tables()
        
        print(f"✓ Detailed statistics saved to: {self.dirs['supplementary']}")
    
    def _create_formatted_statistical_tables(self):
        """Create formatted statistical tables for publication."""
        if len(self.statistical_results) == 0:
            return
        
        # Table 1: Between-group comparisons
        between = self.statistical_results[
            self.statistical_results['comparison'].str.contains('PD_vs_CO', na=False)
        ].copy()
        
        if len(between) > 0:
            between_table = between[['parameter', 'condition', 'n_pd', 'n_co', 
                                     'mean_pd', 'mean_co', 'sd_pd', 'sd_co',
                                     'statistic', 'p_value', 'p_corrected', 
                                     'significant', 'effect_size']].copy()
            
            # Format numbers
            for col in ['mean_pd', 'mean_co', 'sd_pd', 'sd_co']:
                between_table[col] = between_table[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'NA')
            
            between_table['p_value'] = between_table['p_value'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'NA')
            between_table['p_corrected'] = between_table['p_corrected'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'NA')
            
            # --- SCIENTIFIC UPGRADE 2: Update label to Hedges' g ---
            between_table['effect_size'] = between_table['effect_size'].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'NA')
            between_table['significant'] = between_table['significant'].apply(lambda x: 'Yes' if x else 'No')
            
            between_path = os.path.join(self.dirs['tables'], 'Table_Between_Group_Comparisons.csv')
            between_table.to_csv(between_path, index=False)
        
        # Table 2: Within-group comparisons (PD)
        pd_within = self.statistical_results[
            (self.statistical_results['group'] == 'PD') & 
            (~self.statistical_results['comparison'].str.contains('PD_vs_CO', na=False))
        ].copy()
        
        if len(pd_within) > 0:
            pd_table = pd_within[['parameter', 'condition1', 'condition2', 'n',
                                  'mean_1', 'mean_2', 'sd_1', 'sd_2',
                                  'statistic', 'p_value', 'p_corrected',
                                  'significant', 'effect_size']].copy()
            
            # Format numbers
            for col in ['mean_1', 'mean_2', 'sd_1', 'sd_2']:
                pd_table[col] = pd_table[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'NA')
            
            pd_table['p_value'] = pd_table['p_value'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'NA')
            pd_table['p_corrected'] = pd_table['p_corrected'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'NA')
            
            # --- SCIENTIFIC UPGRADE 2: Update label to Hedges' g ---
            pd_table['effect_size'] = pd_table['effect_size'].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'NA')
            pd_table['significant'] = pd_table['significant'].apply(lambda x: 'Yes' if x else 'No')
            
            pd_path = os.path.join(self.dirs['tables'], 'Table_PD_Within_Group_Comparisons.csv')
            pd_table.to_csv(pd_path, index=False)
        
        # Table 3: Within-group comparisons (Control)
        co_within = self.statistical_results[
            (self.statistical_results['group'] == 'CO') & 
            (~self.statistical_results['comparison'].str.contains('PD_vs_CO', na=False))
        ].copy()
        
        if len(co_within) > 0:
            co_table = co_within[['parameter', 'condition1', 'condition2', 'n',
                                  'mean_1', 'mean_2', 'sd_1', 'sd_2',
                                  'statistic', 'p_value', 'p_corrected',
                                  'significant', 'effect_size']].copy()
            
            # Format numbers
            for col in ['mean_1', 'mean_2', 'sd_1', 'sd_2']:
                co_table[col] = co_table[col].apply(lambda x: f'{x:.2f}' if pd.notna(x) else 'NA')
            
            co_table['p_value'] = co_table['p_value'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'NA')
            co_table['p_corrected'] = co_table['p_corrected'].apply(lambda x: f'{x:.4f}' if pd.notna(x) else 'NA')
            
            # --- SCIENTIFIC UPGRADE 2: Update label to Hedges' g ---
            co_table['effect_size'] = co_table['effect_size'].apply(lambda x: f'{x:.3f}' if pd.notna(x) else 'NA')
            co_table['significant'] = co_table['significant'].apply(lambda x: 'Yes' if x else 'No')
            
            co_path = os.path.join(self.dirs['tables'], 'Table_Control_Within_Group_Comparisons.csv')
            co_table.to_csv(co_path, index=False)
    
    def calculate_individual_statistics(self) -> pd.DataFrame:
        """
        Calculate individual animal statistics and responses.
        Now uses configurable response thresholds.
        """
        print("\n" + "=" * 60)
        print("CALCULATING INDIVIDUAL STATISTICS")
        print("=" * 60)
        
        individual_stats = []
        
        for animal in self.wide_df['AnimalID'].unique():
            animal_data = self.wide_df[self.wide_df['AnimalID'] == animal]
            group = animal_data['Group'].iloc[0]
            
            for param in self.PARAMETERS_OF_INTEREST:
                # Get values for each condition
                pre_vals = animal_data[animal_data['Condition'] == 'Pre'][param].values
                stim_vals = animal_data[animal_data['Condition'] == 'Stim'][param].values
                post_vals = animal_data[animal_data['Condition'] == 'Post'][param].values
                
                if len(pre_vals) > 0 and len(stim_vals) > 0 and len(post_vals) > 0:
                    pre = pre_vals[0]
                    stim = stim_vals[0]
                    post = post_vals[0]
                    
                    # Calculate percentage changes
                    if pre != 0 and not pd.isna(pre):
                        pct_change_stim = ((stim - pre) / abs(pre)) * 100
                        pct_change_post = ((post - pre) / abs(pre)) * 100
                        stim_vs_post = ((stim - post) / abs(post)) * 100 if post != 0 else np.nan
                    else:
                        pct_change_stim = np.nan
                        pct_change_post = np.nan
                        stim_vs_post = np.nan
                    
                    # --- SCIENTIFIC UPGRADE 4: Use configurable thresholds ---
                    # Classify response (for motor parameters) using RESPONSE_THRESHOLDS
                    if param in ['TotalDistance_cm', 'TimeMoving_s', 'MeanSpeed_cm_s']:
                        response = self._classify_motor_response(pre, stim)
                    elif param == 'Time PercentCentral':
                        response = self._classify_anxiety_response(pre, stim)
                    else:
                        response = 'N/A'
                    
                    individual_stats.append({
                        'AnimalID': animal,
                        'Group': group,
                        'Parameter': param,
                        'Pre': pre,
                        'Stim': stim,
                        'Post': post,
                        'Pct_Change_Stim_vs_Pre': pct_change_stim,
                        'Pct_Change_Post_vs_Pre': pct_change_post,
                        'Pct_Change_Stim_vs_Post': stim_vs_post,
                        'Response': response
                    })
        
        self.individual_df = pd.DataFrame(individual_stats)
        
        # Save to CSV
        csv_path = os.path.join(self.dirs['tables'], 'Individual_Animal_Results.csv')
        self.individual_df.to_csv(csv_path, index=False)
        print(f"✓ Individual results saved to: {csv_path}")
        
        # Save individual animal summaries
        self._save_individual_animal_summaries()
        
        return self.individual_df
    
    def _save_individual_animal_summaries(self):
        """Save individual animal plots and data summaries."""
        animals = self.wide_df['AnimalID'].unique()
        
        # Save individual data summaries
        summary_rows = []
        for animal in animals:
            animal_data = self.individual_df[self.individual_df['AnimalID'] == animal]
            group = animal_data['Group'].iloc[0] if len(animal_data) > 0 else 'Unknown'
            
            # Get average response across motor parameters
            motor_data = animal_data[animal_data['Parameter'].isin(['TotalDistance_cm', 'TimeMoving_s', 'MeanSpeed_cm_s'])]
            
            avg_stim_response = motor_data['Pct_Change_Stim_vs_Pre'].mean() if len(motor_data) > 0 else np.nan
            most_common_response = motor_data['Response'].mode().iloc[0] if len(motor_data) > 0 and len(motor_data['Response'].mode()) > 0 else 'N/A'
            
            summary_rows.append({
                'AnimalID': animal,
                'Group': group,
                'Average_Stim_Response_%': avg_stim_response,
                'Primary_Response': most_common_response
            })
        
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(self.dirs['individual'], 'Individual_Animal_Summary.csv')
        summary_df.to_csv(summary_path, index=False)
        
        # Save individual animal plots
        for animal in animals:
            fig, ax = plt.subplots(figsize=(8, 6))
            self._plot_individual_animal_summary(animal, ax)
            
            # Save plot
            plot_path = os.path.join(self.dirs['individual'], f'{animal}_response_profile.png')
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✓ Individual animal plots saved to: {self.dirs['individual']}")
    
    # --- SCIENTIFIC UPGRADE 4: Use configurable thresholds in classification ---
    def _classify_motor_response(self, pre: float, stim: float) -> str:
        """Classify motor response based on stimulation effect using configurable thresholds."""
        if pd.isna(pre) or pd.isna(stim) or pre == 0:
            return 'N/A'
        
        pct_change = ((stim - pre) / abs(pre)) * 100
        
        if pct_change > self.RESPONSE_THRESHOLDS["excellent"]:
            return 'EXCELLENT'
        elif pct_change > self.RESPONSE_THRESHOLDS["good"]:
            return 'GOOD'
        elif pct_change > self.RESPONSE_THRESHOLDS["moderate"]:
            return 'MODERATE'
        elif pct_change > -50:
            return 'SUPPRESSED'
        else:
            return 'ADVERSE'
    
    def _classify_anxiety_response(self, pre: float, stim: float) -> str:
        """Classify anxiety response based on center time changes using configurable thresholds."""
        if pd.isna(pre) or pd.isna(stim):
            return 'N/A'
        
        if pre == 0:
            return 'EXCELLENT' if stim > 0 else 'N/A'
        
        pct_change = ((stim - pre) / abs(pre)) * 100
        
        if pct_change > self.RESPONSE_THRESHOLDS["excellent"]:
            return 'EXCELLENT'
        elif pct_change > self.RESPONSE_THRESHOLDS["good"]:
            return 'GOOD'
        elif pct_change > self.RESPONSE_THRESHOLDS["moderate"]:
            return 'MODERATE'
        elif pct_change > -50:
            return 'REDUCED'
        else:
            return 'NEGATIVE'
    
    # =========================================================
    # VISUALIZATION
    # =========================================================
    
    def plot_individual_trajectories(self, ax: plt.Axes, parameter: str = 'TotalDistance_cm'):
        """
        Plot individual animal trajectories with group means.
        """
        # Pivot data for easier plotting
        pivot_data = self.wide_df.pivot_table(
            index=['AnimalID', 'Group'],
            columns='Condition',
            values=parameter
        ).reset_index()
        
        # Plot individual trajectories
        for _, row in pivot_data.iterrows():
            group = row['Group']
            color = self.COLORS.get(group, 'gray')
            alpha = 0.3 if group == 'PD' else 0.2
            
            x = [0, 1, 2]
            y = [row.get('Pre', np.nan), row.get('Stim', np.nan), row.get('Post', np.nan)]
            
            if not any(pd.isna(y)):
                ax.plot(x, y, color=color, alpha=alpha, linewidth=1, zorder=1)
        
        # Plot group means with 95% CI
        for group in self.GROUPS:
            group_data = pivot_data[pivot_data['Group'] == group]
            
            means = []
            cis_lower = []
            cis_upper = []
            
            for condition in self.CONDITIONS:
                values = group_data[condition].dropna()
                if len(values) > 0:
                    mean = values.mean()
                    ci_lower, ci_upper = self.compute_95ci(values)
                else:
                    mean = np.nan
                    ci_lower, ci_upper = np.nan, np.nan
                
                means.append(mean)
                cis_lower.append(ci_lower)
                cis_upper.append(ci_upper)
            
            color = self.COLORS.get(group, 'gray')
            
            # Plot mean line (only if we have valid means)
            if not any(pd.isna(means)):
                ax.plot([0, 1, 2], means, color=color, linewidth=3, 
                       label=f'{group} (n={len(group_data)})', zorder=10)
                
                # Add CI as shaded region (only where valid)
                valid_idx = ~np.isnan(cis_lower)
                if any(valid_idx):
                    x_vals = np.array([0, 1, 2])[valid_idx]
                    lower_vals = np.array(cis_lower)[valid_idx]
                    upper_vals = np.array(cis_upper)[valid_idx]
                    ax.fill_between(x_vals, lower_vals, upper_vals, 
                                   color=color, alpha=0.2, zorder=5)
                
                # Add points at means
                ax.scatter([0, 1, 2], means, color=color, s=100, zorder=20,
                          edgecolor='white', linewidth=2)
        
        # Formatting
        ax.set_xlabel('Condition', fontweight='bold')
        ax.set_ylabel(self.PARAMETER_LABELS.get(parameter, parameter), fontweight='bold')
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(self.CONDITIONS)
        self._clean_legend(ax)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Add sample size info
        n_pd = len(pivot_data[pivot_data['Group'] == 'PD'])
        n_co = len(pivot_data[pivot_data['Group'] == 'CO'])
        ax.text(0.02, 0.98, f'PD: n={n_pd}, Control: n={n_co}', 
               transform=ax.transAxes, va='top', fontsize=10,
               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    def plot_percentage_changes(self, ax: plt.Axes):
        """
        Plot percentage changes across parameters.
        """
        if not hasattr(self, 'individual_df'):
            self.calculate_individual_statistics()
        
        # Prepare data
        plot_data = []
        for param in self.PARAMETERS_OF_INTEREST[:3]:  # Motor parameters
            param_data = self.individual_df[self.individual_df['Parameter'] == param]
            for group in self.GROUPS:
                group_data = param_data[param_data['Group'] == group]
                for _, row in group_data.iterrows():
                    if not pd.isna(row['Pct_Change_Stim_vs_Pre']):
                        plot_data.append({
                            'Parameter': param,
                            'Group': group,
                            'Pct_Change': row['Pct_Change_Stim_vs_Pre']
                        })
        
        if plot_data:
            df = pd.DataFrame(plot_data)
            
            # Create boxplot
            boxprops = dict(linewidth=2)
            whiskerprops = dict(linewidth=2)
            capprops = dict(linewidth=2)
            medianprops = dict(linewidth=2, color='black')
            
            sns.boxplot(
                data=df, x='Parameter', y='Pct_Change', hue='Group',
                palette=self.COLORS, ax=ax, showfliers=False,
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, medianprops=medianprops
            )
            
            # Add individual points with deterministic jitter
            for i, param in enumerate(df['Parameter'].unique()):
                for j, group in enumerate(self.GROUPS):
                    group_data = df[(df['Parameter'] == param) & (df['Group'] == group)]
                    if len(group_data) > 0:
                        # Calculate position
                        pos = i + (j - 0.5) * 0.35
                        # Use deterministic jitter
                        x_jitter = np.linspace(pos - 0.05, pos + 0.05, len(group_data))
                        ax.scatter(x_jitter, group_data['Pct_Change'], 
                                 c=self.COLORS[group], s=40, alpha=0.7,
                                 edgecolor='white', linewidth=0.5, zorder=5)
            
            # Formatting
            param_labels = {
                'TotalDistance_cm': 'Distance',
                'TimeMoving_s': 'Moving Time',
                'MeanSpeed_cm_s': 'Speed'
            }
            ax.set_xticklabels([param_labels.get(t.get_text(), t.get_text()) 
                               for t in ax.get_xticklabels()])
            
            ax.set_ylabel('% Change (Stim vs Pre)', fontweight='bold')
            ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            
            # Clean legend
            self._clean_legend(ax)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        else:
            ax.text(0.5, 0.5, 'No percentage change data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def plot_response_classification(self, ax: plt.Axes):
        """
        Plot response classification as stacked bar chart with clean legend.
        """
        if not hasattr(self, 'individual_df'):
            self.calculate_individual_statistics()
        
        # Get response counts for motor parameters
        motor_params = ['TotalDistance_cm', 'TimeMoving_s', 'MeanSpeed_cm_s']
        motor_data = self.individual_df[self.individual_df['Parameter'].isin(motor_params)]
        
        # Define categories
        categories = ['EXCELLENT', 'GOOD', 'MODERATE', 'SUPPRESSED', 'ADVERSE']
        
        # Calculate counts per group
        pd_counts = []
        co_counts = []
        
        for category in categories:
            pd_count = len(motor_data[(motor_data['Group'] == 'PD') & (motor_data['Response'] == category)])
            co_count = len(motor_data[(motor_data['Group'] == 'CO') & (motor_data['Response'] == category)])
            pd_counts.append(pd_count)
            co_counts.append(co_count)
        
        # Calculate totals
        pd_total = sum(pd_counts)
        co_total = sum(co_counts)
        
        # Convert to percentages
        pd_percent = [c/pd_total*100 if pd_total > 0 else 0 for c in pd_counts]
        co_percent = [c/co_total*100 if co_total > 0 else 0 for c in co_counts]
        
        # Plot
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars with labels
        bars1 = ax.bar(x - width/2, pd_percent, width, 
                       label=f'PD (n={pd_total})', 
                       color=self.COLORS['PD'],
                       edgecolor='black', linewidth=1)
        bars2 = ax.bar(x + width/2, co_percent, width, 
                       label=f'Control (n={co_total})', 
                       color=self.COLORS['CO'],
                       edgecolor='black', linewidth=1)
        
        # Formatting
        ax.set_xlabel('Response Category', fontweight='bold', fontsize=12)
        ax.set_ylabel('Percentage of Animals (%)', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right', fontsize=11)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
        
        # Clean legend
        self._clean_legend(ax)
        
        # Add value labels on bars
        for i, (pd_val, co_val) in enumerate(zip(pd_percent, co_percent)):
            if pd_val > 0:
                ax.text(i - width/2, pd_val + 1, f'{pd_val:.1f}%', 
                       ha='center', va='bottom', fontsize=9)
            if co_val > 0:
                ax.text(i + width/2, co_val + 1, f'{co_val:.1f}%', 
                       ha='center', va='bottom', fontsize=9)
    
    def plot_baseline_response_scatter(self, ax: plt.Axes, parameter: str = 'TotalDistance_cm'):
        """
        Scatter plot of baseline vs stimulation response with separate regression lines per group.
        """
        # Prepare data
        scatter_data = []
        
        for animal in self.wide_df['AnimalID'].unique():
            animal_data = self.wide_df[self.wide_df['AnimalID'] == animal]
            group = animal_data['Group'].iloc[0]
            
            baseline = animal_data[animal_data['Condition'] == 'Pre'][parameter].values
            stim = animal_data[animal_data['Condition'] == 'Stim'][parameter].values
            
            if len(baseline) > 0 and len(stim) > 0:
                baseline_val = baseline[0]
                stim_val = stim[0]
                
                if not pd.isna(baseline_val) and not pd.isna(stim_val):
                    scatter_data.append({
                        'AnimalID': animal,
                        'Group': group,
                        'Baseline': baseline_val,
                        'Stim': stim_val
                    })
        
        if scatter_data:
            df = pd.DataFrame(scatter_data)
            
            # Plot points and regression lines separately for each group
            for group in self.GROUPS:
                group_data = df[df['Group'] == group]
                if len(group_data) > 0:
                    # Scatter points
                    ax.scatter(
                        group_data['Baseline'], group_data['Stim'],
                        c=self.COLORS[group], label=f'{group} (n={len(group_data)})',
                        s=80, alpha=0.8, edgecolors='black', linewidth=1, zorder=5
                    )
                    
                    # Add regression line
                    if len(group_data) >= 3:
                        z = np.polyfit(group_data['Baseline'], group_data['Stim'], 1)
                        p = np.poly1d(z)
                        x_line = np.linspace(group_data['Baseline'].min(), group_data['Baseline'].max(), 50)
                        ax.plot(x_line, p(x_line), color=self.COLORS[group], 
                               linewidth=2, linestyle='-', alpha=0.7)
                        
                        # Calculate correlation for this group
                        r, p_val = pearsonr(group_data['Baseline'], group_data['Stim'])
                        
                        # Add correlation annotation
                        x_pos = group_data['Baseline'].max() * 0.7
                        y_pos = p(x_pos)
                        ax.annotate(f'{group}: r={r:.2f}{"*" if p_val<0.05 else ""}', 
                                   (x_pos, y_pos), fontsize=9,
                                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
            
            # Add identity line
            max_val = max(df['Baseline'].max(), df['Stim'].max())
            ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, 
                   label='Identity', linewidth=1.5, zorder=1)
            
            # Formatting
            ax.set_xlabel(f'Baseline {self.PARAMETER_LABELS.get(parameter, parameter)}', 
                         fontweight='bold')
            ax.set_ylabel(f'Stimulation {self.PARAMETER_LABELS.get(parameter, parameter)}', 
                         fontweight='bold')
            self._clean_legend(ax)
            ax.grid(True, alpha=0.3, linestyle='--')
        else:
            ax.text(0.5, 0.5, 'Insufficient data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    # --- SCIENTIFIC UPGRADE 3: SYMMETRIC RECOVERY AXIS ---
    def plot_recovery_analysis(self, ax: plt.Axes, parameter: str = 'TotalDistance_cm'):
        """
        Plot recovery analysis (Post vs Pre) WITHOUT capping raw values.
        Uses symmetric axis around zero for proper visualization.
        """
        recovery_data = []
        extreme_values = []
        
        for animal in self.wide_df['AnimalID'].unique():
            animal_data = self.wide_df[self.wide_df['AnimalID'] == animal]
            group = animal_data['Group'].iloc[0]
            
            pre = animal_data[animal_data['Condition'] == 'Pre'][parameter].values
            post = animal_data[animal_data['Condition'] == 'Post'][parameter].values
            
            if len(pre) > 0 and len(post) > 0:
                pre_val = pre[0]
                post_val = post[0]
                
                if not pd.isna(pre_val) and not pd.isna(post_val) and pre_val != 0:
                    recovery_pct = ((post_val - pre_val) / abs(pre_val)) * 100
                    
                    # Track extreme values (but don't modify them)
                    if abs(recovery_pct) > 200:
                        extreme_values.append({
                            'AnimalID': animal,
                            'Group': group,
                            'Recovery_Pct': recovery_pct
                        })
                    
                    recovery_data.append({
                        'AnimalID': animal,
                        'Group': group,
                        'Recovery_Pct': recovery_pct
                    })
        
        # Report extreme values
        if extreme_values:
            print(f"\n  ⚠ Extreme recovery values detected (>200% or <-200%):")
            for ev in extreme_values:
                print(f"    {ev['AnimalID']} ({ev['Group']}): {ev['Recovery_Pct']:.1f}%")
            print(f"    These values will be visible with symmetric axis limits")
        
        if recovery_data:
            df = pd.DataFrame(recovery_data)
            
            # Calculate symmetric axis limits
            values = df['Recovery_Pct'].values
            
            # Use 2nd and 98th percentiles for initial bounds
            p02 = np.percentile(values, 2)
            p98 = np.percentile(values, 98)
            
            # Find the maximum absolute value within these percentiles
            max_abs = max(abs(p02), abs(p98))
            
            # Add 10% padding
            max_abs = max_abs * 1.1
            
            # Set symmetric limits
            y_min = -max_abs
            y_max = max_abs
            
            # Create boxplot
            boxprops = dict(linewidth=2)
            whiskerprops = dict(linewidth=2)
            capprops = dict(linewidth=2)
            medianprops = dict(linewidth=2, color='black')
            
            sns.boxplot(
                data=df, x='Group', y='Recovery_Pct',
                palette=self.COLORS, ax=ax, showfliers=True,
                flierprops=dict(marker='o', markerfacecolor='red', markersize=6, alpha=0.7),
                boxprops=boxprops, whiskerprops=whiskerprops,
                capprops=capprops, medianprops=medianprops
            )
            
            # Add individual points with deterministic jitter
            for i, group in enumerate(self.GROUPS):
                group_data = df[df['Group'] == group]['Recovery_Pct'].values
                x_jitter = np.linspace(i - 0.1, i + 0.1, len(group_data))
                ax.scatter(x_jitter, group_data, 
                          c=self.COLORS[group], s=30, alpha=0.5,
                          edgecolor='white', linewidth=0.5, zorder=5)
            
            # Formatting
            ax.set_xlabel('Group', fontweight='bold')
            ax.set_ylabel('% Change (Post vs Pre)', fontweight='bold')
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='No change')
            ax.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='Return to baseline?')
            
            # Set symmetric axis limits
            ax.set_ylim(y_min, y_max)
            
            # Add sample sizes
            for i, group in enumerate(self.GROUPS):
                n = len(df[df['Group'] == group])
                ax.text(i, y_max * 0.95, f'n={n}', ha='center', fontsize=10)
            
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')
            
            # Add annotation about extreme values
            if extreme_values:
                ax.text(0.98, 0.02, f'⚠ {len(extreme_values)} extreme values\n(>200% or <-200%)', 
                       transform=ax.transAxes, ha='right', va='bottom', fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        else:
            ax.text(0.5, 0.5, 'Insufficient data', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
    
    def plot_raincloud(self, ax: plt.Axes, parameter: str):
        """
        Create raincloud plot for a parameter across conditions.
        """
        plot_data = self.wide_df[['Group', 'Condition', parameter]].dropna()
        plot_data = plot_data.rename(columns={parameter: 'Value'})
        
        if PTITPRINCE_AVAILABLE and len(plot_data) > 0:
            try:
                pt.RainCloud(
                    x='Condition', y='Value', data=plot_data,
                    hue='Group', palette=self.COLORS, bw=0.2,
                    width_viol=0.6, ax=ax, orient='v', move=0.2,
                    point_size=5, alpha=0.8, pointplot=True
                )
                
                ax.set_xlabel('Condition', fontweight='bold')
                ax.set_ylabel(self.PARAMETER_LABELS.get(parameter, parameter), fontweight='bold')
                self._clean_legend(ax)
                
            except Exception as e:
                print(f"  Raincloud failed for {parameter}, using enhanced boxplot: {e}")
                self._plot_enhanced_boxplot(ax, plot_data, parameter)
        else:
            self._plot_enhanced_boxplot(ax, plot_data, parameter)
    
    def _clean_legend(self, ax: plt.Axes):
        """Remove duplicate legend entries."""
        handles, labels = ax.get_legend_handles_labels()
        if handles and labels:
            # Remove duplicates while preserving order
            unique = {}
            for h, l in zip(handles, labels):
                if l not in unique:
                    unique[l] = h
            
            # Remove existing legend
            if ax.get_legend():
                ax.get_legend().remove()
            
            # Add clean legend
            ax.legend(unique.values(), unique.keys(), title='Group',
                     title_fontsize='12', fontsize='11', loc='best')
    
    def _plot_enhanced_boxplot(self, ax: plt.Axes, data: pd.DataFrame, parameter: str):
        """
        Enhanced boxplot with individual points and clean legend.
        """
        # Create boxplot
        boxprops = dict(linewidth=2)
        whiskerprops = dict(linewidth=2)
        capprops = dict(linewidth=2)
        medianprops = dict(linewidth=2, color='black')
        
        sns.boxplot(
            data=data, x='Condition', y='Value', hue='Group',
            palette=self.COLORS, ax=ax, showfliers=False,
            boxprops=boxprops, whiskerprops=whiskerprops,
            capprops=capprops, medianprops=medianprops
        )
        
        # Add individual points with deterministic jitter
        for i, condition in enumerate(data['Condition'].unique()):
            for j, group in enumerate(self.GROUPS):
                group_data = data[(data['Condition'] == condition) & (data['Group'] == group)]
                if len(group_data) > 0:
                    pos = i + (j - 0.5) * 0.35
                    x_jitter = np.linspace(pos - 0.05, pos + 0.05, len(group_data))
                    ax.scatter(x_jitter, group_data['Value'], 
                             c=self.COLORS[group], s=40, alpha=0.7,
                             edgecolor='white', linewidth=0.5, zorder=5)
        
        # Formatting
        ax.set_xlabel('Condition', fontweight='bold')
        ax.set_ylabel(self.PARAMETER_LABELS.get(parameter, parameter), fontweight='bold')
        
        # Clean legend
        self._clean_legend(ax)
        ax.grid(True, alpha=0.3, linestyle='--', axis='y')
    
    # --- SCIENTIFIC VISUALIZATION UPGRADE: CLEAN, READABLE HEATMAPS ---
# --- SCIENTIFIC VISUALIZATION UPGRADE: CLEAN, READABLE HEATMAPS ---
    def _plot_correlation_heatmap(self, ax: plt.Axes):
        """
        Plot correlation heatmap for stimulation data - SEPARATE BY GROUP.
        Now with proper spacing, readable labels, and clean layout.
            """
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import matplotlib.patches as patches
        
        params = ['TotalDistance_cm', 'Time PercentCentral', 'MeanSpeed_cm_s', 'TimeMoving_s']
        param_labels_short = {
            'TotalDistance_cm': 'Distance',
            'Time PercentCentral': 'Center\nTime',
            'MeanSpeed_cm_s': 'Speed',
            'TimeMoving_s': 'Moving\nTime'
        }
        
        # Get the position of the parent axis
        bbox = ax.get_position()
        
        # Create more space between heatmaps (0.02 gap)
        total_width = bbox.width
        gap = 0.02
        heatmap_width = (total_width - gap) / 2
        
        # Create two axes with gap between them
        ax_pd = ax.figure.add_axes([bbox.x0, bbox.y0, heatmap_width, bbox.height])
        ax_co = ax.figure.add_axes([bbox.x0 + heatmap_width + gap, bbox.y0, heatmap_width, bbox.height])
        
        # Store for panel saving
        self.figure_axes['heatmap_pd'] = ax_pd
        self.figure_axes['heatmap_co'] = ax_co
        
        valid_heatmaps = []
        
        for idx, (group, sub_ax) in enumerate(zip(['PD', 'CO'], [ax_pd, ax_co])):
            stim_data = self.wide_df[
                (self.wide_df['Condition'] == 'Stim') & 
                (self.wide_df['Group'] == group)
            ]
            
            corr_data = stim_data[params].dropna()
            n = len(corr_data)
            
            if n > 1:
                corr_matrix = corr_data.corr()
                
                # Plot heatmap with larger cells
                im = sub_ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, 
                                aspect='equal', interpolation='nearest')
                valid_heatmaps.append(im)
                
                # Add correlation values with smaller font
                for i in range(len(corr_matrix)):
                    for j in range(len(corr_matrix)):
                        val = corr_matrix.iloc[i, j]
                        # Color text based on background for readability
                        text_color = 'white' if abs(val) > 0.5 else 'black'
                        sub_ax.text(j, i, f'{val:.2f}',
                                ha="center", va="center", 
                                color=text_color, 
                                fontweight='bold',
                                fontsize=8)
                
                # Set ticks with abbreviated labels
                sub_ax.set_xticks(range(len(param_labels_short)))
                sub_ax.set_yticks(range(len(param_labels_short)))
                sub_ax.set_xticklabels([param_labels_short[p] for p in params], 
                                    fontsize=9, ha='center')
                sub_ax.set_yticklabels([param_labels_short[p] for p in params], 
                                    fontsize=9, va='center')
                
                # Add group title with sample size
                title_color = self.COLORS.get(group, 'black')
                sub_ax.set_title(f'{group} (n={n})', 
                            fontweight='bold', 
                            fontsize=11,
                            color=title_color,
                            pad=10)
                
                # Add a light border around each heatmap for separation
                for spine in sub_ax.spines.values():
                    spine.set_visible(True)
                    spine.set_linewidth(0.5)
                    spine.set_color('gray')
                
                # Add sample size warning if small (inside the heatmap, not overlapping)
                if n < 10:
                    sub_ax.text(0.5, -0.15, f'⚠ Small n={n}', 
                            transform=sub_ax.transAxes, 
                            ha='center', 
                            fontsize=7,
                            color='red',
                            fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.2', 
                                    facecolor='white', 
                                    alpha=0.8,
                                    edgecolor='red'))
            else:
                sub_ax.text(0.5, 0.5, 'Insufficient\ndata', 
                        ha='center', va='center', 
                        transform=sub_ax.transAxes, 
                        fontsize=10,
                        color='gray',
                        fontweight='bold')
                sub_ax.set_title(f'{group}', fontweight='bold', fontsize=11)
        
        # Add colorbar only if at least one valid heatmap exists
        if valid_heatmaps:
            # Create colorbar that spans both heatmaps visually
            cax = inset_axes(ax, width="3%", height="70%", loc='center right',
                            bbox_to_anchor=(1.02, 0.15, 1, 1), 
                            bbox_transform=ax.transAxes)
            cbar = plt.colorbar(valid_heatmaps[0], cax=cax)
            cbar.set_label('Pearson Correlation', fontweight='bold', fontsize=10)
            cbar.ax.tick_params(labelsize=8)
            
            # Add a subtle background to the colorbar area
            cax.patch.set_alpha(0)
        
        # Hide the original axis
        ax.set_visible(False)
        
        # Add a subtle separator line between groups using one of the existing axes
        # Get the figure and convert the separator position to figure coordinates
        fig = ax.figure
        mid_x = bbox.x0 + heatmap_width + gap/2
        
        # Create a line using a rectangle patch on a temporary axis
        # Instead of adding to figure, we'll add a vertical line to both heatmaps' spines
        # or just rely on the gap for visual separation
        
        # Option 1: Add a vertical line to the right spine of the first heatmap
        ax_pd.spines['right'].set_visible(True)
        ax_pd.spines['right'].set_linewidth(1)
        ax_pd.spines['right'].set_color('lightgray')
        ax_pd.spines['right'].set_linestyle('--')
        
        # Option 2: Add a vertical line to the left spine of the second heatmap
        ax_co.spines['left'].set_visible(True)
        ax_co.spines['left'].set_linewidth(1)
        ax_co.spines['left'].set_color('lightgray')
        ax_co.spines['left'].set_linestyle('--')
    
    def _plot_individual_animal_summary(self, animal_id: str, ax: plt.Axes):
        """Plot summary for a single animal."""
        animal_data = self.wide_df[self.wide_df['AnimalID'] == animal_id]
        
        if len(animal_data) == 0:
            ax.text(0.5, 0.5, f'No data for {animal_id}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            return
            
        group = animal_data['Group'].iloc[0]
        
        # Parameters to plot
        params = ['TotalDistance_cm', 'TimeMoving_s', 'MeanSpeed_cm_s']
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        labels = ['Distance (cm)', 'Moving Time (s)', 'Speed (cm/s)']
        
        x = [0, 1, 2]
        x_labels = ['Pre', 'Stim', 'Post']
        
        # Plot each parameter
        for param, color, label in zip(params, colors, labels):
            values = []
            for condition in x_labels:
                val = animal_data[animal_data['Condition'] == condition][param].values
                values.append(val[0] if len(val) > 0 else np.nan)
            
            if not any(pd.isna(values)):
                ax.plot(x, values, 'o-', color=color, label=label, linewidth=2, markersize=8)
        
        ax.set_xlabel('Condition', fontweight='bold')
        ax.set_ylabel('Value', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(f'{animal_id} ({group})', fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
    
    # =========================================================
    # MAIN FIGURE CREATION
    # =========================================================
    
    def create_figure_1(self) -> plt.Figure:
        """
        Create main figure with group trends and percentage changes.
        """
        print("\n" + "=" * 60)
        print("CREATING FIGURE 1")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Figure 1: Motor Responses to Stimulation', 
                    fontsize=16, fontweight='bold')
        
        # Store axes for panel saving
        self.figure_axes['fig1'] = axes.flatten()
        
        # Panel A: Individual trajectories
        self.plot_individual_trajectories(axes[0, 0])
        axes[0, 0].set_title('A. Individual Animal Trajectories', fontweight='bold', loc='left')
        
        # Panel B: Percentage changes
        self.plot_percentage_changes(axes[0, 1])
        axes[0, 1].set_title('B. Percent Change from Baseline', fontweight='bold', loc='left')
        
        # Panel C: Response classification
        self.plot_response_classification(axes[1, 0])
        axes[1, 0].set_title('C. Response Classification', fontweight='bold', loc='left')
        
        # Panel D: Baseline vs stimulation
        self.plot_baseline_response_scatter(axes[1, 1])
        axes[1, 1].set_title('D. Baseline vs Stimulation Response', fontweight='bold', loc='left')
        
        plt.tight_layout()
        
        # Save figure and panels
        self._save_figure_and_panels(fig, 'Figure1_Main_Results')
        
        return fig
    
    def create_figure_2(self) -> plt.Figure:
        """
        Create second figure with distribution visualizations.
        """
        print("\n" + "=" * 60)
        print("CREATING FIGURE 2")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Figure 2: Distribution of Behavioral Parameters', 
                    fontsize=16, fontweight='bold')
        
        # Store axes for panel saving
        self.figure_axes['fig2'] = axes.flatten()
        
        # Panel A: Total Distance
        self.plot_raincloud(axes[0, 0], 'TotalDistance_cm')
        axes[0, 0].set_title('A. Total Distance', fontweight='bold', loc='left')
        
        # Panel B: Time Moving
        self.plot_raincloud(axes[0, 1], 'TimeMoving_s')
        axes[0, 1].set_title('B. Time Moving', fontweight='bold', loc='left')
        
        # Panel C: Speed
        self.plot_raincloud(axes[1, 0], 'MeanSpeed_cm_s')
        axes[1, 0].set_title('C. Mean Speed', fontweight='bold', loc='left')
        
        # Panel D: Center Time
        self.plot_raincloud(axes[1, 1], 'Time PercentCentral')
        axes[1, 1].set_title('D. Time in Center', fontweight='bold', loc='left')
        
        plt.tight_layout()
        
        # Save figure and panels
        self._save_figure_and_panels(fig, 'Figure2_Distributions')
        
        return fig
    
    def create_supplementary_figure_1(self) -> plt.Figure:
        """
        Create supplementary figure with recovery analysis and correlations.
        """
        print("\n" + "=" * 60)
        print("CREATING SUPPLEMENTARY FIGURE 1")
        print("=" * 60)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Supplementary Figure 1: Additional Analyses', 
                    fontsize=16, fontweight='bold')
        
        # Store axes for panel saving
        self.figure_axes['supp_fig1'] = axes.flatten()
        
        # Panel A: Recovery analysis (now with symmetric axis)
        self.plot_recovery_analysis(axes[0, 0])
        axes[0, 0].set_title('A. Post-Stimulation Recovery', fontweight='bold', loc='left')
        
        # Panel B: Correlation heatmap (improved version)
        self._plot_correlation_heatmap(axes[0, 1])
        axes[0, 1].set_title('B. Parameter Correlations by Group', fontweight='bold', loc='left')
        
        # Get example animals
        pd_animals = self.wide_df[self.wide_df['Group'] == 'PD']['AnimalID'].unique()
        co_animals = self.wide_df[self.wide_df['Group'] == 'CO']['AnimalID'].unique()
        
        example_animals = []
        if len(pd_animals) > 0:
            example_animals.append(pd_animals[0])
        if len(co_animals) > 0:
            example_animals.append(co_animals[0])
        
        # Panels C-D: Individual animal examples
        for i, animal in enumerate(example_animals[:2]):
            ax = axes[1, i]
            self._plot_individual_animal_summary(animal, ax)
            panel_label = 'C' if i == 0 else 'D'
            ax.set_title(f'{panel_label}. Animal: {animal}', 
                        fontweight='bold', loc='left')
        
        # Hide unused subplot if only one example
        if len(example_animals) < 2:
            axes[1, 1].axis('off')
        
        plt.tight_layout()
        
        # Save figure and panels
        self._save_figure_and_panels(fig, 'Supplementary_Figure1_Additional')
        
        return fig
    
    # --- ROBUST PANEL SAVING ---
    def _save_figure_and_panels(self, fig: plt.Figure, base_name: str):
        """
        Save full figure and individual panels using bounding box extraction.
        Filters to only primary subplot axes.
        """
        # Save full figure
        png_path = os.path.join(self.dirs['figures'], f'{base_name}.png')
        svg_path = os.path.join(self.dirs['figures'], f'{base_name}.svg')
        
        fig.savefig(png_path, dpi=300, bbox_inches='tight')
        fig.savefig(svg_path, format='svg', bbox_inches='tight')
        print(f"✓ Figure saved: {png_path}")
        
        # Get the renderer
        fig.canvas.draw()
        if hasattr(fig.canvas, 'get_renderer'):
            renderer = fig.canvas.get_renderer()
        else:
            renderer = fig.canvas.renderer
        
        # Get primary axes (excluding colorbars and hidden axes)
        primary_axes = []
        for ax in fig.axes:
            # Skip if axis is not visible
            if not ax.get_visible():
                continue
            
            # Skip colorbar axes
            if hasattr(ax, 'get_label') and ax.get_label() == '<colorbar>':
                continue
            
            # Skip axes that are likely colorbars
            if hasattr(ax, 'get_ylabel') and ax.get_ylabel() == 'Pearson Correlation':
                continue
            
            # Skip very small axes (likely colorbars or insets)
            bbox = ax.get_position()
            if bbox.width < 0.1 or bbox.height < 0.1:
                continue
            
            primary_axes.append(ax)
        
        # If we have stored axes for this figure, use them preferentially
        for key, stored_axes in self.figure_axes.items():
            if key in base_name.lower():
                primary_axes = stored_axes
                break
        
        panel_labels = ['A', 'B', 'C', 'D']
        
        # Save individual panels
        for idx, ax in enumerate(primary_axes[:4]):
            if idx < len(panel_labels):
                try:
                    # Get the bounding box of the axis
                    bbox = ax.get_tightbbox(renderer)
                    if bbox is None:
                        print(f"  ⚠ Could not get bbox for panel {panel_labels[idx]}, skipping")
                        continue
                    
                    # Transform to inches for saving
                    bbox_inches = bbox.transformed(fig.dpi_scale_trans.inverted())
                    
                    # Add a small margin
                    bbox_inches = bbox_inches.expanded(1.05, 1.05)
                    
                    # Save the panel
                    panel_path = os.path.join(self.dirs['panels'], f'{base_name}_Panel_{panel_labels[idx]}.png')
                    fig.savefig(panel_path, dpi=300, bbox_inches=bbox_inches)
                    
                    # Also save as SVG
                    panel_svg_path = os.path.join(self.dirs['panels'], f'{base_name}_Panel_{panel_labels[idx]}.svg')
                    fig.savefig(panel_svg_path, format='svg', bbox_inches=bbox_inches)
                    
                    print(f"  ✓ Panel {panel_labels[idx]} saved")
                    
                except Exception as e:
                    print(f"  ⚠ Error saving panel {panel_labels[idx]}: {e}")
                    self._save_panel_fallback(fig, ax, idx, panel_labels[idx], base_name)
        
        print(f"✓ All panels saved to: {self.dirs['panels']}")
    
    def _save_panel_fallback(self, fig: plt.Figure, ax: plt.Axes, idx: int, panel_label: str, base_name: str):
        """Fallback method for panel saving."""
        try:
            panel_fig = plt.figure(figsize=(6, 5))
            panel_ax = panel_fig.add_subplot(111)
            
            # Copy title and labels
            if ax.get_title():
                panel_ax.set_title(ax.get_title(), fontweight='bold')
            panel_ax.set_xlabel(ax.get_xlabel(), fontweight='bold')
            panel_ax.set_ylabel(ax.get_ylabel(), fontweight='bold')
            
            # Copy lines
            for line in ax.lines:
                panel_ax.plot(line.get_xdata(), line.get_ydata(), 
                            color=line.get_color(), 
                            linewidth=line.get_linewidth(),
                            alpha=line.get_alpha(),
                            linestyle=line.get_linestyle())
            
            # Set limits and ticks
            panel_ax.set_xlim(ax.get_xlim())
            panel_ax.set_ylim(ax.get_ylim())
            panel_ax.set_xticks(ax.get_xticks())
            panel_ax.set_yticks(ax.get_yticks())
            
            # Save
            panel_path = os.path.join(self.dirs['panels'], f'{base_name}_Panel_{panel_label}_fallback.png')
            panel_fig.savefig(panel_path, dpi=300, bbox_inches='tight')
            plt.close(panel_fig)
            print(f"  ✓ Panel {panel_label} saved (fallback method)")
        except Exception as e:
            print(f"  ⚠ Fallback also failed for panel {panel_label}: {e}")
    
    # =========================================================
    # OUTPUT GENERATION
    # =========================================================
    
    def save_statistical_summary(self) -> str:
        """
        Save statistical results to CSV.
        """
        if len(self.statistical_results) == 0:
            self.perform_full_analysis()
        
        output_df = self.statistical_results.copy()
        
        # Add formatted columns
        if 'p_value' in output_df.columns:
            output_df['p_value_formatted'] = output_df['p_value'].apply(
                lambda x: f'{x:.4f}' if pd.notna(x) else 'NA'
            )
        
        if 'p_corrected' in output_df.columns:
            output_df['p_corrected_formatted'] = output_df['p_corrected'].apply(
                lambda x: f'{x:.4f}' if pd.notna(x) else 'NA'
            )
        
        if 'effect_size' in output_df.columns:
            output_df['effect_size_formatted'] = output_df['effect_size'].apply(
                lambda x: f'{x:.3f}' if pd.notna(x) else 'NA'
            )
        
        # Save main statistical summary
        csv_path = os.path.join(self.dirs['tables'], 'Statistical_Summary.csv')
        output_df.to_csv(csv_path, index=False)
        print(f"✓ Statistical summary saved to: {csv_path}")
        
        # Print summary
        self._print_statistical_summary(output_df)
        
        return csv_path
    
    def _print_statistical_summary(self, results_df: pd.DataFrame):
        """Print formatted statistical summary."""
        print("\n" + "=" * 80)
        print("STATISTICAL SUMMARY")
        print("=" * 80)
        
        if len(results_df) == 0:
            print("No statistical results available.")
            return
        
        # Print ANOVA results first (including partial eta squared - using 'eta^2' instead of η due to encoding)
        print("\nREPEATED MEASURES ANOVA RESULTS:")
        print("-" * 60)
        for param, results in self.anova_results.items():
            if 'error' in results:
                print(f"\n  {param}: {results['error']}")
            else:
                print(f"\n  {param}:")
                if 'group_p' in results:
                    print(f"    Group effect: F={results.get('group_F', np.nan):.2f}, "
                          f"p={results['group_p']:.4f}", end='')
                    if 'group_eta_sq_partial' in results:
                        print(f", eta²p={results['group_eta_sq_partial']:.3f}")
                    else:
                        print()
                if 'condition_p' in results:
                    print(f"    Condition effect: p={results['condition_p']:.4f}", end='')
                    if 'condition_eta_sq_partial' in results:
                        print(f", eta²p={results['condition_eta_sq_partial']:.3f}")
                    else:
                        print()
                if 'interaction_p' in results:
                    print(f"    Interaction: p={results['interaction_p']:.4f}", end='')
                    if 'interaction_eta_sq_partial' in results:
                        print(f", eta²p={results['interaction_eta_sq_partial']:.3f}")
                    else:
                        print()
        
        # Between-group comparisons
        print("\nBETWEEN-GROUP COMPARISONS (PD vs Control) [Hedges' g]:")
        print("-" * 60)
        
        between = results_df[results_df['comparison'].str.contains('PD_vs_CO', na=False)]
        if len(between) > 0:
            for param in self.PARAMETERS_OF_INTEREST:
                print(f"\n  {param}:")
                param_between = between[between['parameter'] == param]
                for _, row in param_between.iterrows():
                    condition = row.get('condition', 'Unknown')
                    p = row.get('p_value', np.nan)
                    p_corr = row.get('p_corrected', np.nan)
                    sig = row.get('significant', False)
                    g = row.get('effect_size', np.nan)
                    
                    sig_marker = " ***" if sig else ""
                    if not pd.isna(p):
                        print(f"    {condition}: p={p:.4f}{sig_marker}, g={g:.3f}", end='')
                        if not pd.isna(p_corr):
                            print(f" (FDR: {p_corr:.4f})")
                        else:
                            print()
        else:
            print("  No between-group comparisons available.")
        
        # Within-group comparisons
        print("\nWITHIN-GROUP COMPARISONS [Hedges' g]:")
        print("-" * 60)
        
        for group in self.GROUPS:
            print(f"\n  {group} group:")
            within = results_df[
                (results_df['comparison'].str.contains(group, na=False)) & 
                (~results_df['comparison'].str.contains('PD_vs_CO', na=False))
            ]
            
            if len(within) > 0:
                for param in self.PARAMETERS_OF_INTEREST:
                    param_within = within[within['parameter'] == param]
                    if len(param_within) > 0:
                        print(f"    {param}:")
                        for _, row in param_within.iterrows():
                            comp = row.get('comparison', 'Unknown').replace(f'{group}_', '')
                            p = row.get('p_value', np.nan)
                            p_corr = row.get('p_corrected', np.nan)
                            sig = row.get('significant', False)
                            g = row.get('effect_size', np.nan)
                            
                            sig_marker = " ***" if sig else ""
                            if not pd.isna(p):
                                print(f"      {comp}: p={p:.4f}{sig_marker}, g={g:.3f}")
            else:
                print(f"    No within-group comparisons for {group}")
    
    # --- FIXED: README with proper encoding handling ---
    def generate_readme(self):
        """Generate README file with analysis summary."""
        readme_path = os.path.join(self.output_dir, "README.txt")
        
        # Use utf-8 encoding to handle special characters
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write("OPEN FIELD TEST ANALYSIS - PUBLICATION PIPELINE\n")
            f.write("=" * 60 + "\n\n")
            
            f.write("ANALYSIS SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Analysis date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Data file: {self.data_path}\n\n")
            
            f.write("SAMPLE SIZES\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total animals: {len(self.wide_df['AnimalID'].unique())}\n")
            f.write(f"PD animals: {len(self.wide_df[self.wide_df['Group'] == 'PD']['AnimalID'].unique())}\n")
            f.write(f"Control animals: {len(self.wide_df[self.wide_df['Group'] == 'CO']['AnimalID'].unique())}\n\n")
            
            f.write("OUTLIER HANDLING\n")
            f.write("-" * 40 + "\n")
            f.write(f"Method: MAD-based robust Z-score\n")
            f.write(f"Outliers replaced with group mean: {self.replace_outliers}\n")
            
            outlier_count = 0
            if hasattr(self, 'outlier_log') and isinstance(self.outlier_log, pd.DataFrame) and len(self.outlier_log) > 0:
                outlier_count = len(self.outlier_log)
            f.write(f"Total outliers detected: {outlier_count}\n")
            f.write(f"Log file: logs/outlier_detection_log.csv\n\n")
            
            f.write("PARAMETERS ANALYZED\n")
            f.write("-" * 40 + "\n")
            f.write(f"Primary outcome: TotalDistance_cm\n")
            f.write(f"Secondary outcomes: {', '.join(self.SECONDARY_PARAMETERS)}\n\n")
            
            f.write("STATISTICAL METHODS\n")
            f.write("-" * 40 + "\n")
            f.write("• 2×3 Repeated Measures ANOVA (Group × Condition)\n")
            f.write("• Partial eta squared (η²p) reported for ANOVA effects\n")
            f.write("• Post hoc pairwise comparisons only after significant interaction\n")
            f.write("• Between-group comparisons: Independent t-tests (Welch correction)\n")
            f.write("• Within-group comparisons: Paired t-tests\n")
            f.write("• Effect sizes: Hedges' g (bias-corrected Cohen's d)\n")
            f.write("• Non-parametric alternatives when assumptions violated\n")
            f.write("• FDR correction applied to secondary outcomes only\n")
            f.write("• Confidence intervals: 95% CI based on t-distribution\n")
            f.write("• Correlations computed separately for each group\n")
            f.write("• Spearman correlation reported for n<10\n\n")
            
            f.write("RESPONSE CLASSIFICATION THRESHOLDS\n")
            f.write("-" * 40 + "\n")
            f.write(f"EXCELLENT: > {self.RESPONSE_THRESHOLDS['excellent']}% change\n")
            f.write(f"GOOD: {self.RESPONSE_THRESHOLDS['good']}-{self.RESPONSE_THRESHOLDS['excellent']}% change\n")
            f.write(f"MODERATE: {self.RESPONSE_THRESHOLDS['moderate']}-{self.RESPONSE_THRESHOLDS['good']}% change\n\n")
            
            f.write("OUTPUT DIRECTORY STRUCTURE\n")
            f.write("-" * 40 + "\n")
            f.write("1. figures/ - Publication figures (PNG + SVG)\n")
            f.write("2. figure_panels/ - Individual figure panels\n")
            f.write("3. tables/ - Formatted results tables\n")
            f.write("4. supplementary_tables/ - Detailed statistical tables\n")
            f.write("5. individual_plots/ - Individual animal plots\n")
            f.write("6. logs/ - Outlier detection and decision tree logs\n")
        
        print(f"✓ README saved to: {readme_path}")
    
    # --- COMPREHENSIVE TRANSPARENCY REPORT ---
    def _print_transparency_report(self):
        """Print comprehensive transparency report at end of analysis."""
        print("\n" + "=" * 80)
        print("TRANSPARENCY REPORT")
        print("=" * 80)
        
        # Sample sizes
        print(f"\n📊 SAMPLE SIZES:")
        print(f"  • Total animals: {len(self.wide_df['AnimalID'].unique())}")
        print(f"  • PD: {len(self.wide_df[self.wide_df['Group'] == 'PD']['AnimalID'].unique())}")
        print(f"  • Control: {len(self.wide_df[self.wide_df['Group'] == 'CO']['AnimalID'].unique())}")
        
        # Outlier handling
        print(f"\n🔍 OUTLIER HANDLING:")
        if hasattr(self, 'outlier_log') and isinstance(self.outlier_log, pd.DataFrame) and len(self.outlier_log) > 0:
            detected = len(self.outlier_log)
            replaced = self.outlier_log['Replaced'].sum() if 'Replaced' in self.outlier_log.columns else 0
            print(f"  • Outliers detected: {detected}")
            print(f"  • Outliers replaced: {replaced}")
            print(f"  • Log file: logs/outlier_detection_log.csv")
        else:
            print(f"  • No outliers detected")
        
        # Statistical decisions
        if hasattr(self, 'anova_results'):
            print(f"\n📈 STATISTICAL DECISIONS:")
            sig_interactions = 0
            post_hoc_performed = 0
            
            for param, res in self.anova_results.items():
                if 'interaction_p' in res and not pd.isna(res['interaction_p']):
                    if res['interaction_p'] < 0.05:
                        sig_interactions += 1
                        post_hoc_performed += 1
                        print(f"  • {param}: Interaction significant (p={res['interaction_p']:.4f}) → post-hoc performed")
                    else:
                        print(f"  • {param}: Interaction NOT significant (p={res['interaction_p']:.4f}) → post-hoc skipped")
            
            print(f"\n  Summary: {sig_interactions}/{len(self.anova_results)} significant interactions")
            print(f"           {post_hoc_performed} post-hoc analyses performed")
        
        # Correlation warnings
        if hasattr(self, 'correlation_results'):
            print(f"\n📊 CORRELATION ANALYSIS:")
            for group in self.GROUPS:
                if group in self.correlation_results and len(self.correlation_results[group]) > 0:
                    n = self.correlation_results[group]['N'].iloc[0] if 'N' in self.correlation_results[group].columns else '?'
                    print(f"  • {group}: n={n}")
                    if isinstance(n, (int, float)) and n < 10:
                        print(f"    ⚠ Small sample - interpret cautiously")
        
        # Recovery analysis
        print(f"\n📉 RECOVERY ANALYSIS:")
        print(f"  • Raw values preserved (no capping)")
        print(f"  • Symmetric axis limits (balanced around zero)")
        print(f"  • Extreme values flagged but not modified")
        
        # Response thresholds
        print(f"\n🎯 RESPONSE CLASSIFICATION THRESHOLDS:")
        print(f"  • EXCELLENT: > {self.RESPONSE_THRESHOLDS['excellent']}%")
        print(f"  • GOOD: {self.RESPONSE_THRESHOLDS['good']}–{self.RESPONSE_THRESHOLDS['excellent']}%")
        print(f"  • MODERATE: {self.RESPONSE_THRESHOLDS['moderate']}–{self.RESPONSE_THRESHOLDS['good']}%")
        
        # Final summary
        print("\n" + "=" * 80)
        print("DECISION TREE SUMMARY")
        print("=" * 80)
        print("✓ Interaction significant → post-hoc executed")
        print("✓ Interaction not significant → post-hoc skipped")
        print("✓ FDR correction applied to secondary outcomes only")
        print("✓ Partial eta squared (η²p) reported for ANOVA")
        print("✓ Hedges' g reported instead of Cohen's d")
        print("✓ Correlations computed separately per group")
        print("✓ Random seed fixed (42) for reproducibility")
        print("✓ No biological values altered unless explicitly requested")
        print("=" * 80)
    
    # --- INTEGRATED ANALYSIS PIPELINE ---
    def run_full_analysis(self):
        """
        Run complete analysis pipeline with transparency reporting.
        """
        print("\n" + "=" * 80)
        print("OPEN FIELD TEST ANALYSIS - PUBLICATION PIPELINE")
        print("=" * 80)
        
        # Run ANOVA
        self.run_all_anovas()
        
        # Perform statistical analysis with hierarchy
        self.perform_full_analysis()
        
        # Calculate individual statistics
        self.calculate_individual_statistics()
        
        # Run correlation analysis (separate by group)
        self.correlation_results = self.run_correlation_analysis()
        
        # Create figures
        print("\n" + "=" * 60)
        print("CREATING FIGURES")
        print("=" * 60)
        
        fig1 = self.create_figure_1()
        plt.close(fig1)
        
        fig2 = self.create_figure_2()
        plt.close(fig2)
        
        supp_fig1 = self.create_supplementary_figure_1()
        plt.close(supp_fig1)
        
        # Save results
        self.save_statistical_summary()
        self.generate_readme()
        
        # Print transparency report
        self._print_transparency_report()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE!")
        print("=" * 80)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"\nTo cite this analysis:")
        print(f"  • Statistical methods: 2×3 RM-ANOVA with hierarchical post-hoc testing")
        print(f"  • Effect sizes: Partial eta squared (η²p) and Hedges' g")
        print(f"  • Outlier detection: MAD-based robust Z-score (threshold=3.5)")
        print(f"  • Multiple comparison correction: Benjamini-Hochberg FDR")
        
        # Close all figures
        plt.close('all')
        
        return self.statistical_results


# =========================================================
# COMMAND LINE INTERFACE
# =========================================================

def main():
    """Main function with command line argument parsing."""
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser = argparse.ArgumentParser(
        description='Open Field Test Analysis Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python oft_analysis.py --data OFT.xlsx --output ./Results
  python oft_analysis.py --data /path/to/OFT.xlsx --output /path/to/output --replace-outliers
  python oft_analysis.py --data data.csv --output ./Results --outlier-method iqr --outlier-threshold 2.5
        """
    )
    
    parser.add_argument(
        '--data', '-d',
        type=str,
        default=os.path.join(script_dir, 'OFT.xlsx'),
        help='Path to data file (supports .xlsx, .xls, .csv) (default: OFT.xlsx in script directory)'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=os.path.join(script_dir, 'OFT_Results'),
        help='Output directory for results (default: ./OFT_Results)'
    )
    
    parser.add_argument(
        '--outlier-method',
        type=str,
        choices=['iqr', 'mad', 'zscore'],
        default='mad',
        help='Outlier detection method (default: mad)'
    )
    
    parser.add_argument(
        '--outlier-threshold',
        type=float,
        default=3.5,
        help='Threshold for outlier detection (default: 3.5 for MAD, 1.5 for IQR, 3 for zscore)'
    )
    
    parser.add_argument(
        '--replace-outliers',
        action='store_true',
        help='Replace outliers with group mean (default: False)'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("OPEN FIELD TEST ANALYSIS PIPELINE")
    print("=" * 80)
    print(f"Data file: {args.data}")
    print(f"Output directory: {args.output}")
    print(f"Outlier method: {args.outlier_method}")
    print(f"Outlier threshold: {args.outlier_threshold}")
    print(f"Replace outliers: {args.replace_outliers}")
    
    # Check if file exists
    if not os.path.exists(args.data):
        print(f"\n❌ ERROR: Data file not found: {args.data}")
        print("Please provide the correct path using --data.")
        sys.exit(1)
    
    # Create analyzer
    try:
        analyzer = PublicationOFTAnalyzer(
            data_path=args.data, 
            output_dir=args.output,
            replace_outliers=args.replace_outliers
        )
    except PermissionError:
        print(f"\n❌ ERROR: Cannot write to output directory '{args.output}'. Please specify a writable location using --output.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: Failed to initialize analyzer: {e}")
        sys.exit(1)
    
    # Override outlier settings if specified
    if args.outlier_method != 'mad' or args.outlier_threshold != 3.5:
        analyzer._detect_and_replace_outliers(
            method=args.outlier_method,
            threshold=args.outlier_threshold,
            replace=args.replace_outliers
        )
    
    # Run analysis
    try:
        results = analyzer.run_full_analysis()
    except Exception as e:
        print(f"\n❌ ERROR: Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print("\n" + "=" * 80)
    print("Analysis completed successfully!")
    print("=" * 80)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())