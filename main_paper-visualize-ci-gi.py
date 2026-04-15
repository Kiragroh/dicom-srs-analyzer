import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import os
import sys
import config

# Set the style to paper-friendly
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# Read the metrics CSV with CI and GI data
metrics_path = config.METRICS_CSV_PATH
df_metrics = pd.read_csv(metrics_path)
print(f"Loaded metrics from: {metrics_path}")
print(f"Metrics shape: {df_metrics.shape}")
print(f"Metrics columns: {df_metrics.columns.tolist()}")

# Check if DosisStudy.xlsx exists, otherwise use metrics data only
excel_path = os.path.join(config.DATA_ROOT, 'DosisStudy.xlsx')
if os.path.exists(excel_path):
    df_original = pd.read_excel(excel_path, sheet_name='Raw')
    print("Original columns:", df_original.columns.tolist())
else:
    print(f"Warning: {excel_path} not found. Using metrics data only.")
    df_original = None
print("\nMetrics columns:", df_metrics.columns.tolist())
print("\nMetrics scenarios:", df_metrics['Scenario'].unique())

# Filter metrics data for HA plans only and relevant scenarios
# Map scenarios to Tx labels
scenario_mapping = {
    'nominal': '1',  # Reference (no setup error)
    '0.29/0.29/0.29//0.29/0.29/0.29': '2',  # 0.5mm/0.5deg
    '0.4/0.4/0.4//0.58/0.58/0.58': '3',      # 0.7mm/1deg
    '0.58/0.58/0.58//0.58/0.58/0.58': '4',   # 1mm/1deg
    '0/0/0//0.58/0.58/0.58': '5',            # 0mm/1deg (rotation only)
    '0.58/0.58/0.58//0/0/0': '6',            # 1mm/0deg (translation only)
}

# Filter for HA plans and map scenarios
df_metrics_ha = df_metrics[df_metrics['PlanType'] == 'HA'].copy()
df_metrics_ha['Tx'] = df_metrics_ha['Scenario'].map(scenario_mapping)
df_metrics_ha = df_metrics_ha.dropna(subset=['Tx'])

print(f"\nMetrics HA data shape: {df_metrics_ha.shape}")
print(f"Unique Tx in metrics: {df_metrics_ha['Tx'].unique()}")

# Use StructureName_Original to create ptvs_clean for matching
df_metrics_ha['ptvs_clean'] = df_metrics_ha['StructureName_Original'].str.lower().str.strip()

if df_original is not None:
    # Original merge logic with Excel file
    df_original['ptvs_clean'] = df_original['ptvs'].str.lower().str.strip()

    print("\nSample ptvs_clean from metrics:")
    print(df_metrics_ha[['PatientFolder', 'StructureName_Original', 'ptvs_clean', 'Tx', 'PaddickCI', 'GI']].head(10))

    print("\nSample ptvs_clean from original:")
    print(df_original[['MRN', 'ptvs', 'ptvs_clean', 'Tx']].head(10))

    # Check unique values for matching
    print("\nUnique ptvs_clean in original (s2):", df_original[df_original['MRN'] == 'zz_Complex_MM_UKE_s2']['ptvs_clean'].unique())
    print("\nUnique ptvs_clean in metrics (s2):", df_metrics_ha[df_metrics_ha['PatientFolder'] == 'zz_Complex_MM_UKE_s2']['ptvs_clean'].unique())

    # Merge the dataframes using ptvs_clean and Tx
    df_merged = df_original.merge(
        df_metrics_ha[['PatientFolder', 'StructureName_Original', 'ptvs_clean', 'Tx',
                       'PaddickCI', 'RTOG_CI', 'GI', 'HI']],
        left_on=['MRN', 'ptvs_clean', 'Tx'],
        right_on=['PatientFolder', 'ptvs_clean', 'Tx'],
        how='left'
    )
else:
    # Use metrics data only - create synthetic df_merged
    print("\nUsing metrics data only (DosisStudy.xlsx not found)")
    print("\nSample ptvs_clean from metrics:")
    print(df_metrics_ha[['PatientFolder', 'StructureName_Original', 'ptvs_clean', 'Tx', 'PaddickCI', 'GI']].head(10))

    # Create df_merged from metrics data with renamed columns for compatibility
    df_merged = df_metrics_ha.copy()
    df_merged['MRN'] = df_merged['PatientFolder']
    df_merged['ptvs'] = df_merged['StructureName_Original']
    # TV_cc is the volume in cc
    df_merged['volume'] = df_merged['TV_cc']
    # D98_Gy -> D98%[%] for compatibility (note: these are different units but kept for compatibility)
    df_merged['D98%[%]'] = df_merged['D98_Gy']

print(f"\nMerged data shape: {df_merged.shape}")
print(f"Rows with CI data: {df_merged['PaddickCI'].notna().sum()}")
print(f"Rows with GI data: {df_merged['GI'].notna().sum()}")

# Show sample of merged data
print("\nSample merged data:")
print(df_merged[['MRN', 'ptvs', 'Tx', 'PaddickCI', 'GI', 'D98%[%]']].head(20))

# Save merged data for verification
output_excel = os.path.join(config.OUTPUT_DIR, 'merged_data_with_ci_gi.xlsx')
df_merged.to_excel(output_excel, index=False)
print(f"\nMerged data saved to '{output_excel}'")

# ===== VISUALIZATION FOR CI AND GI =====
# Use df_merged which now contains CI and GI data

df = df_merged.copy()

# Filter only rows with CI and GI data
df = df.dropna(subset=['PaddickCI', 'GI'])
print(f"\nData with CI/GI for visualization: {df.shape[0]} rows")

# Create custom x-tick labels (without HA)
setup_labels = ['no setup\nerror', '0.5 mm\n0.5°', '0.7 mm\n1°', '1 mm\n1°', '0 mm\n1°', '1 mm\n0°']
tx_mapping = dict(zip(['1', '2', '3', '4', '5', '6'], setup_labels))
df['Setup'] = df['Tx'].map(tx_mapping)

# Convert CI and GI to numeric (handle any string markers like *)
df['PaddickCI'] = pd.to_numeric(df['PaddickCI'].astype(str).str.replace('*', ''), errors='coerce')
df['GI'] = pd.to_numeric(df['GI'].astype(str).str.replace('*', ''), errors='coerce')

# Calculate percentage differences from HA_1 (reference)
# For CI (higher is better, decrease is negative impact)
# For GI (lower is better, increase is negative impact)
ha1_values = df[df['Tx'] == '1'].set_index(['MRN', 'ptvs'])

diff_data = []
for tx in ['2', '3', '4', '5', '6']:
    tx_data = df[df['Tx'] == tx].set_index(['MRN', 'ptvs'])
    
    for metric_name, metric_col in [('CI', 'PaddickCI'), ('GI', 'GI')]:
        for (mrn, ptv) in tx_data.index:
            if (mrn, ptv) in ha1_values.index and (mrn, ptv) in tx_data.index:
                ref_val = ha1_values.loc[(mrn, ptv), metric_col]
                tx_val = tx_data.loc[(mrn, ptv), metric_col]
                
                if pd.notna(ref_val) and pd.notna(tx_val) and ref_val != 0:
                    # Calculate percentage difference
                    pct_diff = ((tx_val - ref_val) / ref_val) * 100
                    
                    diff_data.append({
                        'MRN': mrn,
                        'ptvs': ptv,
                        'Setup': tx_mapping[tx],
                        'Metric': metric_name,
                        'Difference': pct_diff,
                        'Ref_Value': ref_val,
                        'Tx_Value': tx_val
                    })

diff_df = pd.DataFrame(diff_data)
print(f"\nDifference data shape: {diff_df.shape}")
print(f"CI diff rows: {(diff_df['Metric'] == 'CI').sum()}")
print(f"GI diff rows: {(diff_df['Metric'] == 'GI').sum()}")

# Calculate median volume from reference setup for volume-based splitting
median_volume = df[df['Tx'] == '1']['volume'].median()
small_vol_mrn_ptvs = df[(df['Tx'] == '1') & (df['volume'] <= median_volume)][['MRN', 'ptvs']].values.tolist()
large_vol_mrn_ptvs = df[(df['Tx'] == '1') & (df['volume'] > median_volume)][['MRN', 'ptvs']].values.tolist()

# Create the plot with 3x1 subplots for CI and GI
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 5.7), sharex=True, sharey=True)

# Custom colors for each metric
colors = {'CI': '#1f77b4', 'GI': '#ff7f0e'}

# Function to create boxplot
def create_boxplot(data, ax, title):
    sns.boxplot(data=data, x='Setup', y='Difference', hue='Metric', 
                palette=colors, width=0.7, ax=ax)
    
    # Add mean markers for each setup and metric
    for setup in data['Setup'].unique():
        setup_data = data[data['Setup'] == setup]
        for metric in setup_data['Metric'].unique():
            metric_data = setup_data[setup_data['Metric'] == metric]
            mean_val = metric_data['Difference'].mean()
            
            # Get x-position (setup index)
            setup_idx = list(data['Setup'].unique()).index(setup)
            if metric == 'CI':
                x_pos = setup_idx - 0.2
            else:  # GI
                x_pos = setup_idx + 0.2
            
            # Plot mean marker (white square with black edge)
            ax.plot(x_pos, mean_val, 's', color='white', markeredgecolor='black', 
                   markersize=0, markeredgewidth=1.5, zorder=10)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if ax != ax1:
        ax.get_legend().remove()

# Get overall y-axis limits
all_diffs = diff_df['Difference']
y_min = all_diffs.min()
y_max = all_diffs.max()
y_margin = (y_max - y_min) * 0.1
y_limits = [y_min - y_margin, y_max + y_margin]

# Plot 1: All volumes
create_boxplot(diff_df, ax1, 'all volumes')
ax1.set_ylim(y_limits)

# Plot 2: Small volumes
small_vol_data = diff_df[diff_df.apply(lambda x: [x['MRN'], x['ptvs']] in small_vol_mrn_ptvs, axis=1)]
create_boxplot(small_vol_data, ax2, f'volume ≤ {median_volume:.1f} cm³')
ax2.set_ylim(y_limits)

# Plot 3: Large volumes
large_vol_data = diff_df[diff_df.apply(lambda x: [x['MRN'], x['ptvs']] in large_vol_mrn_ptvs, axis=1)]
create_boxplot(large_vol_data, ax3, f'volume > {median_volume:.1f} cm³')
ax3.set_ylim(y_limits)
ax3.set_xlabel('setup error', labelpad=-5)

# Add common y-label in the middle
fig.text(0.08, 0.55, 'relative difference to no setup error (%)', 
          va='center', rotation='vertical')

# Move legend outside to top right
legend = ax1.legend(title='metric', bbox_to_anchor=(1.02, 1.055), loc='upper left')
legend.get_frame().set_alpha(0.5)

# Adjust layout to prevent label cutoff
plt.tight_layout()
plt.subplots_adjust(left=0.15, right=0.85)

# Save figure
output_fig1 = os.path.join(config.OUTPUT_DIR, 'ci_gi_percentage_diff.png')
fig.savefig(output_fig1, dpi=600, bbox_inches='tight')
print(f"\nFigure saved to '{output_fig1}'")

# Calculate statistics for CI and GI differences
print("\n=== Statistics for CI/GI Percentage Differences ===")
stats_by_setup = []
for setup in diff_df['Setup'].unique():
    setup_data = diff_df[diff_df['Setup'] == setup]
    stats = {}
    for metric in ['CI', 'GI']:
        metric_data = setup_data[setup_data['Metric'] == metric]['Difference']
        if len(metric_data) > 0:
            stats.update({
                f'{metric} Min': metric_data.min(),
                f'{metric} Max': metric_data.max(),
                f'{metric} Median': metric_data.median(),
                f'{metric} Mean': metric_data.mean(),
                f'{metric} Std Dev': metric_data.std()
            })
    stats_by_setup.append(pd.Series(stats, name=setup))

# Create DataFrame with all statistics
ci_gi_stats = pd.DataFrame(stats_by_setup)
ci_gi_stats = ci_gi_stats.round(3)
print("\nCI/GI Statistics by Setup Error:")
print(ci_gi_stats)

# Save statistics to Excel
output_stats1 = os.path.join(config.OUTPUT_DIR, 'ci_gi_statistics.xlsx')
ci_gi_stats.to_excel(output_stats1)
print(f"\nStatistics saved to '{output_stats1}'")

# ===== COMBINED PLOT: CI, GI, D98% =====
print("\n=== Creating Combined Plot (CI, GI, D98%) ===")

# Get D98% from metrics data
df_combined = df_merged.copy()
df_combined = df_combined.dropna(subset=['PaddickCI', 'GI', 'D98%[%]'])

# Convert all to numeric
df_combined['PaddickCI'] = pd.to_numeric(df_combined['PaddickCI'].astype(str).str.replace('*', ''), errors='coerce')
df_combined['GI'] = pd.to_numeric(df_combined['GI'].astype(str).str.replace('*', ''), errors='coerce')
df_combined['D98%[%]'] = pd.to_numeric(df_combined['D98%[%]'], errors='coerce')

# Map Tx to Setup labels
df_combined['Setup'] = df_combined['Tx'].map(tx_mapping)

# Calculate percentage differences from HA_1 reference
ha1_ref = df_combined[df_combined['Tx'] == '1'].set_index(['MRN', 'ptvs'])

combined_diff_data = []
for tx in ['2', '3', '4', '5', '6']:
    tx_data = df_combined[df_combined['Tx'] == tx].set_index(['MRN', 'ptvs'])
    
    for metric_name, metric_col in [
        ('CI', 'PaddickCI'),
        ('GI', 'GI'),
        ('D98%', 'D98%[%]')
    ]:
        for (mrn, ptv) in tx_data.index:
            if (mrn, ptv) in ha1_ref.index and (mrn, ptv) in tx_data.index:
                ref_val = ha1_ref.loc[(mrn, ptv), metric_col]
                tx_val = tx_data.loc[(mrn, ptv), metric_col]
                
                if pd.notna(ref_val) and pd.notna(tx_val) and ref_val != 0:
                    pct_diff = ((tx_val - ref_val) / ref_val) * 100
                    
                    combined_diff_data.append({
                        'MRN': mrn,
                        'ptvs': ptv,
                        'Setup': tx_mapping[tx],
                        'Metric': metric_name,
                        'Difference': pct_diff,
                        'Ref_Value': ref_val,
                        'Tx_Value': tx_val
                    })

combined_diff_df = pd.DataFrame(combined_diff_data)
print(f"Combined diff data shape: {combined_diff_df.shape}")
print(f"Rows per metric:")
for m in ['CI', 'GI', 'D98%']:
    print(f"  {m}: {(combined_diff_df['Metric'] == m).sum()}")

# Create combined plot with 3x1 subplots (all volumes, small, large)
median_vol_combined = df_combined[df_combined['Tx'] == '1']['volume'].median()
small_vol_idx = df_combined[(df_combined['Tx'] == '1') & (df_combined['volume'] <= median_vol_combined)][['MRN', 'ptvs']].values.tolist()
large_vol_idx = df_combined[(df_combined['Tx'] == '1') & (df_combined['volume'] > median_vol_combined)][['MRN', 'ptvs']].values.tolist()

fig_comb, (ax1_comb, ax2_comb, ax3_comb) = plt.subplots(3, 1, figsize=(12, 6.5), sharex=True, sharey=True)

# Colors for 3 metrics
colors_comb = {'CI': '#1f77b4', 'GI': '#ff7f0e', 'D98%': '#2ca02c'}

def create_combined_boxplot(data, ax, title):
    sns.boxplot(data=data, x='Setup', y='Difference', hue='Metric', 
                palette=colors_comb, width=0.8, ax=ax)
    
    # Add mean markers
    for setup in data['Setup'].unique():
        setup_data = data[data['Setup'] == setup]
        for metric in setup_data['Metric'].unique():
            metric_data = setup_data[setup_data['Metric'] == metric]
            mean_val = metric_data['Difference'].mean()
            
            setup_idx = list(data['Setup'].unique()).index(setup)
            metric_order = ['CI', 'GI', 'D98%']
            if metric in metric_order:
                offset = (metric_order.index(metric) - 1.5) * 0.15
                x_pos = setup_idx + offset
                ax.plot(x_pos, mean_val, 's', color='white', markeredgecolor='black', 
                       markersize=0, markeredgewidth=1.5, zorder=10)
    
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title)
    ax.tick_params(axis='x', rotation=45)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    if ax != ax1_comb:
        ax.get_legend().remove()

# Get y-limits
all_comb_diffs = combined_diff_df['Difference']
y_min_comb = all_comb_diffs.min()
y_max_comb = all_comb_diffs.max()
y_margin_comb = (y_max_comb - y_min_comb) * 0.1
y_limits_comb = [y_min_comb - y_margin_comb, y_max_comb + y_margin_comb]

# Plot 1: All volumes
create_combined_boxplot(combined_diff_df, ax1_comb, '')
ax1_comb.set_ylim(y_limits_comb)

# Plot 2: Small volumes
small_comb_data = combined_diff_df[combined_diff_df.apply(lambda x: [x['MRN'], x['ptvs']] in small_vol_idx, axis=1)]
create_combined_boxplot(small_comb_data, ax2_comb, f'volume ≤ {median_vol_combined:.1f} cm³')
ax2_comb.set_ylim(y_limits_comb)

# Plot 3: Large volumes
large_comb_data = combined_diff_df[combined_diff_df.apply(lambda x: [x['MRN'], x['ptvs']] in large_vol_idx, axis=1)]
create_combined_boxplot(large_comb_data, ax3_comb, f'volume > {median_vol_combined:.1f} cm³')
ax3_comb.set_ylim(y_limits_comb)
ax3_comb.set_xlabel('setup error', labelpad=-5)

# Common y-label
fig_comb.text(0.06, 0.55, 'relative difference to no setup error (%)', 
              va='center', rotation='vertical')

# Legend - placed at top of figure
legend_comb = ax1_comb.legend(title='metric', bbox_to_anchor=(0.5, 1.15), loc='lower center', ncol=3)
legend_comb.get_frame().set_alpha(0.5)

plt.tight_layout()
plt.subplots_adjust(left=0.12, right=0.95, top=0.90)

output_fig2 = os.path.join(config.OUTPUT_DIR, 'combined_ci_gi_d98_percentage_diff.png')
fig_comb.savefig(output_fig2, dpi=600, bbox_inches='tight')
print(f"\nCombined figure saved to '{output_fig2}'")

# Statistics for combined metrics
print("\n=== Statistics for Combined Metrics ===")
combined_stats = []
for setup in combined_diff_df['Setup'].unique():
    setup_data = combined_diff_df[combined_diff_df['Setup'] == setup]
    stats_row = {'Setup': setup}
    for metric in ['CI', 'GI', 'D98%']:
        metric_data = setup_data[setup_data['Metric'] == metric]['Difference']
        if len(metric_data) > 0:
            stats_row[f'{metric}_Mean'] = metric_data.mean()
            stats_row[f'{metric}_Std'] = metric_data.std()
            stats_row[f'{metric}_Median'] = metric_data.median()
    combined_stats.append(stats_row)

combined_stats_df = pd.DataFrame(combined_stats)
combined_stats_df = combined_stats_df.round(3)
print("\nCombined statistics:")
print(combined_stats_df.to_string(index=False))
output_stats2 = os.path.join(config.OUTPUT_DIR, 'combined_metrics_statistics.xlsx')
combined_stats_df.to_excel(output_stats2, index=False)
print(f"\nCombined statistics saved to '{output_stats2}'")

# ===== ADDITIONAL ANALYSES FOR REVIEWER =====

# Figure: Absolute CI and GI values across setup errors (similar to dose metrics plot)
fig_abs, ax_abs = plt.subplots(1, 1, figsize=(10, 5))

# Prepare data for absolute values plot
df_abs = df.copy()
df_abs_ci = df_abs[['Setup', 'PaddickCI']].copy()
df_abs_ci['Metric'] = 'Paddick CI'
df_abs_ci['Value'] = df_abs_ci['PaddickCI']
df_abs_gi = df_abs[['Setup', 'GI']].copy()
df_abs_gi['Metric'] = 'Gradient Index'
df_abs_gi['Value'] = df_abs_gi['GI']
df_abs_plot = pd.concat([
    df_abs_ci[['Setup', 'Metric', 'Value']],
    df_abs_gi[['Setup', 'Metric', 'Value']]
])

# Create boxplot for absolute values
colors_abs = {'Paddick CI': '#1f77b4', 'Gradient Index': '#ff7f0e'}
sns.boxplot(data=df_abs_plot, x='Setup', y='Value', hue='Metric', 
            palette=colors_abs, width=0.7, ax=ax_abs)

# Add mean markers
for i, setup in enumerate(df_abs_plot['Setup'].unique()):
    for metric in ['Paddick CI', 'Gradient Index']:
        data_subset = df_abs_plot[(df_abs_plot['Setup'] == setup) & (df_abs_plot['Metric'] == metric)]
        if len(data_subset) > 0:
            mean_val = data_subset['Value'].mean()
            x_pos = i - 0.2 if metric == 'Paddick CI' else i + 0.2
            ax_abs.plot(x_pos, mean_val, 's', color='white', markeredgecolor='black', 
                       markersize=0, markeredgewidth=1.5, zorder=10)

ax_abs.set_xlabel('setup error')
ax_abs.set_ylabel('index value')
ax_abs.set_title('Conformity and Gradient Indices Across Setup Errors')
ax_abs.tick_params(axis='x', rotation=45)
ax_abs.legend(title='metric', bbox_to_anchor=(1.02, 1), loc='upper left')

plt.tight_layout()
plt.subplots_adjust(right=0.85)
output_fig3 = os.path.join(config.OUTPUT_DIR, 'ci_gi_absolute_values.png')
fig_abs.savefig(output_fig3, dpi=600, bbox_inches='tight')
print(f"\nAbsolute values figure saved to '{output_fig3}'")

# Correlation Analysis: CI and GI vs Setup Error Magnitude
print("\n=== Correlation Analysis: CI and GI vs Setup Error Magnitude ===")

# Define setup error magnitudes (approximate translation magnitude in mm)
setup_magnitude = {
    '1': 0,      # no error
    '2': 0.5,    # 0.5mm
    '3': 0.7,    # 0.7mm  
    '4': 1.0,    # 1mm
    '5': 0,      # rotation only (0mm translation)
    '6': 1.0     # 1mm translation only
}

df_corr = df.copy()
df_corr['SetupMag'] = df_corr['Tx'].map(setup_magnitude)

# Calculate correlations
from scipy.stats import pearsonr, ttest_rel
ci_corr = pearsonr(df_corr['SetupMag'], df_corr['PaddickCI'])
gi_corr = pearsonr(df_corr['SetupMag'], df_corr['GI'])

print(f"Paddick CI vs Setup Error Magnitude: r = {ci_corr[0]:.4f}, p = {ci_corr[1]:.4e}")
print(f"Gradient Index vs Setup Error Magnitude: r = {gi_corr[0]:.4f}, p = {gi_corr[1]:.4e}")

# Statistical Tests: Compare each setup to HA_1 (reference)
print("\n=== Statistical Tests: Each Setup vs Reference ===")

ha1_data = df[df['Tx'] == '1']
setup_tests = []

for tx in ['2', '3', '4', '5', '6']:
    tx_data = df[df['Tx'] == tx]
    
    # Match PTVs between setups for paired test
    ha1_matched = ha1_data.set_index(['MRN', 'ptvs'])
    tx_matched = tx_data.set_index(['MRN', 'ptvs'])
    
    # Get common indices
    common_idx = ha1_matched.index.intersection(tx_matched.index)
    
    if len(common_idx) > 0:
        # CI paired t-test
        ha1_ci = ha1_matched.loc[common_idx, 'PaddickCI']
        tx_ci = tx_matched.loc[common_idx, 'PaddickCI']
        ci_tstat, ci_pval = ttest_rel(tx_ci, ha1_ci)
        
        # GI paired t-test
        ha1_gi = ha1_matched.loc[common_idx, 'GI']
        tx_gi = tx_matched.loc[common_idx, 'GI']
        gi_tstat, gi_pval = ttest_rel(tx_gi, ha1_gi)
        
        setup_tests.append({
            'Setup': tx_mapping[tx],
            'CI_p_value': ci_pval,
            'CI_significant': ci_pval < 0.05,
            'GI_p_value': gi_pval,
            'GI_significant': gi_pval < 0.05,
            'N_pairs': len(common_idx)
        })

test_df = pd.DataFrame(setup_tests)
print("\nPaired t-test results (Setup vs 1 Reference):")
print(test_df.to_string(index=False))

# Save test results
output_test = os.path.join(config.OUTPUT_DIR, 'ci_gi_statistical_tests.xlsx')
test_df.to_excel(output_test, index=False)
print(f"\nStatistical test results saved to '{output_test}'")

# Comprehensive Summary Table
print("\n=== Comprehensive Summary: All Metrics by Setup ===")

summary_data = []
for tx in ['1', '2', '3', '4', '5', '6']:
    tx_data = df[df['Tx'] == tx]
    
    summary_data.append({
        'Setup': tx_mapping[tx],
        'N': len(tx_data),
        'CI_Mean': tx_data['PaddickCI'].mean(),
        'CI_Std': tx_data['PaddickCI'].std(),
        'CI_Median': tx_data['PaddickCI'].median(),
        'GI_Mean': tx_data['GI'].mean(),
        'GI_Std': tx_data['GI'].std(),
        'GI_Median': tx_data['GI'].median()
    })

summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.round(4)
print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

output_summary = os.path.join(config.OUTPUT_DIR, 'ci_gi_summary_by_setup.xlsx')
summary_df.to_excel(output_summary, index=False)
print(f"\nSummary saved to '{output_summary}'")

# ICRU 91 / ISRS Compliance Analysis
print("\n=== ICRU 91 / ISRS Standards Compliance Analysis ===")
print("Note: ICRU 91 recommends CI > 0.6 for SRS, ISRS suggests GI < 3-4 for optimal conformity")

# Check compliance thresholds
CI_THRESHOLD = 0.6  # ICRU 91 recommendation
GI_THRESHOLD = 4.0  # ISRS recommendation (lower is better)

compliance_data = []
for tx in ['1', '2', '3', '4', '5', '6']:
    tx_data = df[df['Tx'] == tx]
    
    ci_compliant = (tx_data['PaddickCI'] >= CI_THRESHOLD).sum()
    gi_compliant = (tx_data['GI'] <= GI_THRESHOLD).sum()
    total = len(tx_data)
    
    compliance_data.append({
        'Setup': tx_mapping[tx],
        'Total_PTVs': total,
        'CI_ge_0.6': ci_compliant,
        'CI_Compliance_%': 100 * ci_compliant / total,
        'GI_le_4': gi_compliant,
        'GI_Compliance_%': 100 * gi_compliant / total,
        'Both_Compliant': ((tx_data['PaddickCI'] >= CI_THRESHOLD) & 
                          (tx_data['GI'] <= GI_THRESHOLD)).sum(),
        'Both_Compliance_%': 100 * ((tx_data['PaddickCI'] >= CI_THRESHOLD) & 
                                    (tx_data['GI'] <= GI_THRESHOLD)).sum() / total
    })

compliance_df = pd.DataFrame(compliance_data)
compliance_df = compliance_df.round(1)
print("\nCompliance with ICRU 91 (CI ≥ 0.6) and ISRS (GI ≤ 4.0) Standards:")
print(compliance_df.to_string(index=False))

output_compliance = os.path.join(config.OUTPUT_DIR, 'ci_gi_compliance_analysis.xlsx')
compliance_df.to_excel(output_compliance, index=False)
print(f"\nCompliance analysis saved to '{output_compliance}'")

# ===== SUPPLEMENTARY: COMPREHENSIVE METRICS ANALYSIS =====
print("\n" + "="*60)
print("SUPPLEMENTARY ANALYSIS - All Metrics with V100% and D50")
print("="*60)

# Create comprehensive supplementary dataset with all available metrics
df_supp = df_merged.copy()

# Ensure all numeric columns are properly converted
numeric_cols = ['PaddickCI', 'RTOG_CI', 'GI', 'HI', 'Coverage_pct', 'D2_Gy', 'D50_Gy', 'D98_Gy', 'Dmax_Gy', 'V12Gy_cc']
for col in numeric_cols:
    if col in df_supp.columns:
        df_supp[col] = pd.to_numeric(df_supp[col].astype(str).str.replace('*', ''), errors='coerce')

# Map D98%[%] if exists, otherwise use D98_Gy
if 'D98%[%]' in df_supp.columns:
    df_supp['D98_pct'] = pd.to_numeric(df_supp['D98%[%]'], errors='coerce')
else:
    df_supp['D98_pct'] = df_supp['D98_Gy']

# Create comprehensive metrics table by setup
print("\n=== Supplementary Table: All Metrics by Setup ===")

supp_metrics_all = []
all_metrics = ['PaddickCI', 'RTOG_CI', 'GI', 'HI', 'Coverage_pct', 'D2_Gy', 'D50_Gy', 'D98_Gy', 'Dmax_Gy', 'V12Gy_cc']

for tx in ['1', '2', '3', '4', '5', '6']:
    tx_data = df_supp[df_supp['Tx'] == tx]
    if len(tx_data) == 0:
        continue

    row = {'Setup': tx_mapping[tx], 'N': len(tx_data)}

    for metric in all_metrics:
        if metric in tx_data.columns:
            vals = tx_data[metric].dropna()
            if len(vals) > 0:
                row[f'{metric}_Mean'] = vals.mean()
                row[f'{metric}_Std'] = vals.std()
                row[f'{metric}_Median'] = vals.median()
                row[f'{metric}_Min'] = vals.min()
                row[f'{metric}_Max'] = vals.max()

    supp_metrics_all.append(row)

supp_all_df = pd.DataFrame(supp_metrics_all)
supp_all_df = supp_all_df.round(4)
print("\nComprehensive Metrics Summary (Supplementary):")
print(supp_all_df.to_string(index=False))

output_supp_all = os.path.join(config.OUTPUT_DIR, 'supplementary_all_metrics.xlsx')
supp_all_df.to_excel(output_supp_all, index=False)
print(f"\nSupplementary metrics saved to '{output_supp_all}'")

# Create percentage difference data for supplementary metrics (including V100% and D50)
print("\n=== Supplementary Percentage Differences (including V100%, D50, D2, Dmax) ===")

supp_diff_metrics = [
    ('CI', 'PaddickCI'),
    ('GI', 'GI'),
    ('D98%', 'D98_Gy'),
    ('V100%', 'Coverage_pct'),
    ('D50%', 'D50_Gy'),
    ('D2%', 'D2_Gy'),
    ('Dmax%', 'Dmax_Gy'),
    ('HI', 'HI'),
    ('RTOG_CI', 'RTOG_CI')
]

ha1_supp = df_supp[df_supp['Tx'] == '1'].set_index(['MRN', 'ptvs'])
supp_diff_data = []

for tx in ['2', '3', '4', '5', '6']:
    tx_data = df_supp[df_supp['Tx'] == tx].set_index(['MRN', 'ptvs'])

    for metric_name, metric_col in supp_diff_metrics:
        if metric_col not in df_supp.columns:
            continue

        for (mrn, ptv) in tx_data.index:
            if (mrn, ptv) in ha1_supp.index and (mrn, ptv) in tx_data.index:
                ref_val = ha1_supp.loc[(mrn, ptv), metric_col]
                tx_val = tx_data.loc[(mrn, ptv), metric_col]

                if pd.notna(ref_val) and pd.notna(tx_val) and ref_val != 0:
                    pct_diff = ((tx_val - ref_val) / ref_val) * 100

                    supp_diff_data.append({
                        'MRN': mrn,
                        'ptvs': ptv,
                        'Setup': tx_mapping[tx],
                        'Metric': metric_name,
                        'Difference': pct_diff,
                        'Ref_Value': ref_val,
                        'Tx_Value': tx_val
                    })

supp_diff_df = pd.DataFrame(supp_diff_data)
print(f"\nSupplementary diff data shape: {supp_diff_df.shape}")
for m in ['CI', 'GI', 'D98%', 'V100%', 'D50%', 'D2%', 'Dmax%', 'HI', 'RTOG_CI']:
    count = (supp_diff_df['Metric'] == m).sum()
    if count > 0:
        print(f"  {m}: {count}")

# Create supplementary figure with multiple subplots (3x3 grid)
print("\n=== Creating Supplementary Multi-Metric Figure ===")

fig_supp, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
axes = axes.flatten()

# Define metrics for each subplot with colors
supp_plot_metrics = [
    ('CI', '#1f77b4'),
    ('GI', '#ff7f0e'),
    ('D98%', '#2ca02c'),
    ('V100%', '#d62728'),
    ('D50%', '#9467bd'),
    ('D2%', '#8c564b'),
    ('Dmax%', '#e377c2'),
    ('HI', '#7f7f7f'),
    ('RTOG_CI', '#bcbd22')
]

for idx, (metric_name, color) in enumerate(supp_plot_metrics):
    ax = axes[idx]
    metric_data = supp_diff_df[supp_diff_df['Metric'] == metric_name]

    if len(metric_data) == 0:
        ax.set_visible(False)
        continue

    # Create boxplot for this metric
    sns.boxplot(data=metric_data, x='Setup', y='Difference', color=color, width=0.6, ax=ax)

    # Add mean markers
    for i, setup in enumerate(metric_data['Setup'].unique()):
        setup_data = metric_data[metric_data['Setup'] == setup]
        mean_val = setup_data['Difference'].mean()
        ax.plot(i, mean_val, 's', color='white', markeredgecolor='black',
               markersize=6, markeredgewidth=1.5, zorder=10)

    ax.set_title(f'{metric_name}', fontsize=11, fontweight='bold')
    ax.set_xlabel('')
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

    # Set consistent y-axis label
    if idx % 3 == 0:
        ax.set_ylabel('% difference', fontsize=9)
    else:
        ax.set_ylabel('')

fig_supp.suptitle('Supplementary: Percentage Difference in All Metrics vs Reference (1)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

output_supp_fig = os.path.join(config.OUTPUT_DIR, 'supplementary_all_metrics_subplots.png')
fig_supp.savefig(output_supp_fig, dpi=600, bbox_inches='tight')
print(f"\nSupplementary figure saved to '{output_supp_fig}'")

# Supplementary statistics table (must be defined before PDF creation) - WITHOUT MEDIAN
print("\n=== Supplementary Statistics by Metric ===")
supp_stats = []
for setup in supp_diff_df['Setup'].unique():
    setup_data = supp_diff_df[supp_diff_df['Setup'] == setup]
    stats_row = {'Setup': setup}
    for metric in ['CI', 'GI', 'D98%', 'V100%', 'D50%', 'D2%', 'Dmax%', 'HI', 'RTOG_CI']:
        metric_data = setup_data[setup_data['Metric'] == metric]['Difference']
        if len(metric_data) > 0:
            stats_row[f'{metric}_Mean'] = metric_data.mean()
            stats_row[f'{metric}_Std'] = metric_data.std()
            # Median removed as requested
    supp_stats.append(stats_row)

supp_stats_df = pd.DataFrame(supp_stats)
supp_stats_df = supp_stats_df.round(3)
print("\nSupplementary statistics:")
print(supp_stats_df.to_string(index=False))

output_supp_stats = os.path.join(config.OUTPUT_DIR, 'supplementary_statistics.xlsx')
supp_stats_df.to_excel(output_supp_stats, index=False)
print(f"\nSupplementary statistics saved to '{output_supp_stats}'")

# Create multi-page PDF with figure and tables (after all data is computed)
from matplotlib.backends.backend_pdf import PdfPages

output_supp_pdf = os.path.join(config.OUTPUT_DIR, 'supplementary_all_metrics.pdf')
with PdfPages(output_supp_pdf) as pdf:
    # Page 1: The figure
    fig_supp.suptitle('Supplementary Figure: Percentage Difference in All Metrics vs Reference (1)', fontsize=14, fontweight='bold', y=1.02)
    pdf.savefig(fig_supp, dpi=300, bbox_inches='tight')
    print(f"\nSupplementary page 1 (figure) added to PDF")

    # Page 2: Comprehensive metrics table - VERY WIDE, low row height
    fig_table1, ax_table1 = plt.subplots(figsize=(18, 9))  # Extra wide, same for both tables
    ax_table1.axis('tight')
    ax_table1.axis('off')
    ax_table1.set_title('Supplementary Table 1: Comprehensive Metrics by Setup', fontsize=13, fontweight='bold', pad=15)

    # Prepare display columns
    display_cols = ['Setup', 'N']
    for metric in ['PaddickCI', 'RTOG_CI', 'GI', 'HI', 'Coverage_pct', 'D50_Gy', 'D98_Gy', 'Dmax_Gy']:
        if f'{metric}_Mean' in supp_all_df.columns:
            display_cols.append(f'{metric}_Mean')
        if f'{metric}_Std' in supp_all_df.columns:
            display_cols.append(f'{metric}_Std')

    display_cols = [c for c in display_cols if c in supp_all_df.columns]
    table_data = supp_all_df[display_cols].round(3)

    # Create three-line headers: Metric name / Statistic / Unit
    col_labels = ['Setup', 'N']
    for col in display_cols[2:]:  # Skip Setup and N
        # Determine base metric name and unit
        if 'PaddickCI' in col or 'RTOG_CI' in col or '_CI' in col:
            metric_name = col.replace('_Mean', '').replace('_Std', '').replace('Paddick', '')
            unit = '(–)'
        elif 'GI' in col:
            metric_name = 'GI'
            unit = '(–)'
        elif 'HI' in col:
            metric_name = 'HI'
            unit = '(–)'
        elif 'Coverage_pct' in col or 'V100%' in col:
            metric_name = 'V100%'
            unit = '(%)'
        elif '_Gy' in col:
            metric_name = col.replace('_Mean', '').replace('_Std', '').replace('_Gy', '')
            unit = '(Gy)'
        elif 'V12Gy_cc' in col:
            metric_name = 'V12Gy'
            unit = '(cc)'
        else:
            metric_name = col.replace('_Mean', '').replace('_Std', '')
            unit = ''

        if '_Mean' in col:
            col_labels.append(f'{metric_name}\nMean\n{unit}')
        elif '_Std' in col:
            col_labels.append(f'{metric_name}\nSD\n{unit}')
        else:
            col_labels.append(col)

    table1 = ax_table1.table(
        cellText=table_data.values,
        colLabels=col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.01, 0.08, 0.98, 0.84]
    )
    table1.auto_set_font_size(False)
    table1.set_fontsize(6)
    table1.scale(1, 0.9)  # Lower row height

    # Style header row - match content font size
    for i in range(len(col_labels)):
        table1[(0, i)].set_facecolor('#4472C4')
        table1[(0, i)].set_text_props(weight='bold', color='white', size=6)

    pdf.savefig(fig_table1, dpi=300, bbox_inches='tight', orientation='landscape')
    plt.close(fig_table1)
    print(f"Supplementary page 2 (metrics table) added to PDF")

    # Page 3: Statistics table - SAME WIDTH as page 2, low row height
    fig_table2, ax_table2 = plt.subplots(figsize=(18, 9))  # Same extra wide size
    ax_table2.axis('tight')
    ax_table2.axis('off')
    ax_table2.set_title('Supplementary Table 2: Percentage Difference Statistics', fontsize=13, fontweight='bold', pad=15)

    # Create three-line headers: Metric name / Statistic / Unit
    # For percentage difference table, unit is always %
    stats_col_labels = ['Setup']
    for col in supp_stats_df.columns:
        if col == 'Setup':
            continue
        # Parse column name and add % as unit (all are percentage differences)
        if '_Mean' in col:
            metric_name = col.replace('_Mean', '')
            stats_col_labels.append(f'{metric_name}\nMean\n(%)')
        elif '_Std' in col:
            metric_name = col.replace('_Std', '')
            stats_col_labels.append(f'{metric_name}\nSD\n(%)')
        else:
            stats_col_labels.append(col)

    table2 = ax_table2.table(
        cellText=supp_stats_df.round(2).values,
        colLabels=stats_col_labels,
        cellLoc='center',
        loc='center',
        bbox=[0.01, 0.08, 0.98, 0.84]
    )
    table2.auto_set_font_size(False)
    table2.set_fontsize(6)
    table2.scale(1, 0.9)  # Lower row height

    # Style header row - match content font size
    for i in range(len(stats_col_labels)):
        table2[(0, i)].set_facecolor('#4472C4')
        table2[(0, i)].set_text_props(weight='bold', color='white', size=6)

    pdf.savefig(fig_table2, dpi=300, bbox_inches='tight', orientation='landscape')
    plt.close(fig_table2)
    print(f"Supplementary page 3 (statistics table) added to PDF")

print(f"\nComplete supplementary PDF saved to '{output_supp_pdf}' (3 pages)")

print("\n" + "="*60)
print("ANALYSIS COMPLETE - All requested metrics generated")
print("="*60)
