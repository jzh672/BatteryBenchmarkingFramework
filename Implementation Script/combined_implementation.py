import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
R = 8.314
T_REF = 298.15
C_REF = 1.0
SOH_EOL = 0.8

def calc_temp_stress(T_celsius, Ea):
    return np.exp((Ea / R) * ((1 / T_REF) - (1 / (T_celsius + 273.15))))

def calc_crate_stress(C_rate, beta):
    return (C_rate / C_REF) ** beta

def predict_soh(X, Ea, beta, k, b):
    Ah_norm, T_celsius, C_rate = X
    phi_t = calc_temp_stress(T_celsius, Ea)
    phi_c = calc_crate_stress(C_rate, beta)
    Ah_eff = Ah_norm * phi_t * phi_c
    return np.maximum(1 - k * (Ah_eff ** b), 0)

p0 = [1000, 2.0, 0.01, 0.4]
bounds = ([100, 0.01, 1e-6, 0.01], [3000, 20, 5, 5])
N_BOOT = 500  # reduced from 1000 since this now runs 3x (once per chemistry)

def keep_main_cycling_phase(cell_df):
    """Keep the longest uninterrupted C-rate block for one cell."""
    g = cell_df.sort_values("cycle_number").copy()

    # Round only to prevent tiny floating-point differences from creating false transitions.
    rate = g["c_rate"].round(1)
    run_id = rate.ne(rate.shift()).cumsum()

    run_lengths = run_id.value_counts()
    main_run = run_lengths.idxmax()  # longest C-rate block
    main = g.loc[run_id == main_run].copy()

    # Audit fields: retain these so exclusions are traceable.
    main_rate = rate.loc[main.index].iloc[0]
    start_cycle = int(main["cycle_number"].iloc[0])

    main["main_c_rate"] = main_rate
    main["phase_start_cycle"] = start_cycle
    main["cycle_number_main"] = (
            main["cycle_number"] - start_cycle + 1
    )

    # Rebuild throughput from the retained phase.
    # For constant DoD = 1, this becomes 0, 1, 2, ...
    dod = main["depth_of_discharge"].astype(float)
    main["Ah_norm_main"] = dod.cumsum() - dod.iloc[0]

    # The model forces SOH(Ah_norm=0) = 1.
    # Re-normalize to the first retained main-phase capacity.
    main["SOH_main"] = main["SOH"] / main["SOH"].iloc[0]

    return main

def filtering_critical_cols(df):
    '''
    Create an audit file for record and filter out unnecessay columns in Dataframe
    '''
    # Preserve an audit file: important for the methods section.
    audit = (df.groupby("cell_id", as_index=False).agg(
        main_c_rate=("main_c_rate", "first"),
        phase_start_cycle=("phase_start_cycle", "first"),
        retained_cycles=("cycle_number_main", "size"),
    )
    )
    print(audit.sort_values("phase_start_cycle").head(20))

    # Keep only the derived main-cycling variables, then rename them.
    df = df[
        [
            "cycle_number_main",
            "mean_temperature_in_C",
            "c_rate",
            "Ah_norm_main",
            "SOH_main",
            "cell_id",
            "depth_of_discharge",
            'cathode_material'
        ]
    ].rename(columns={
        "cycle_number_main": "cycle_number",
        "Ah_norm_main": "Ah_norm",
        "SOH_main": "SOH",
    }).dropna()

    df['cell_id'] = df['cell_id'].astype(str)

    return df


# Li-ion Dataset
li_ion_raw_df = pd.read_csv(
    "D:/Jason/UTSG/Summer Research/Codes for testing/Phase 1 Implementation/li_ion_all_data_combined.csv")  # Enter dataset directory
li_df = (li_ion_raw_df.groupby("cell_id", group_keys=False).apply(keep_main_cycling_phase).reset_index(drop=True))
li_df_filtered = filtering_critical_cols(li_df)

# Split base on chemistry
LFP_df_filtered = li_df_filtered[li_df_filtered['cathode_material'] == 'LFP']
NCA_df_filtered = li_df_filtered[li_df_filtered['cathode_material'] == 'NCA']
NMC_df_filtered = li_df_filtered[li_df_filtered['cathode_material'] == 'NMC']

# Na-ion Dataset (NEW: RWTH Klick et al. data, -10C to 50C, extracted from
# raw BaSyTec check-up files -- see naion_modeling_ready.csv)
#
# NOTE: this dataset is NOT run through keep_main_cycling_phase(). That
# function assumes one row = one cycle (it builds Ah_norm as a row-count
# cumsum of depth_of_discharge), which holds for the old densely-sampled
# datasets but NOT for this one: this dataset has only 1-14 rows per cell
# (one row per check-up, not per cycle), each already carrying a correctly
# pre-computed cumulative Ah_norm (via real Ah_counter integration from the
# raw cycling logs). Running it through keep_main_cycling_phase would
# silently recompute Ah_norm as "row count" instead, collapsing real
# throughput values as high as ~4800 Ah down to single digits -- no error,
# just quietly wrong data. So we skip straight to the same output schema
# filtering_critical_cols() produces, using the already-correct columns.
na_ion_raw_df = pd.read_csv(
    "D:/Jason/UTSG/Summer Research/Dataset/naion_modeling_ready_filtered.csv")  # Enter dataset directory
na_df_filtered = na_ion_raw_df[
    [
        "cycle_number",
        "mean_temperature_in_C",
        "c_rate",
        "Ah_norm",
        "SOH",
        "cell_id",
        "depth_of_discharge",
        "cathode_material",
    ]
].dropna()
na_df_filtered['cell_id'] = na_df_filtered['cell_id'].astype(str)

# Zn-ion Dataset
zn_ion_raw_df = pd.read_csv(
    "D:/Jason/UTSG/Summer Research/Dataset/Combined/all_data_combined_zn_coin.csv")  # Enter dataset directory -- CONFIRM THIS PATH
zn_df = (zn_ion_raw_df.groupby("cell_id", group_keys=False).apply(keep_main_cycling_phase).reset_index(drop=True))
zn_df_filtered = filtering_critical_cols(zn_df)

# ---------------------------------------------------------------
# Exploratory Data Analysis
# ---------------------------------------------------------------
def plot_coverage_subplots(df_list, titles, temp_col='mean_temperature_in_C', rate_col='c_rate'):
    """
    Create subplots for test coverage of multiple datasets.

    Parameters:
    - df_list: List of DataFrames
    - titles: List of titles for each subplot
    """
    fig, axes = plt.subplots(1, 5, figsize=(28, 6), constrained_layout=True)
    fig.suptitle('Test Coverage: Temperature × C-rate', fontsize=18, fontweight='bold')

    for idx, (df, title) in enumerate(zip(df_list, titles)):
        # Create 2D histogram
        temp_bins = np.linspace(df[temp_col].min(), df[temp_col].max(), 10)
        rate_bins = np.linspace(df[rate_col].min(), df[rate_col].max(), 10)

        coverage, _, _ = np.histogram2d(df[temp_col], df[rate_col], bins=[temp_bins, rate_bins])

        # Plot on subplot with extent to show actual values
        extent = [temp_bins[0], temp_bins[-1], rate_bins[0], rate_bins[-1]]
        im = axes[idx].imshow(coverage.T, cmap='YlOrRd', aspect='auto', origin='lower', extent=extent)
        axes[idx].set_title(title, fontsize=15, fontweight='bold', pad=10)
        axes[idx].set_xlabel('Temperature (°C)', fontsize=12)
        if idx == 0:
            axes[idx].set_ylabel('C-rate', fontsize=12)
        axes[idx].tick_params(labelsize=10)

        # Add colorbar to each subplot
        cbar = fig.colorbar(im, ax=axes[idx], label='Count', shrink=0.85)
        cbar.ax.tick_params(labelsize=9)
        cbar.set_label('Count', fontsize=10)

    plt.savefig('coverage_heatmap_poster.png', dpi=250, bbox_inches='tight')
    plt.show()


def plot_correlation_subplots(df_list, titles, columns=None):
    """
    Create correlation matrix heatmap subplots for multiple datasets.

    Parameters:
    - df_list: List of DataFrames
    - titles: List of titles for each subplot
    - columns: List of columns to include (default: all numeric)
    """
    fig, axes = plt.subplots(1, 5, figsize=(30, 6.5), constrained_layout=True)
    fig.suptitle('Feature Correlation Matrix by Chemistry', fontsize=18, fontweight='bold')

    if len(df_list) != len(titles):
        raise ValueError("Number of DataFrames must match number of titles")

    for idx, (df, title) in enumerate(zip(df_list, titles)):
        # Select columns
        if columns is not None:
            corr_df = df[columns]
        else:
            corr_df = df.select_dtypes(include=[np.number])

        # Calculate correlation matrix
        corr_matrix = corr_df.corr()

        # Plot heatmap (single shared colorbar on the last panel -- correlation
        # is always on a fixed -1 to 1 scale, so one colorbar covers all five)
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            vmin=-1,
            vmax=1,
            square=True,
            cbar=(idx == len(df_list) - 1),
            ax=axes[idx],
            annot_kws={'size': 9},
            cbar_kws={'label': 'Correlation', 'shrink': 0.85} if idx == len(df_list) - 1 else None,
            xticklabels=True,
            yticklabels=(idx == 0)
        )
        axes[idx].set_title(title, fontsize=15, fontweight='bold', pad=10)
        axes[idx].set_xlabel('')
        axes[idx].set_ylabel('')
        axes[idx].tick_params(axis='x', labelsize=9, rotation=45)
        axes[idx].tick_params(axis='y', labelsize=9, rotation=0)

    plt.savefig('correlation_matrix_poster.png', dpi=250, bbox_inches='tight')
    plt.show()


# Usage
df_list = [LFP_df_filtered, NCA_df_filtered, NMC_df_filtered, na_df_filtered, zn_df_filtered]
titles = ['Li-ion LFP', 'Li-ion NCA', 'Li-ion NMC', 'Na-ion', 'Zn-ion']
df_dictionary = {'Li-ion LFP': LFP_df_filtered, 'Li-ion NCA': NCA_df_filtered, 'Li-ion NMC': NMC_df_filtered,
                 'Na-ion': na_df_filtered, 'Zn-ion': zn_df_filtered}

plot_coverage_subplots(df_list, titles)
plot_correlation_subplots(df_list, titles)


# --------------------------
# Model Fitting
# --------------------------
def fit_chemistry(sub_df, chem_name, seed=42):
    cells = sub_df['cell_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(cells)
    n_test = max(1, int(round(0.2 * len(cells))))
    test_cells = set(cells[:n_test])
    train_cells = set(cells[n_test:])

    train_df = sub_df[sub_df['cell_id'].isin(train_cells)].reset_index(drop=True)
    test_df = sub_df[sub_df['cell_id'].isin(test_cells)].reset_index(drop=True)

    X_train = (train_df['Ah_norm'].values, train_df['mean_temperature_in_C'].values, train_df['c_rate'].values)
    y_train = train_df['SOH'].values
    X_test = (test_df['Ah_norm'].values, test_df['mean_temperature_in_C'].values, test_df['c_rate'].values)
    y_test = test_df['SOH'].values

    popt, _ = curve_fit(predict_soh, X_train, y_train, p0=p0, bounds=bounds, maxfev=20000)
    Ea, beta, k, b = popt

    # Bootstrap 95% CI (cell-grouped)
    train_cell_list = list(train_cells)
    cell_arrays = {}
    for c in train_cell_list:
        s = train_df[train_df['cell_id'] == c]
        cell_arrays[c] = (s['Ah_norm'].values, s['mean_temperature_in_C'].values, s['c_rate'].values, s['SOH'].values)

    boot_params = []
    for i in range(N_BOOT):
        sample_cells = rng.choice(train_cell_list, size=len(train_cell_list), replace=True)
        ah = np.concatenate([cell_arrays[c][0] for c in sample_cells])
        temp = np.concatenate([cell_arrays[c][1] for c in sample_cells])
        crate = np.concatenate([cell_arrays[c][2] for c in sample_cells])
        soh = np.concatenate([cell_arrays[c][3] for c in sample_cells])
        try:
            popt_i, _ = curve_fit(predict_soh, (ah, temp, crate), soh, p0=popt, bounds=bounds, maxfev=3000)
            boot_params.append(popt_i)
        except Exception:
            continue
    boot_params = np.array(boot_params)
    ci_lo = np.percentile(boot_params, 2.5, axis=0) if len(boot_params) else [np.nan] * 4
    ci_hi = np.percentile(boot_params, 97.5, axis=0) if len(boot_params) else [np.nan] * 4

    # Evaluate
    pred = predict_soh(X_test, *popt)
    resid = y_test - pred
    mae, rmse = np.mean(np.abs(resid)), np.sqrt(np.mean(resid ** 2))

    Ah_80_eff = ((1 - SOH_EOL) / k) ** (1 / b)
    n80_rows = []
    for c in test_df['cell_id'].unique():
        # Calculate N80
        s = test_df[test_df['cell_id'] == c]
        T_mean, C_mean, DoD_mean = s['mean_temperature_in_C'].mean(), s['c_rate'].mean(), s['depth_of_discharge'].mean()
        phi_t, phi_c = calc_temp_stress(T_mean, Ea), calc_crate_stress(C_mean, beta)
        Ah_80_norm = Ah_80_eff / (phi_t * phi_c)
        N_80 = Ah_80_norm / DoD_mean if DoD_mean > 0 else np.nan
        n80_rows.append({'cell_id': c, 'N_80_predicted': N_80, 'actual_cycles_observed': s['cycle_number'].max()})
    n80_df = pd.DataFrame(n80_rows)

    print(f"\n=== {chem_name} (n_cells={len(cells)}, train={len(train_cells)}, test={len(test_cells)}) ===")
    print(f"Ea={Ea:.2f} [{ci_lo[0]:.2f}, {ci_hi[0]:.2f}]  beta={beta:.4f} [{ci_lo[1]:.4f}, {ci_hi[1]:.4f}]  "
          f"k={k:.6f} [{ci_lo[2]:.6f}, {ci_hi[2]:.6f}]  b={b:.4f} [{ci_lo[3]:.4f}, {ci_hi[3]:.4f}]")
    print(f"MAE={mae:.5f}  RMSE={rmse:.5f}  bootstrap_success={len(boot_params)}/{N_BOOT}")
    print(
        f"Median N_80 predicted: {n80_df['N_80_predicted'].median():,.0f}  |  Median actual cycles observed: {n80_df['actual_cycles_observed'].median():,.0f}")

    return {
        'chemistry': chem_name, 'n_cells': len(cells), 'n_train_cells': len(train_cells),
        'n_test_cells': len(test_cells),
        'params': {'Ea': Ea, 'beta': beta, 'k': k, 'b': b},
        'bootstrap_ci': {'Ea': [float(ci_lo[0]), float(ci_hi[0])], 'beta': [float(ci_lo[1]), float(ci_hi[1])],
                         'k': [float(ci_lo[2]), float(ci_hi[2])], 'b': [float(ci_lo[3]), float(ci_hi[3])]},
        'bootstrap_n_success': int(len(boot_params)),
        'mae': float(mae), 'rmse': float(rmse),
        'median_N_80_predicted': float(n80_df['N_80_predicted'].median()),
        'median_actual_cycles': float(n80_df['actual_cycles_observed'].median()),
        'X_test': X_test, 'y_test': y_test, 'pred': pred, 'n80_df': n80_df, 'test_df': test_df
    }


# Implement
results = {}
for name, df in df_dictionary.items():
    result = fit_chemistry(df, name)
    results[name] = result

# ---------------------------------------------------------------
# Comparison plot: per-chemistry SOH fits + MAE/RMSE/N80 bar comparison
# ---------------------------------------------------------------
colors = {'Li-ion LFP': '#1E2761', 'Li-ion NCA': '#C0392B', 'Li-ion NMC': '#1E8449', 'Na-ion': "#FFE207",
          'Zn-ion': "#7F1E84"}
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

chems = list(results.keys())
x = np.arange(len(chems))
mae_vals = [results[c]['mae'] for c in chems]
rmse_vals = [results[c]['rmse'] for c in chems]
axes[0].bar(x - 0.18, mae_vals, width=0.36, label='MAE', color='#3A4590')
axes[0].bar(x + 0.18, rmse_vals, width=0.36, label='RMSE', color='#F9A825')
axes[0].axhline(0.0535, color='#3A4590', linestyle=':', alpha=0.7, label='Combined-fit MAE (0.0535)')
axes[0].axhline(0.0675, color='#F9A825', linestyle=':', alpha=0.7, label='Combined-fit RMSE (0.0675)')
axes[0].set_xticks(x);
axes[0].set_xticklabels(chems)
axes[0].set_title('Error by Chemistry vs Combined Baseline');
axes[0].legend(fontsize=8);
axes[0].grid(True, axis='y', linestyle=':', alpha=0.5)

n80_pred = [results[c]['median_N_80_predicted'] for c in chems]
n80_actual = [results[c]['median_actual_cycles'] for c in chems]
axes[1].bar(x - 0.18, n80_actual, width=0.36, label='Actual median cycles', color='gray')
axes[1].bar(x + 0.18, n80_pred, width=0.36, label='Predicted N_80 (median)', color='#C0392B')
axes[1].set_yscale('log')
axes[1].set_xticks(x);
axes[1].set_xticklabels(chems)
axes[1].set_title('N_80: Predicted vs Actual (per chemistry)');
axes[1].legend(fontsize=8);
axes[1].grid(True, axis='y', linestyle=':', alpha=0.5)

for chem, res in results.items():
    test_df = res['test_df']
    cell_ids = test_df['cell_id'].unique()
    if len(cell_ids) == 0:
        continue
    cid = cell_ids[0]
    sample_df = test_df[test_df['cell_id'] == cid].sort_values('cycle_number')
    Ea = res['params']['Ea']
    beta = res['params']['beta']
    k = res['params']['k']
    b = res['params']['b']
    pred = predict_soh(
        (sample_df['Ah_norm'].values, sample_df['mean_temperature_in_C'].values, sample_df['c_rate'].values),
        Ea, beta, k, b
    )
    color = colors.get(chem, '#000000')
    axes[2].plot(sample_df['cycle_number'], sample_df['SOH'], marker='o', linestyle='-', color=color, alpha=0.7,
                 linewidth=1, label=f'{chem} actual')
    axes[2].plot(sample_df['cycle_number'], pred, marker='x', linestyle='--', color=color, alpha=0.9, linewidth=1,
                 label=f'{chem} predicted')

axes[2].set_title('Single-sample predicted vs actual')
axes[2].set_xlabel('Cycle number')
axes[2].set_ylabel('SOH')
axes[2].legend(fontsize=6, loc='best')
axes[2].grid(True, linestyle=':', alpha=0.5)

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------
# Matched-Condition Plots: Compare all chemistries at same conditions
# ---------------------------------------------------------------
def plot_matched_conditions(results, df_dictionary, colors, temp_tolerance=3.0, rate_tolerance=0.5):
    """
    Create 3 specialized matched-condition plots:
    1. LFP vs NCA vs NMC vs Na-ion @ 25°C, 1.0 C-rate
    2. LFP vs NMC @ 25°C, 3.0 C-rate
    """

    # Define target conditions and chemistries for each plot
    plot_specs = [
        {
            'title_prefix': 'LFP vs NCA vs NMC vs Na-ion',
            'temp_target': 25.0,
            'rate_target': 1.0,
            'chemistries': ['Li-ion LFP', 'Li-ion NCA', 'Li-ion NMC', 'Na-ion']
        },
        {
            'title_prefix': 'LFP vs NMC',
            'temp_target': 25.0,
            'rate_target': 3.0,
            'chemistries': ['Li-ion LFP', 'Li-ion NMC']
        },
    ]

    ah_max = max(df['Ah_norm'].max() for df in df_dictionary.values())
    ah_grid = np.linspace(0, ah_max, 200)

    for spec in plot_specs:
        fig, ax = plt.subplots(figsize=(10, 6))

        temp_target = spec['temp_target']
        rate_target = spec['rate_target']
        selected_chems = spec['chemistries']

        any_chemistry = False

        for chem in selected_chems:
            if chem not in df_dictionary:
                continue

            df = df_dictionary[chem]
            temp_mask = np.abs(df['mean_temperature_in_C'] - temp_target) <= temp_tolerance
            rate_mask = np.abs(df['c_rate'] - rate_target) <= rate_tolerance

            if len(df[temp_mask & rate_mask]) == 0:
                continue

            any_chemistry = True

            if chem not in results:
                continue

            Ea = results[chem]['params']['Ea']
            beta = results[chem]['params']['beta']
            k = results[chem]['params']['k']
            b = results[chem]['params']['b']

            temp_array = np.full_like(ah_grid, temp_target)
            rate_array = np.full_like(ah_grid, rate_target)
            pred_curve = predict_soh((ah_grid, temp_array, rate_array), Ea, beta, k, b)

            ax.plot(
                ah_grid,
                pred_curve,
                color=colors[chem],
                linewidth=2.5,
                linestyle='-',
                label=f'{chem} (fit)',
                alpha=0.85,
            )

        if not any_chemistry:
            ax.text(0.5, 0.5, 'No chemistry matches this condition', ha='center', va='center', transform=ax.transAxes)

        ax.axhline(0.8, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='SOH threshold (0.8)')
        ax.set_xlabel('Normalized Throughput (Ah)', fontsize=12)
        ax.set_ylabel('State of Health (SOH)', fontsize=12)
        ax.set_title(f'{spec["title_prefix"]}\nT={temp_target:.1f}°C, C-rate={rate_target:.2f}',
                     fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, linestyle=':', alpha=0.4)
        ax.set_ylim([0.65, 1.1])
        ax.set_xlim([0, ah_max * 1.05])

        plt.tight_layout()
        plt.show()

# Generate matched-condition plots for all chemistries
plot_matched_conditions(results, df_dictionary, colors)

# ------------------------------------
# Degradation Rates
# ------------------------------------
def plot_degradation_rates_matched_bins(results, df_dictionary, colors, temp_tolerance=2.0, rate_tolerance=0.5):
    """Reproduce Fig. 4 / Fig. 5 style plots for each matched (T, C-rate) bin.

    Both panels draw ONE curve per chemistry per matched bin (matching the
    manuscript's Fig. 4 / Fig. 5, which each show a single "A" and single "B"
    line per panel):

    - Fig. 4 analogue: intrinsic rate vs SOH (linear y-axis).
      dSOH/dAh_eff|_SOH = -k * b * (Ah_eff(SOH))**(b-1), with
      Ah_eff(SOH) = ((1-SOH)/k)**(1/b). Depends only on chemistry-level k, b
      (Eq. 10).
    - Fig. 5 analogue: per-cycle rate vs SOH (log y-axis), using the matched
      bin's mean temperature, C-rate, and DoD across all tests in that
      chemistry/bin (Eq. 11):
      dSOH/dN|_SOH ≈ intrinsic_rate * phi_T * phi_C * DoD_mean
    """

    plot_specs = [
        ('LFP vs NCA vs NMC vs Na-ion', 25.0, 1.0, ['Na-ion', 'Li-ion LFP', 'Li-ion NCA', 'Li-ion NMC']),
        ('NMC vs LFP', 25.0, 3.0, ['Li-ion NMC', 'Li-ion LFP']),
    ]

    n_bins = len(plot_specs)
    fig_intr, axes_intr = plt.subplots(1, n_bins, figsize=(6 * n_bins, 5), squeeze=False)
    fig_app, axes_app = plt.subplots(1, n_bins, figsize=(6 * n_bins, 5), squeeze=False)
    axes_intr = axes_intr[0]
    axes_app = axes_app[0]

    for panel_idx, (title_prefix, temp_target, rate_target, chems) in enumerate(plot_specs):
        ax_intr = axes_intr[panel_idx]
        ax_app = axes_app[panel_idx]
        any_chemistry = False

        for chem in chems:
            if chem not in df_dictionary or chem not in results:
                continue

            df = df_dictionary[chem]
            res = results[chem]
            Ea = res['params']['Ea']
            beta = res['params']['beta']
            k = res['params']['k']
            b = res['params']['b']

            temp_mask = np.abs(df['mean_temperature_in_C'] - temp_target) <= temp_tolerance
            rate_mask = np.abs(df['c_rate'] - rate_target) <= rate_tolerance
            matched = df[temp_mask & rate_mask]
            if matched.empty:
                continue

            any_chemistry = True

            # --- Fig. 4 analogue: ONE intrinsic-rate curve per chemistry ---
            # (chemistry-level k, b only; anchored at SOH = 1 since Ah_eff = 0
            # there by construction, Eq. 6; lower bound set by how far this
            # chemistry's matched tests were actually observed to degrade)
            soh_min_intr = max(matched['SOH'].min(), SOH_EOL)
            soh_max_intr = 1.0
            if soh_min_intr >= soh_max_intr:
                soh_min_intr = SOH_EOL

            soh_plot_intr = np.linspace(soh_min_intr, soh_max_intr, 200)
            Ah_eff_intr = ((1.0 - soh_plot_intr) / k) ** (1.0 / b)
            intrinsic_rate = -k * b * (Ah_eff_intr ** (b - 1))
            ax_intr.plot(soh_plot_intr, intrinsic_rate, color=colors.get(chem, '#000000'),
                         linewidth=2.0, label=chem)

            # --- Fig. 5 analogue: ONE per-cycle-rate curve per chemistry ---
            # Uses the matched bin's mean T, C-rate, and DoD across all tests
            # in that chemistry/bin, mirroring how Fig. 5 in the manuscript
            # shows a single "A" and single "B" line per panel rather than one
            # line per individual test.
            T_mean = matched['mean_temperature_in_C'].mean()
            C_mean = matched['c_rate'].mean()
            DoD_mean = matched['depth_of_discharge'].mean()
            if np.isnan(DoD_mean) or DoD_mean <= 0:
                continue

            phi_t = calc_temp_stress(T_mean, Ea)
            phi_c = calc_crate_stress(C_mean, beta)

            app_rate = intrinsic_rate * phi_t * phi_c * DoD_mean
            # plotted as magnitude on a log axis, per Fig. 5 (-dSOH/dN)
            ax_app.plot(soh_plot_intr, -app_rate, color=colors.get(chem, '#000000'),
                        linewidth=2.0, label=chem)

        if not any_chemistry:
            ax_intr.text(0.5, 0.5, 'No chemistry matches this condition', ha='center', va='center',
                         transform=ax_intr.transAxes)
            ax_app.text(0.5, 0.5, 'No chemistry matches this condition', ha='center', va='center',
                        transform=ax_app.transAxes)
        else:
            ax_intr.set_xlabel('SOH (normalized)')
            ax_intr.set_ylabel('dSOH/dAh_eff')
            ax_intr.invert_xaxis()
            ax_intr.set_title(f'{title_prefix}\nT={temp_target:.0f}°C, C={rate_target:.1f}C')
            ax_intr.grid(True, linestyle=':', alpha=0.4)
            ax_intr.legend(fontsize=8)

            ax_app.set_xlabel('SOH (normalized)')
            ax_app.set_ylabel('-dSOH/dN, per cycle (log scale)')
            ax_app.set_yscale('log')
            ax_app.invert_xaxis()
            ax_app.set_title(f'{title_prefix}\nT={temp_target:.0f}°C, C={rate_target:.1f}C')
            ax_app.grid(True, which='both', linestyle=':', alpha=0.4)
            ax_app.legend(fontsize=8)

    fig_intr.suptitle('Intrinsic Degradation Rate versus SOH (matched bins)', fontweight='bold')
    fig_app.suptitle('Application-facing Degradation Rate (dSOH/dN, per cycle) versus SOH — matched bins, log scale',
                     fontweight='bold')
    fig_intr.tight_layout(rect=[0, 0.03, 1, 0.92])
    fig_app.tight_layout(rect=[0, 0.03, 1, 0.92])
    plt.show()

# Generate matched-bin degradation-rate plots (Fig. 4 / Fig. 5 analogues)
plot_degradation_rates_matched_bins(results, df_dictionary, colors)
