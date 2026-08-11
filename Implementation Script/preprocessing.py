import pickle
import pandas as pd
import numpy as np
import os
from pathlib import Path

def process_pkl_file(file_path, nominal_capacity_cache=None):
    """
    Process a single PKL file and return processed data.
    
    Parameters:
    - file_path: path to the PKL file
    - nominal_capacity_cache: dict to store/retrieve nominal capacities by filename
    
    Returns:
    - DataFrame with processed cycle data
    - nominal_capacity used
    - metadata dictionary
    """
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    # ===== EXTRACT METADATA =====
    metadata = {}
    if isinstance(data, dict):
        metadata = {
            'cell_id': data.get('cell_id', 'unknown'),
            'form_factor': data.get('form_factor', 'unknown'),
            'anode_material': data.get('anode_material', 'unknown'),
            'cathode_material': data.get('cathode_material', 'unknown'),
            'electrolyte_material': data.get('electrolyte_material', 'unknown'),
            'nominal_capacity_in_Ah': data.get('nominal_capacity_in_Ah', np.nan),
            'depth_of_charge': data.get('depth_of_charge', np.nan),
            'depth_of_discharge': data.get('depth_of_discharge', np.nan),
            'already_spent_cycles': data.get('already_spent_cycles', 0),
            'max_voltage_limit_in_V': data.get('max_voltage_limit_in_V', np.nan),
            'min_voltage_limit_in_V': data.get('min_voltage_limit_in_V', np.nan),
            'max_current_limit_in_A': data.get('max_current_limit_in_A', np.nan),
            'min_current_limit_in_A': data.get('min_current_limit_in_A', np.nan),
            'reference': data.get('reference', 'unknown'),
            'description': data.get('description', 'unknown'),
        }
    
    # First pass: Extract all cycle data to find the 10th cycle's discharge capacity
    cycles_data = []
    
    if isinstance(data, dict) and 'cycle_data' in data:
        cycle_data = data['cycle_data']
        
        for cycle_idx, cycle in enumerate(cycle_data):
            if isinstance(cycle, dict):
                cycle_num = cycle.get('cycle_number', cycle_idx)
                discharge_cap = cycle.get('discharge_capacity_in_Ah', [])
                
                # Get max discharge capacity for this cycle
                if len(discharge_cap) > 0:
                    valid_caps = [c for c in discharge_cap if c is not None and not np.isnan(c)]
                    max_discharge_capacity = max(valid_caps) if valid_caps else np.nan
                else:
                    max_discharge_capacity = np.nan
                
                cycles_data.append({
                    'cycle_number': cycle_num,
                    'max_discharge_capacity': max_discharge_capacity,
                    'cycle_data': cycle
                })
        
        # Get nominal capacity
        nominal_capacity = metadata.get('nominal_capacity_in_Ah', np.nan)
        
        # Process all cycles
        records = []
        max_capacities_by_cycle = {}  # For SOH calculation
        
        for cycle_info in cycles_data:
            cycle = cycle_info['cycle_data']
            cycle_num = cycle_info['cycle_number']
            
            # Extract data arrays
            charge_cap = cycle.get('charge_capacity_in_Ah', [])
            discharge_cap = cycle.get('discharge_capacity_in_Ah', [])
            current = cycle.get('current_in_A', [])
            temperature = cycle.get('temperature_in_C', [])
            voltage = cycle.get('voltage_in_V', [])
            time_s = cycle.get('time_in_s', [])
            resistance = cycle.get('internal_resistance_in_ohm', [])
            
            # 1. Extract max capacity achieved in this cycle
            all_capacities = []
            if len(charge_cap) > 0:
                valid_charge = [c for c in charge_cap if c is not None and not np.isnan(c)]
                all_capacities.extend(valid_charge)
            if len(discharge_cap) > 0:
                valid_discharge = [c for c in discharge_cap if c is not None and not np.isnan(c)]
                all_capacities.extend(valid_discharge)
            
            max_capacity = max(all_capacities) if all_capacities else np.nan
            
            # Store for SOH calculation
            max_capacities_by_cycle[cycle_num] = max_capacity
            
            # 2. Take the mean temperature across the whole cycle
            if temperature and len(temperature) > 0:
                valid_temps = [t for t in temperature if t is not None and not np.isnan(t)]
                mean_temperature = np.mean(valid_temps) if valid_temps else np.nan
            else:
                mean_temperature = np.nan
            
            # 3. Calculate C-rate using nominal capacity
            if len(current) > 0 and nominal_capacity and nominal_capacity > 0:
                valid_currents = [c for c in current if c is not None and not np.isnan(c)]
                max_abs_current = max(abs(np.array(valid_currents))) if valid_currents else np.nan
                c_rate = max_abs_current / nominal_capacity if nominal_capacity > 0 else np.nan
            else:
                max_abs_current = np.nan
                c_rate = np.nan
            
            record = {
                'cycle_number': cycle_num,
                'max_capacity_in_Ah': max_capacity,
                'mean_temperature_in_C': mean_temperature,
                'max_abs_current_in_A': max_abs_current,
                'c_rate': c_rate,
                'nominal_capacity_Ah': nominal_capacity,
                'depth_of_discharge': metadata.get('depth_of_discharge', np.nan),
                'depth_of_charge': metadata.get('depth_of_charge', np.nan),
                'anode_material': metadata.get('anode_material', 'unknown'),
                'cathode_material': metadata.get('cathode_material', 'unknown'),
                'electrolyte_material': metadata.get('electrolyte_material', 'unknown'),
                'cell_id': metadata.get('cell_id', 'unknown'),
            }
            records.append(record)
        
        # ===== CALCULATE SOH using initial measured capacity =====
        if cycles_data:
            first_cycle_num = min(max_capacities_by_cycle.keys()) if max_capacities_by_cycle else None
            if first_cycle_num is not None:
                initial_capacity = max_capacities_by_cycle.get(first_cycle_num, np.nan)
            else:
                initial_capacity = np.nan
        else:
            initial_capacity = np.nan
        
        # Add SOH to each record
        for record in records:
            max_cap = record['max_capacity_in_Ah']
            
            if initial_capacity and initial_capacity > 0 and max_cap and max_cap > 0:
                soh = max_cap / initial_capacity
                soh = max(0.7, min(1.0, soh))  # Clip to realistic range
            else:
                soh = np.nan
            
            record['SOH'] = soh
        
        return pd.DataFrame(records), nominal_capacity, metadata
    
    else:
        print(f"  Warning: {file_path} has unexpected format")
        return None, None, metadata


def process_all_pkl_files(input_folder, output_folder=None):
    """
    Process all PKL files in a folder.
    
    Parameters:
    - input_folder: path to folder containing PKL files
    - output_folder: path to output folder (default: same as input_folder)
    """
    # Set output folder
    if output_folder is None:
        output_folder = input_folder
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Find all PKL files
    pkl_files = list(Path(input_folder).glob('*.pkl'))
    
    if not pkl_files:
        print(f"No PKL files found in {input_folder}")
        return
    
    print(f"Found {len(pkl_files)} PKL files to process")
    print("-" * 60)
    
    # Process each file
    results_summary = []
    all_data = []
    
    for pkl_file in pkl_files:
        print(f"Processing: {pkl_file.name}")
        
        try:
            # Process the file
            df, nominal_cap, metadata = process_pkl_file(pkl_file)
            
            if df is not None and not df.empty:
                # Add source file
                df['source_file'] = pkl_file.stem
                
                # Generate output filename
                output_filename = pkl_file.stem + '_processed.csv'
                output_path = Path(output_folder) / output_filename
                
                # Save to CSV
                df.to_csv(output_path, index=False)
                
                # Add to combined data
                all_data.append(df)
                
                # Store summary info
                results_summary.append({
                    'filename': pkl_file.name,
                    'output_file': output_filename,
                    'num_cycles': len(df),
                    'nominal_capacity_Ah': nominal_cap,
                    'depth_of_discharge': metadata.get('depth_of_discharge', np.nan),
                    'anode': metadata.get('anode_material', 'unknown'),
                    'cathode': metadata.get('cathode_material', 'unknown'),
                    'electrolyte': metadata.get('electrolyte_material', 'unknown'),
                    'max_capacity_Ah': df['max_capacity_in_Ah'].max(),
                    'mean_c_rate': df['c_rate'].mean(),
                    'mean_SOH': df['SOH'].mean() if 'SOH' in df.columns else np.nan,
                    'min_SOH': df['SOH'].min() if 'SOH' in df.columns else np.nan,
                    'status': 'Success'
                })
                
                print(f"  ✓ Saved {len(df)} cycles to {output_filename}")
                print(f"  ✓ Nominal capacity: {nominal_cap:.6f} Ah")
                print(f"  ✓ Depth of discharge: {metadata.get('depth_of_discharge', 'N/A')}")
                print(f"  ✓ Max capacity: {df['max_capacity_in_Ah'].max():.6f} Ah")
                print(f"  ✓ Mean C-rate: {df['c_rate'].mean():.4f}")
                print(f"  ✓ SOH range: {df['SOH'].min():.4f} - {df['SOH'].max():.4f}")
            else:
                results_summary.append({
                    'filename': pkl_file.name,
                    'status': 'Failed - Invalid format',
                    'num_cycles': 0
                })
                print(f"  ✗ Failed to process - invalid format")
        
        except Exception as e:
            print(f"  ✗ Error processing {pkl_file.name}: {str(e)}")
            results_summary.append({
                'filename': pkl_file.name,
                'status': f'Failed - {str(e)[:50]}',
                'num_cycles': 0
            })
        
        print("-" * 60)
    
    # Save summary to CSV
    summary_df = pd.DataFrame(results_summary)
    summary_path = Path(output_folder) / 'processing_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    
    # Save combined data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = Path(output_folder) / 'all_data_combined.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✅ Combined dataset saved: {combined_path} ({len(combined_df)} rows)")
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(summary_df[summary_df['status'] == 'Success'])} files")
    print(f"Failed: {len(summary_df[summary_df['status'] != 'Success'])} files")
    print(f"\nSummary saved to: {summary_path}")
    
    return summary_df, combined_df if all_data else None


# If you want to also keep the original data lists (voltage, current, time)
def process_all_pkl_files_with_lists(input_folder, output_folder=None):
    """
    Process all PKL files and keep original measurement lists.
    """
    if output_folder is None:
        output_folder = input_folder
    
    os.makedirs(output_folder, exist_ok=True)
    pkl_files = list(Path(input_folder).glob('*.pkl'))
    
    print(f"Found {len(pkl_files)} PKL files to process")
    print("-" * 60)
    
    results_summary = []
    all_data = []
    
    for pkl_file in pkl_files:
        print(f"Processing: {pkl_file.name}")
        
        try:
            with open(pkl_file, 'rb') as f:
                data = pickle.load(f)
            
            # Extract metadata
            metadata = {}
            if isinstance(data, dict):
                metadata = {
                    'cell_id': data.get('cell_id', 'unknown'),
                    'form_factor': data.get('form_factor', 'unknown'),
                    'anode_material': data.get('anode_material', 'unknown'),
                    'cathode_material': data.get('cathode_material', 'unknown'),
                    'electrolyte_material': data.get('electrolyte_material', 'unknown'),
                    'nominal_capacity_in_Ah': data.get('nominal_capacity_in_Ah', np.nan),
                    'depth_of_charge': data.get('depth_of_charge', np.nan),
                    'depth_of_discharge': data.get('depth_of_discharge', np.nan),
                    'already_spent_cycles': data.get('already_spent_cycles', 0),
                }
            
            if isinstance(data, dict) and 'cycle_data' in data:
                # First pass to get nominal capacity
                cycles_data = []
                for cycle_idx, cycle in enumerate(data['cycle_data']):
                    if isinstance(cycle, dict):
                        cycle_num = cycle.get('cycle_number', cycle_idx)
                        discharge_cap = cycle.get('discharge_capacity_in_Ah', [])
                        valid_caps = [c for c in discharge_cap if c is not None and not np.isnan(c)]
                        max_discharge = max(valid_caps) if valid_caps else np.nan
                        cycles_data.append({
                            'cycle_number': cycle_num,
                            'max_discharge_capacity': max_discharge,
                            'cycle_data': cycle
                        })
                
                # Determine nominal capacity
                nominal_capacity = metadata.get('nominal_capacity_in_Ah', np.nan)
                if nominal_capacity is None or np.isnan(nominal_capacity):
                    for cycle_info in cycles_data:
                        if cycle_info['cycle_number'] == 10:
                            nominal_capacity = cycle_info['max_discharge_capacity']
                            break
                    if (nominal_capacity is None or np.isnan(nominal_capacity)) and len(cycles_data) >= 10:
                        nominal_capacity = cycles_data[9]['max_discharge_capacity']
                    if nominal_capacity is None or np.isnan(nominal_capacity):
                        all_caps = [c['max_discharge_capacity'] for c in cycles_data 
                                   if not np.isnan(c['max_discharge_capacity'])]
                        nominal_capacity = max(all_caps) if all_caps else 1.0
                
                # Second pass: process all cycles with full data
                records = []
                max_capacities_by_cycle = {}  # For SOH calculation
                
                for cycle_info in cycles_data:
                    cycle = cycle_info['cycle_data']
                    cycle_num = cycle_info['cycle_number']
                    
                    charge_cap = cycle.get('charge_capacity_in_Ah', [])
                    discharge_cap = cycle.get('discharge_capacity_in_Ah', [])
                    current = cycle.get('current_in_A', [])
                    temperature = cycle.get('temperature_in_C', [])
                    voltage = cycle.get('voltage_in_V', [])
                    time_s = cycle.get('time_in_s', [])
                    resistance = cycle.get('internal_resistance_in_ohm', [])
                    
                    # Calculate features
                    all_capacities = []
                    if len(charge_cap) > 0:
                        all_capacities.extend([c for c in charge_cap if c is not None and not np.isnan(c)])
                    if len(discharge_cap) > 0:
                        all_capacities.extend([c for c in discharge_cap if c is not None and not np.isnan(c)])
                    max_capacity = max(all_capacities) if all_capacities else np.nan
                    
                    # Store for SOH
                    max_capacities_by_cycle[cycle_num] = max_capacity
                    
                    if temperature and len(temperature) > 0:
                        valid_temps = [t for t in temperature if t is not None and not np.isnan(t)]
                        mean_temperature = np.mean(valid_temps) if valid_temps else np.nan
                    else:
                        mean_temperature = np.nan
                    
                    if len(current) > 0 and nominal_capacity and nominal_capacity > 0:
                        valid_currents = [c for c in current if c is not None and not np.isnan(c)]
                        max_abs_current = max(abs(np.array(valid_currents))) if valid_currents else np.nan
                        c_rate = max_abs_current / nominal_capacity if nominal_capacity > 0 else np.nan
                    else:
                        max_abs_current = np.nan
                        c_rate = np.nan
                    
                    record = {
                        'cycle_number': cycle_num,
                        'max_capacity_in_Ah': max_capacity,
                        'mean_temperature_in_C': mean_temperature,
                        'max_abs_current_in_A': max_abs_current,
                        'c_rate': c_rate,
                        'nominal_capacity_Ah': nominal_capacity,
                        'depth_of_discharge': metadata.get('depth_of_discharge', np.nan),
                        'depth_of_charge': metadata.get('depth_of_charge', np.nan),
                        'anode_material': metadata.get('anode_material', 'unknown'),
                        'cathode_material': metadata.get('cathode_material', 'unknown'),
                        'electrolyte_material': metadata.get('electrolyte_material', 'unknown'),
                        'cell_id': metadata.get('cell_id', 'unknown'),
                        'current_in_A': current,
                        'voltage_in_V': voltage,
                        'time_in_s': time_s,
                        'charge_capacity_in_Ah': charge_cap,
                        'discharge_capacity_in_Ah': discharge_cap,
                        'temperature_in_C': temperature,
                        'internal_resistance_in_ohm': resistance,
                    }
                    records.append(record)
                
                # ===== Calculate SOH using initial measured capacity =====
                if cycles_data:
                    first_cycle_num = min(max_capacities_by_cycle.keys()) if max_capacities_by_cycle else None
                    if first_cycle_num is not None:
                        initial_capacity = max_capacities_by_cycle.get(first_cycle_num, np.nan)
                    else:
                        initial_capacity = np.nan
                else:
                    initial_capacity = np.nan
                
                # Add SOH to each record
                for record in records:
                    max_cap = record['max_capacity_in_Ah']
                    
                    if initial_capacity and initial_capacity > 0 and max_cap and max_cap > 0:
                        soh = max_cap / initial_capacity
                        soh = max(0.7, min(1.0, soh))  # Clip to realistic range
                    else:
                        soh = np.nan
                    
                    record['SOH'] = soh
                
                df = pd.DataFrame(records)
                df['source_file'] = pkl_file.stem
                
                output_filename = pkl_file.stem + '_processed_full.csv'
                output_path = Path(output_folder) / output_filename
                df.to_csv(output_path, index=False)
                
                all_data.append(df)
                
                results_summary.append({
                    'filename': pkl_file.name,
                    'output_file': output_filename,
                    'num_cycles': len(df),
                    'nominal_capacity_Ah': nominal_capacity,
                    'depth_of_discharge': metadata.get('depth_of_discharge', np.nan),
                    'mean_SOH': df['SOH'].mean() if 'SOH' in df.columns else np.nan,
                    'min_SOH': df['SOH'].min() if 'SOH' in df.columns else np.nan,
                    'status': 'Success'
                })
                print(f"  ✓ Saved {len(df)} cycles to {output_filename}")
                print(f"  ✓ Nominal capacity: {nominal_capacity:.6f} Ah")
                print(f"  ✓ Depth of discharge: {metadata.get('depth_of_discharge', 'N/A')}")
                print(f"  ✓ SOH range: {df['SOH'].min():.4f} - {df['SOH'].max():.4f}")
            else:
                results_summary.append({'filename': pkl_file.name, 'status': 'Failed - Invalid format'})
                print(f"  ✗ Failed to process - invalid format")
        
        except Exception as e:
            print(f"  ✗ Error: {str(e)}")
            results_summary.append({'filename': pkl_file.name, 'status': f'Failed - {str(e)[:50]}'})
        
        print("-" * 60)
    
    summary_df = pd.DataFrame(results_summary)
    summary_df.to_csv(Path(output_folder) / 'processing_summary.csv', index=False)
    
    # Save combined data
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = Path(output_folder) / 'all_data_combined_full.csv'
        combined_df.to_csv(combined_path, index=False)
        print(f"\n✅ Combined dataset saved: {combined_path} ({len(combined_df)} rows)")
    
    return summary_df, combined_df if all_data else None

# ============ USAGE ============

# Set your input folder path here
input_folder = r'D:\Jason\UTSG\Summer Research\Dataset\LI-ion'  # <-- CHANGE THIS
output_folder = r'D:\Jason\UTSG\Summer Research\Dataset\LI-ion\Processed Li Ion Dataset'     # <-- CHANGE THIS

# Run the processing (basic version)
summary, combined = process_all_pkl_files(input_folder, output_folder)

# If you want to keep original data lists (voltage, current, time), use:
# summary, combined = process_all_pkl_files_with_lists(input_folder, output_folder)

# Display summary
print("\nProcessing Summary:")
if summary is not None:
    print(summary[['filename', 'nominal_capacity_Ah', 'depth_of_discharge', 'num_cycles', 'status']].head(10))
