import glob
import os
import warnings
import xml.etree.ElementTree as ET
import neurokit2 as nk
import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ==========================================
# GLOBAL SETTINGS
# ==========================================
TARGET_LEAD = 2
SAMPLING_RATE = 500
WINDOW_SEC = 60
STEP_SEC = 10

# Paths should be relative to the repository root
INPUT_CSV_LIST = ["data/id.csv"]
BASE_DATA_DIR = "data/ecg_raw"
HRV_INDICES_CSV = "HRV_indices.csv"
SUB_FOLDERS = [
    "control",
    "MASLD"
]


def load_target_hrv_features(csv_path):
    """Load target HRV indices from the 2nd column and add 'HRV_' prefix."""
    try:
        df = pd.read_csv(csv_path, header=None)
        raw_feats = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        return raw_feats
    except Exception as e:
        print(f"Error loading HRV indices from {csv_path}: {e}")
        return []


def load_and_process_ecg_lead(file_path, target_lead=2):
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()

        full_disclosure_node = root.find(".//FullDisclosureData")
        if full_disclosure_node is None:
            return None

        full_data_str = full_disclosure_node.text.strip()
        full_data = np.array(
            [
                int(x)
                for x in full_data_str.replace("\n", "").split(",")
                if x.strip()
            ]
        )

        chunk_size = 500
        num_leads = 3
        stride = chunk_size * num_leads
        num_blocks = len(full_data) // stride

        reshaped_data = full_data[: num_blocks * stride].reshape(
            num_blocks, stride
        )

        lead_index = target_lead - 1
        start_col = lead_index * chunk_size
        end_col = (lead_index + 1) * chunk_size

        return reshaped_data[:, start_col:end_col].flatten()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None


def process_sliding_window(
    clean_signal,
    rpeaks_indices,
    target_features,
    sampling_rate=500,
    window_sec=60,
    step_sec=10,
):
    signal_len = len(clean_signal)
    window_samples = window_sec * sampling_rate
    step_samples = step_sec * sampling_rate
    start_offset_samples = 0

    results_list = []
    remaining_len = signal_len - start_offset_samples

    if remaining_len < window_samples:
        return pd.DataFrame()

    num_windows = (remaining_len - window_samples) // step_samples + 1

    for i in range(int(num_windows)):
        start_idx = int(start_offset_samples + (i * step_samples))
        end_idx = start_idx + window_samples

        window_rpeaks = rpeaks_indices[
            (rpeaks_indices >= start_idx) & (rpeaks_indices < end_idx)
        ]
        if len(window_rpeaks) < 10:
            continue

        local_peaks = window_rpeaks - start_idx

        try:
            hrv_results = nk.hrv(
                peaks=local_peaks, sampling_rate=sampling_rate, show=False
            )

            filtered_results = {}
            for feature in target_features:
                if feature in hrv_results.columns:
                    filtered_results[feature] = hrv_results[feature].values[0]
                else:
                    filtered_results[feature] = np.nan

            
            df_win = pd.DataFrame([filtered_results])
            df_win["Window_Index"] = i
            df_win["Window_Start_Time_s"] = start_idx / sampling_rate
            df_win["Window_End_Time_s"] = end_idx / sampling_rate
            results_list.append(df_win)
        except:
            continue

    return (
        pd.concat(results_list, axis=0).reset_index(drop=True)
        if results_list
        else pd.DataFrame()
    )


if __name__ == "__main__":
    print("Building file index...")
    file_map = {}
    for sub in SUB_FOLDERS:
        folder_path = os.path.join(BASE_DATA_DIR, sub)
        xml_files = glob.glob(os.path.join(folder_path, "*.xml"))
        for f_path in xml_files:
            f_name = os.path.basename(f_path)
            file_map[f_name] = f_path
    print(f"Index built. Found {len(file_map)} XML files.")

    # Load the target feature list for alignment
    target_hrv_features = load_target_hrv_features(HRV_INDICES_CSV)
    if not target_hrv_features:
        print("No valid HRV indices list found. Exiting.")
        exit()

    for excel_file in INPUT_CSV_LIST:
        print(
            f"\nProcessing: {excel_file} (Target Lead: Lead {TARGET_LEAD})"
        )

        if not os.path.exists(excel_file):
            print(f"Warning: File not found {excel_file}")
            continue

        df_files = pd.read_csv(excel_file)
        target_files = df_files.iloc[:, 0].astype(str).tolist()
        all_windows_data = []

        for f_name in tqdm(target_files, desc=f"Lead {TARGET_LEAD}"):
            search_name = (
                f_name if f_name.endswith(".xml") else f_name + ".xml"
            )

            if search_name not in file_map:
                continue

            file_path = file_map[search_name]
            ecg_data = load_and_process_ecg_lead(
                file_path, target_lead=TARGET_LEAD
            )
            if ecg_data is None:
                continue

            try:
                cleaned_data = nk.ecg_clean(
                    ecg_data, sampling_rate=SAMPLING_RATE, method="vg"
                )
                _, rpeaks_info = nk.ecg_peaks(
                    cleaned_data, sampling_rate=SAMPLING_RATE, method="vg"
                )
                rpeaks = rpeaks_info["ECG_R_Peaks"]

                df_file_results = process_sliding_window(
                    cleaned_data,
                    rpeaks,
                    target_features=target_hrv_features,
                    sampling_rate=SAMPLING_RATE,
                    window_sec=WINDOW_SEC,
                    step_sec=STEP_SEC,
                )

                if not df_file_results.empty:
                    df_file_results.insert(0, "Filename", search_name)
                    parent_folder = os.path.basename(
                        os.path.dirname(file_path)
                    )
                    df_file_results.insert(1, "Group", parent_folder)
                    df_file_results.insert(2, "Lead_Used", TARGET_LEAD)
                    all_windows_data.append(df_file_results)
            except:
                continue

        if all_windows_data:
            final_df = pd.concat(all_windows_data, axis=0, ignore_index=True)
            output_name = os.path.basename(excel_file).replace(
                ".csv", f"_Lead{TARGET_LEAD}_SlidingWindow_Results.csv"
            )
            output_path = os.path.join(os.path.dirname(excel_file), output_name)
            
            # Align final column ordering with target features present in the dataframe
            main_cols = ["Filename", "Group", "Lead_Used"]
            meta_cols = ["Window_Index", "Window_Start_Time_s", "Window_End_Time_s"]
            feat_cols = [c for c in target_hrv_features if c in final_df.columns]
            
            final_df = final_df[main_cols + feat_cols + meta_cols]
            final_df.to_csv(output_path, index=False)
            print(f"Results saved to: {output_path}")
        else:
            print("No valid data generated for this list.")

    print("\nAll tasks completed.")