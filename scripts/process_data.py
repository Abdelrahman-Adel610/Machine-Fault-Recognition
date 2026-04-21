import os
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import warnings

# Import our custom modules
from src.utils.config import load_config
from src.utils.labels import get_label
from src.preprocessing.preprocess import preprocess_audio
from src.feature_extraction.extract import extract_mel_spectrograms

warnings.filterwarnings('ignore')

# Global config placeholder for the workers
worker_config = None

def init_worker(config):
    """This function runs once when each worker process starts."""
    global worker_config
    worker_config = config

def process_file(task):
    """Worker function for multiprocessing."""
    global worker_config
    
    file_path       = task["file_path"]
    label           = task["label"]
    machine         = task["machine"]
    label_name      = task["label_name"]
    base_name       = task["base_name"]
    machine_out_dir = task["machine_out_dir"]

    save_name = f"{machine}_{label_name}_{base_name}.npz"
    save_path = os.path.join(machine_out_dir, save_name)

    if os.path.exists(save_path):
        try:
            with np.load(save_path) as d:
                return (save_name, label, machine, d["features"].shape[0], save_path)
        except Exception:
            os.remove(save_path)

    try:
        # Pass config values to preprocessing
        chunks = preprocess_audio(
            file_path=file_path,
            target_sr=worker_config['audio']['target_sr'],
            chunk_duration=worker_config['audio']['chunk_duration'],
            step_duration=worker_config['audio']['step_duration'],
            trim_top_db=worker_config['audio']['trim_top_db']
        )

        # Pass config values to feature extraction
        features = extract_mel_spectrograms(
            chunks=chunks,
            sr=worker_config['audio']['target_sr'],
            n_fft=worker_config['features']['n_fft'],
            hop_length=worker_config['features']['hop_length'],
            n_mels=worker_config['features']['n_mels']
        )

        np.savez(save_path, features=features)
        return (save_name, label, machine, features.shape[0], save_path)

    except Exception as e:
        # It's helpful to see the actual error during debugging
        return f"Error processing {file_path}: {str(e)}"

def main():
    # Allow overriding config path from command line
    parser = argparse.ArgumentParser(description="Process Audio Data")
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Path to config file')
    args = parser.parse_args()

    # Load config in the main process
    main_config = load_config(args.config)
    
    input_dir = main_config['paths']['input_dir']
    output_dir = main_config['paths']['output_dir']
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")

    # Gather tasks
    tasks = []
    # Check if directory exists to avoid crashes
    if not os.path.exists(input_dir):
        print(f"❌ Error: Input directory '{input_dir}' not found.")
        return

    for machine in sorted(os.listdir(input_dir)):
        machine_path = os.path.join(input_dir, machine)
        if not os.path.isdir(machine_path): continue

        machine_out_dir = os.path.join(output_dir, machine)
        os.makedirs(machine_out_dir, exist_ok=True)

        # Your structure: Machine 1 -> machine_data -> Normal
        data_path = os.path.join(machine_path, "machine_data")
        if not os.path.exists(data_path): continue

        for label_name in ["Normal", "Abnormal"]:
            class_path = os.path.join(data_path, label_name)
            if not os.path.exists(class_path): continue

            label = get_label(machine, label_name)

            for file in sorted(os.listdir(class_path)):
                if file.endswith(".wav"):
                    tasks.append({
                        "file_path": os.path.join(class_path, file),
                        "label": label,
                        "machine": machine,
                        "label_name": label_name,
                        "base_name": os.path.splitext(file)[0],
                        "machine_out_dir": machine_out_dir,
                    })

    print(f"Total audio files to process: {len(tasks)}")

    # Execute Multiprocessing with Initializer
    all_results = []
    # Use initializer to pass config to workers
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(main_config,)) as pool:
        for res in tqdm(pool.imap_unordered(process_file, tasks), total=len(tasks)):
            if isinstance(res, tuple):
                all_results.append(res)
            else:
                print(res) # Print the error string if it failed

    # Save Metadata
    if all_results:
        df = pd.DataFrame(all_results, columns=["file", "label", "machine", "num_chunks", "file_path"])
        csv_path = os.path.join(output_dir, main_config['paths']['csv_name'])
        df.to_csv(csv_path, index=False)
        print(f"✅ Metadata saved to {csv_path}")
    else:
        print("❌ No files were processed successfully.")

if __name__ == "__main__":
    main()