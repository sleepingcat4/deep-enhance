import torch
import torchaudio
import pandas as pd
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance

device = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_audio(input_audio_path, output_audio_path, run_dir, solver="midpoint", nfe=64, tau=0.5):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)
    enhanced_audio, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau, run_dir=run_dir)
    torchaudio.save(output_audio_path, enhanced_audio.unsqueeze(0), new_sr)
    print(f"Enhanced audio saved to: {output_audio_path}")

def enhance_folder(input_folder, output_folder, run_dir, solver="midpoint", nfe=64, tau=0.5):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    input_files = list(input_folder.glob("*.mp3")) + list(input_folder.glob("*.wav"))
    if not input_files:
        print(f"No audio files found in {input_folder}.")
        return

    for input_file in input_files:
        output_audio = output_folder / f"{input_file.stem}_enhanced.wav"
        enhance_audio(input_file, output_audio, run_dir, solver, nfe, tau)

def decode_subfolder(csv_file, output_base_folder, run_dir, gpu_id=0, solver="midpoint", nfe=64, tau=0.5):
    df = pd.read_csv(csv_file)
    if 'folder_path' not in df.columns:
        print("CSV file must contain a 'folder_path' column.")
        return

    device = f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu"

    for folder_path in df['folder_path']:
        folder_path = Path(folder_path)
        if not folder_path.exists():
            print(f"Folder {folder_path} does not exist.")
            continue
        
        output_folder = Path(output_base_folder) / folder_path.name
        enhance_folder(folder_path, output_folder, run_dir, solver, nfe, tau)
