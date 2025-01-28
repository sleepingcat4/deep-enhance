import torch
import torchaudio
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance
import csv

device = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_audio(input_audio_path, output_audio_path, run_dir, solver="midpoint", nfe=64, tau=0.5):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)
    enhanced_audio, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau, run_dir=run_dir)
    torchaudio.save(output_audio_path, enhanced_audio.unsqueeze(0), new_sr)
    print(f"Enhanced audio saved to: {output_audio_path}")

def path_enhance(csv_path, output_folder, run_dir, log_file="error.log", solver="midpoint", nfe=64, tau=0.5):
    output_folder = Path(output_folder)
    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    with open(csv_path, "r") as file, open(log_file, "a") as log:
        reader = csv.reader(file)
        for row in reader:
            try:
                input_audio_path = Path(row[0])
                if not input_audio_path.exists():
                    raise FileNotFoundError(f"File not found: {input_audio_path}")

                output_audio_path = output_folder / f"{input_audio_path.stem}_enhanced.wav"
                enhance_audio(input_audio_path, output_audio_path, run_dir, solver, nfe, tau)
            except Exception as e:
                log.write(f"{input_audio_path}, {str(e)}\n")
                log.flush()
                print(f"Error processing {input_audio_path}: {e}")
