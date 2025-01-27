import torch
import torchaudio
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance
from accelerate import Accelerator

device = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_audio(input_audio_path, output_audio_path, run_dir, device, solver="midpoint", nfe=64, tau=0.5):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)
    enhanced_audio, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau, run_dir=run_dir)
    torchaudio.save(output_audio_path, enhanced_audio.unsqueeze(0), new_sr)
    print(f"Enhanced audio saved to: {output_audio_path}")

def node_inference(input_folder, output_folder, run_dir, solver="midpoint", nfe=64, tau=0.5):
    accelerator = Accelerator()
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    input_files = set(input_folder.glob("*.mp3")) | set(input_folder.glob("*.wav"))
    if not input_files:
        print("No audio files found in the input folder.")
        return

    input_files = list(input_files)
    input_files = accelerator.prepare(input_files)

    num_processes = accelerator.num_processes
    process_index = accelerator.process_index

    chunk_size = len(input_files) // num_processes
    start_idx = process_index * chunk_size
    end_idx = (process_index + 1) * chunk_size if process_index != num_processes - 1 else len(input_files)

    files_to_process = input_files[start_idx:end_idx]
    
    seen_files = set()

    for input_file in files_to_process:
        if input_file.stem in seen_files:
            print(f"Duplicate file detected: {input_file.stem}, skipping.")
            continue
        
        seen_files.add(input_file.stem)
        
        current_device = accelerator.device
        output_audio = output_folder / f"{input_file.stem}_enhanced.wav"
        enhance_audio(input_file, output_audio, run_dir, current_device, solver, nfe, tau)
