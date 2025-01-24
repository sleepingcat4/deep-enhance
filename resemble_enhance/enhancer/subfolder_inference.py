import torch
import torchaudio
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance

device = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_audio(input_audio_path, output_audio_path, run_dir, solver="midpoint", nfe=64, tau=0.5):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)
    enhanced_audio, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau, run_dir=run_dir)
    torchaudio.save(output_audio_path, enhanced_audio.unsqueeze(0), new_sr)
    print(f"Enhanced audio saved to: {output_audio_path}")

def tree_inference(input_folder, output_folder, run_dir, solver="midpoint", nfe=64, tau=0.5):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    for input_file in input_folder.rglob("*.mp3") + input_folder.rglob("*.wav"):
        relative_path = input_file.relative_to(input_folder)
        output_file_path = output_folder / relative_path.parent / f"{relative_path.stem}_enhanced.wav"
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        enhance_audio(input_file, output_file_path, run_dir, solver, nfe, tau)
