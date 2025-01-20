import torch
import torchaudio
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance

device = "cuda" if torch.cuda.is_available() else "cpu"

def enhance_audio(input_audio_path, output_audio_path, solver="midpoint", nfe=64, tau=0.5):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)

    enhanced_audio, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau)

    torchaudio.save(output_audio_path, enhanced_audio.unsqueeze(0), new_sr)
    print(f"Enhanced audio saved to: {output_audio_path}")

def enhance_folder(input_folder, output_folder, solver="midpoint", nfe=64, tau=0.5):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    input_files = list(input_folder.glob("*.mp3")) + list(input_folder.glob("*.wav"))
    if not input_files:
        print("No audio files found in the input folder.")
        return

    for input_file in input_files:
        converted_file = output_folder / f"{input_file.stem}_converted.wav"
        if input_file.suffix.lower() == ".mp3":
            wav_audio, sr = torchaudio.load(input_file)
            torchaudio.save(converted_file, wav_audio, sr)
            print(f"Converted {input_file} to {converted_file}")
        else:
            converted_file = input_file

        output_audio = output_folder / f"{converted_file.stem}_enhanced.wav"
        enhance_audio(converted_file, output_audio, solver, nfe, tau)