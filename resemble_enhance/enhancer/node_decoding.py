import torch
import torchaudio
from pathlib import Path
from resemble_enhance.enhancer.inference import enhance

def enhance_audio(input_audio_path, output_audio_path, run_dir, device, solver="midpoint", nfe=64, tau=0.5):
    dwav, sr = torchaudio.load(input_audio_path)
    dwav = dwav.mean(dim=0)
    enhanced_audio, new_sr = enhance(dwav, sr, device, nfe=nfe, solver=solver, lambd=0.1, tau=tau, run_dir=run_dir)
    torchaudio.save(output_audio_path, enhanced_audio.unsqueeze(0), new_sr)

def node_inference(input_folder, output_folder, run_dir, solver="midpoint", nfe=64, tau=0.5):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)

    if not output_folder.exists():
        output_folder.mkdir(parents=True, exist_ok=True)

    input_files = list(input_folder.glob("*.mp3")) + list(input_folder.glob("*.wav"))
    if not input_files:
        return

    available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else ["cpu"]
    processed_files = set()

    def process_file(input_file, device):
        if input_file in processed_files:
            return
        processed_files.add(input_file)

        output_audio = output_folder / f"{input_file.stem}_enhanced.wav"
        enhance_audio(input_file, output_audio, run_dir, device, solver, nfe, tau)

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=len(available_devices)) as executor:
        for idx, input_file in enumerate(input_files):
            device = available_devices[idx % len(available_devices)]
            executor.submit(process_file, input_file, device)
