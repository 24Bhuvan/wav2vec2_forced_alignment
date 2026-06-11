from pathlib import Path
import subprocess

input_folder = Path(r"data\audio")
output_folder = Path(r"data\wav")

output_folder.mkdir(parents=True, exist_ok=True)

mp3_files = list(input_folder.glob("*.mp3"))

for i, mp3_file in enumerate(mp3_files, start=1):
    wav_file = output_folder / f"{mp3_file.stem}.wav"

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp3_file),
        "-ac", "1",      # mono
        "-ar", "16000",  # 16 kHz
        str(wav_file)
    ]

    subprocess.run(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    print(f"[{i}/{len(mp3_files)}] Done")

print("Finished")