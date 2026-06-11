import os
import json
import re
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio

try:
    import soundfile as sf
except ImportError:
    sf = None


# =====================================================
# CONFIG
# =====================================================

AUDIO_DIR = Path("data/wav")
TRANSCRIPT_DIR = Path("data/transcripts")
OUTPUT_DIR = Path("wav2vec2_outputs")

OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"\nUsing device: {device}")


# =====================================================
# LOAD MODEL
# =====================================================

bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H

model = bundle.get_model().to(device)
model.eval()

labels = bundle.get_labels()

token_to_idx = {
    c.lower(): i
    for i, c in enumerate(labels)
}


# =====================================================
# TEXT NORMALIZATION
# =====================================================

def normalize_text(text: str):

    text = text.lower()

    text = re.sub(
        r"[^a-z\s]",
        "",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# =====================================================
# LOAD AUDIO
# =====================================================

def _load_with_soundfile(audio_path):

    data, sr = sf.read(
        str(audio_path),
        dtype="float32"
    )

    if data.ndim == 1:
        waveform = torch.from_numpy(
            data[np.newaxis, :]
        )
    else:
        waveform = torch.from_numpy(
            data.T
        )

    return waveform, sr


def _load_with_wave(audio_path):

    with wave.open(str(audio_path), "rb") as wf:
        sr = wf.getframerate()
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    if sampwidth == 1:
        data = np.frombuffer(frames, dtype=np.uint8)
        data = (data.astype(np.float32) - 128.0) / 128.0
    elif sampwidth == 2:
        data = np.frombuffer(frames, dtype=np.int16)
        data = data.astype(np.float32) / 32768.0
    elif sampwidth == 3:
        raw = np.frombuffer(frames, dtype=np.uint8)
        raw = raw.reshape(-1, 3)
        data = (
            raw[:, 0].astype(np.int32)
            | (raw[:, 1].astype(np.int32) << 8)
            | (raw[:, 2].astype(np.int32) << 16)
        )
        data = np.where(
            data & 0x800000,
            data - 0x1000000,
            data,
        ).astype(np.float32) / 8388608.0
    elif sampwidth == 4:
        data = np.frombuffer(frames, dtype=np.int32)
        data = data.astype(np.float32) / 2147483648.0
    else:
        raise ValueError(
            f"Unsupported WAV sample width: {sampwidth}"
        )

    if channels > 1:
        data = data.reshape(-1, channels).T
        waveform = torch.from_numpy(data)
    else:
        waveform = torch.from_numpy(data[np.newaxis, :])

    return waveform, sr


def load_audio(audio_path):

    try:
        waveform, sr = torchaudio.load(audio_path)
        loader = "torchaudio"

    except Exception:
        if sf is not None:
            try:
                waveform, sr = _load_with_soundfile(audio_path)
                loader = "soundfile"
            except Exception:
                waveform, sr = _load_with_wave(audio_path)
                loader = "wave"
        else:
            waveform, sr = _load_with_wave(audio_path)
            loader = "wave"

    if waveform.shape[0] > 1:
        waveform = waveform.mean(
            dim=0,
            keepdim=True
        )

    if sr != bundle.sample_rate:

        waveform = torchaudio.functional.resample(
            waveform,
            sr,
            bundle.sample_rate
        )

    print(
        f"Loaded {audio_path} via {loader} "
        f"(sr={sr}, shape={tuple(waveform.shape)})"
    )

    return waveform.to(device)


# =====================================================
# EMISSIONS
# =====================================================

def get_emission(waveform):

    with torch.inference_mode():

        emissions, _ = model(waveform)

        emissions = torch.log_softmax(
            emissions,
            dim=-1
        )

    return emissions[0].cpu()


# =====================================================
# FORCED ALIGNMENT
# =====================================================

def align_transcript(emission, transcript):

    transcript_ctc = transcript.replace(
        " ",
        "|"
    )

    token_ids = [
        token_to_idx[c]
        for c in transcript_ctc
        if c in token_to_idx
    ]

    if len(token_ids) == 0:

        raise ValueError(
            "Transcript contains no valid tokens."
        )

    # -----------------------------------------
    # CTC FEASIBILITY CHECK
    # -----------------------------------------

    repeats = sum(
        token_ids[i] == token_ids[i - 1]
        for i in range(1, len(token_ids))
    )

    available_frames = emission.shape[0]
    required_frames = len(token_ids) + repeats

    print(
        f"Frames={available_frames} | "
        f"Tokens={len(token_ids)} | "
        f"Required={required_frames}"
    )

    if available_frames < required_frames:

        print("\n================================")
        print("SKIPPING FILE")
        print("Transcript:", transcript)
        print(
            f"Need {required_frames} frames "
            f"but only have {available_frames}"
        )
        print("================================\n")

        return None

    targets = torch.tensor(
        [token_ids],
        dtype=torch.int32
    )

    input_lengths = torch.tensor(
        [available_frames],
        dtype=torch.int32
    )

    target_lengths = torch.tensor(
        [len(token_ids)],
        dtype=torch.int32
    )

    alignments, scores = torchaudio.functional.forced_align(
        emission.unsqueeze(0),
        targets,
        input_lengths,
        target_lengths,
        blank=0
    )

    return (
        token_ids,
        alignments[0].tolist(),
        scores[0].tolist()
    )


# =====================================================
# WORD TIMESTAMPS
# =====================================================

def extract_word_timestamps(
    transcript,
    token_ids,
    alignment,
):

    frame_duration = (
        320 / bundle.sample_rate
    )

    words = transcript.split()

    results = []

    token_pointer = 0

    for word in words:

        word_len = len(word)

        start_token = token_pointer
        end_token = token_pointer + word_len - 1

        start_frame = None
        end_frame = None

        target_pos = 0

        for frame_idx, token in enumerate(alignment):

            if target_pos >= len(token_ids):
                break

            if token == token_ids[target_pos]:

                if target_pos == start_token:
                    start_frame = frame_idx

                if target_pos == end_token:
                    end_frame = frame_idx

                target_pos += 1

        token_pointer += word_len + 1

        if (
            start_frame is not None
            and end_frame is not None
        ):

            results.append(
                {
                    "word": word,
                    "start": round(
                        start_frame *
                        frame_duration,
                        3
                    ),
                    "end": round(
                        end_frame *
                        frame_duration,
                        3
                    )
                }
            )

    return results


# =====================================================
# PROCESS FILE
# =====================================================

def process_file(audio_file):

    stem = audio_file.stem

    transcript_file = (
        TRANSCRIPT_DIR /
        f"{stem}.txt"
    )

    if not transcript_file.exists():

        print(
            f"Transcript missing: "
            f"{transcript_file}"
        )
        return

    print(f"\nProcessing {stem}")

    transcript = transcript_file.read_text(
        encoding="utf-8"
    )

    transcript = normalize_text(
        transcript
    )

    waveform = load_audio(
        audio_file
    )

    emission = get_emission(
        waveform
    )

    print("Transcript:", transcript)
    print(
        "Transcript chars:",
        len(transcript)
    )
    print(
        "Emission frames:",
        emission.shape[0]
    )

    result = align_transcript(
        emission,
        transcript
    )

    if result is None:

        print(
            f"Skipped -> {stem}"
        )
        return

    token_ids, alignment, scores = result

    words = extract_word_timestamps(
        transcript,
        token_ids,
        alignment
    )

    output_json = (
        OUTPUT_DIR /
        f"{stem}.json"
    )

    with open(
        output_json,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            words,
            f,
            indent=4
        )

    print(
        f"Saved -> {output_json}"
    )

    for item in words:

        print(
            f"{item['word']:15s}"
            f"{item['start']:8.2f}s"
            f" -> "
            f"{item['end']:8.2f}s"
        )


# =====================================================
# MAIN
# =====================================================

def main():

    audio_files = sorted(
        AUDIO_DIR.glob("*.wav")
    )

    if not audio_files:

        print(
            "No audio files found."
        )
        return

    for audio_file in audio_files:

        try:

            process_file(
                audio_file
            )

        except Exception as e:

            print(
                f"\nERROR processing "
                f"{audio_file.stem}"
            )

            print(e)

            continue


if __name__ == "__main__":
    main()