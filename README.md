# Wav2Vec2 Forced Alignment

## Overview

This project performs word-level forced alignment using the Wav2Vec2 ASR model from Torchaudio.

The workflow:

1. Clean dataset files.
2. Generate word-level alignments.
3. Compare generated alignments against human annotations.
4. Produce evaluation reports and error metrics.

---

## Project Structure

```text
wav2vec2_forced_alignment/
│
├── data/
│   ├── audio/
│   └── transcripts/
│
├── human_annotations/
├── wav2vec2_outputs/
│
├── align.py
├── clean.py
├── compare.py
├── requirements.txt
└── README.md
```

---

## Installation

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows CMD:

```bash
venv\Scripts\activate.bat
```

Windows PowerShell:

```powershell
.\venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Requirements

```text
torch==2.3.1
torchaudio==2.3.1
numpy==1.26.4
```

---

## Dataset Format

Audio files:

```text
data/audio/
```

Transcript files:

```text
data/transcripts/
```

Human annotations:

```text
human_annotations/
```

Example:

```text
audio:
0001.wav

transcript:
0001.txt

annotation:
0001_Annotated.txt
```

---

## Step 1: Clean Dataset

Run:

```bash
python clean.py
```

This script:

* Removes invalid annotation sets
* Removes orphan transcripts
* Removes orphan audio files
* Generates a cleaning report

Output:

```text
cleaned.txt
```

---

## Step 2: Generate Forced Alignments

Run:

```bash
python align.py
```

This script:

* Loads audio files
* Normalizes transcripts
* Runs Wav2Vec2 forced alignment
* Generates word timestamps

Output:

```text
wav2vec2_outputs/*.json
```

Example output:

```json
[
  {
    "word": "hello",
    "start": 0.25,
    "end": 0.48
  }
]
```

---

## Step 3: Evaluate Alignments

Run:

```bash
python compare.py
```

This script compares generated alignments with human annotations.

Metrics:

* Word mismatch detection
* Timestamp difference detection
* Mean start error
* Mean end error
* Mean combined error

Threshold:

```text
70 ms
```

Generated reports:

```text
summary.json
red_flag_files.csv
word_level_errors.csv
```

---

## Outputs

### summary.json

Contains overall evaluation statistics.

### red_flag_files.csv

Lists files requiring manual review.

### word_level_errors.csv

Contains word-level alignment errors.

---

## Model

Model used:

WAV2VEC2_ASR_BASE_960H

Provided by Torchaudio.

---

## Usage

```bash
python clean.py
python align.py
python compare.py
```
