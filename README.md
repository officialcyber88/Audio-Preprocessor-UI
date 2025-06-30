---
license: unlicense
title: Audio Preprocessor
sdk: gradio
emoji: 🧪

colorFrom: green
colorTo: gray
short_description: Audio Preprocessing
pinned: true
thumbnail: >-
  https://cdn-uploads.huggingface.co/production/uploads/685e0a37361aae99d2a3d907/reRACpFiIzn30l-uR2ZWy.png
sdk_version: 5.34.2
---

# Audio Preprocessor GUI

Try it out: [Audio Preprocessor GUI](https://officialcyber88-audio-preprocessor-ui.hf.space)

An interactive Gradio-based GUI tool for preprocessing audio files with options for resampling, loudness normalization, silence trimming, panning correction, segmentation, visualization, and export in various formats.

## Features

- **Multi-format Support**: Accepts `.wav`, `.mp3`, `.flac`, `.aiff`, `.ogg`, `.m4a` files  
- **Input Methods**: Upload files, enter a local path, or import from a Google Drive shared folder  
- **Preprocessing Options**:  
  - Loudness normalization (LUFS + peak-based)  
  - Optional silence trimming from start and end  
  - Stereo panning correction  
  - Sample rate conversion  
  - Bit depth adjustment  
  - Mono/stereo channel conversion  
- **Segmentation**: Slice long audio into equal time-based segments  
- **Visualization**: View original and processed waveform plots, including zoomed-in silence zones  
- **Batch Processing**: Multi-threaded support for folders or ZIP archives  
- **Export Formats**: `.wav`, `.mp3`, `.flac`, `.aiff` with configurable settings  
- **ZIP Export**: Bundle outputs into a downloadable ZIP archive  
- **CUDA Acceleration**: Uses GPU if available for faster processing

## Dependencies

- Python ≥ 3.7  
- `numpy`, `librosa`, `matplotlib`, `soundfile`, `torch`, `pyloudnorm`, `gradio`, `gdown`, `resampy`  
- System: `ffmpeg` (must be installed and on PATH)

> `resampy` and `gdown` are auto-installed at runtime if missing

## Installation

```bash
git clone https://github.com/officialcyber88/Audio-Preprocessor-GUI.git
cd audio-preprocessor-gui
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Make sure ffmpeg is installed and available in your PATH
```

## Usage

```bash
python app.py
```

1. Open the URL shown in your terminal (e.g. `http://127.0.0.1:7860`)

2. Choose an Input Method:  
   - **Upload**: Drop or browse multiple audio files  
   - **Path**: Enter a directory or ZIP archive path  
   - **Google Drive URL**: Paste a shareable folder link

3. Configure Output Settings:  
   - **Format**: `wav`, `mp3`, `flac`, `aiff`  
   - **Sample Rate**: `16000Hz`, `44100Hz`, `48000Hz`  
   - **Bit Depth**: `16`, `24`, `32` *(FLAC 32-bit falls back to 24-bit)*  
   - **Channels**: `mono`, `stereo`  
   - **MP3 Bitrate**: `128k`, `192k`, `256k`, `320k` (only shown for MP3)

4. Expand Processing Options:  
   - **Normalization Profile**: `No Normalization` | `Spotify` (–14 LUFS, –1 dB) | `Apple Music` (–16 LUFS, –1 dB)  
   - **Panning Correction**: Yes / No  
   - **Silence Trimming**: Yes / No  
   - **Show Visualizations**: Toggle waveform & silence‐zoom plots  
   - **Enable Segmentation**: Split into fixed‐length chunks  
     - *Duration*: slider + time unit (`Milliseconds`, `Seconds`, `Minutes`, `Hours`)  
   - **Save outputs as ZIP**: Yes / No + optional ZIP name

5. Click **Process Audio** and view:  
   - **Logs**: Detailed processing trace  
   - **Visualizations**: Waveform & silence‐zoom gallery  
   - **Output Files**: Download processed audio or ZIP  
   - **Audio Player**: Play processed files in-browser  

## Code Structure

- `Config` dataclass holds global settings (sample rate, bit depth, LUFS targets, etc.)  
- Pre-flight checks auto-install missing Python deps and verify `ffmpeg`  
- Helper functions for loudness measurement/normalization, panning correction, clipping detection, silence trimming, and plotting  
- `process_file` executes the pipeline per file, logs each step, generates plots, and exports  
- Gradio UI (`gr.Blocks`) organizes inputs, settings, and result tabs (Logs, Visualizations, Files, Player)  
- Concurrency via `ThreadPoolExecutor` for batch speed

## License

MIT License

---

Happy audio processing!
