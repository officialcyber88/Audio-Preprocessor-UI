# @title Audio Preprocessor (Google Colab)
import sys
import shutil
import os
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
from pydub import AudioSegment
import concurrent.futures
from tqdm import tqdm
import torch
import tempfile
import warnings
import pyloudnorm as pyln
import subprocess  # Added missing import
from dataclasses import dataclass

# === 0. Dependency Check ===
def check_dependencies():
    """Ensure all external binaries and Python packages are available."""
    missing = []
    if shutil.which("ffmpeg") is None:
        missing.append("ffmpeg (binary on PATH)")
    if missing:
        sys.stderr.write("Missing dependencies:\n")
        for m in missing:
            sys.stderr.write(f"  • {m}\n")
        sys.exit(1)

check_dependencies()

# Suppress librosa warnings
warnings.filterwarnings("ignore", category=UserWarning)

# === Check GPU Availability ===
use_cuda_global = torch.cuda.is_available()
device_global = torch.device('cuda' if use_cuda_global else 'cpu')
print(f"[Info] Using {'CUDA GPU' if use_cuda_global else 'CPU'} for processing")
print(f"[Info] Device: {device_global}")
cpu_count = os.cpu_count() or 1
print(f"[Info] Detected {cpu_count} CPU cores → Using "
      f"{max(1, min(cpu_count // 2, 4)) if use_cuda_global else max(1, min(cpu_count, 8))} workers")
print()  # Blank line

# === Audio Processing Configuration ===
# @markdown ---
# @markdown ## Input Settings
input_path = "/content/drive/MyDrive/Drum Kits/Metro Grams Vol.1/snare/aggressive/snare_4.wav"  # @param {type:"string"}
output_path = "/content/processed/dataset"  # @param {type:"string"}

# @markdown ---
# @markdown ## Audio Format Settings
out_format = "wav"  # @param ["wav", "mp3", "flac", "aiff"]
sample_rate = "41000"  # @param ["16000", "41000", "48000", "96000"]
bit_depth = "24"  # @param ["16", "24", "32"]
channels = "mono"  # @param ["mono", "stereo"]
mp3_bitrate = "192k"  # @param ["128k", "192k", "320k"]
normalization_profile = "Spotify"  # @param ["Spotify", "default"]
# Define platform-specific loudness targets
profile_settings = {
    "Spotify": {"lufs": -14.0, "peak": -1.0},
    "default": {"lufs": None, "peak": None}  # No change
}

# Set target values based on profile
if normalization_profile == "default":
    print("[Info] Default selected: using original audio LUFS and Peak.")
    TARGET_LUFS = None
    TARGET_PEAK_DB = None
    normalize_audio = False  # Disable normalization for default
else:
    TARGET_LUFS = profile_settings[normalization_profile]["lufs"]
    TARGET_PEAK_DB = profile_settings[normalization_profile]["peak"]
    normalize_audio = True   # Enable normalization for Spotify
    print(f"[Normalize] Target: {TARGET_LUFS} LUFS, {TARGET_PEAK_DB} dBTP for {normalization_profile}")

# User setting: panning correction enabled or not (Yes/No)
panning_correction = "Yes"  # @param ["Yes", "No"]
# Convert to boolean
panning_correction = (panning_correction == "Yes")
if panning_correction:
    print("[Info] Panning correction is enabled.")
else:
    print("[Info] Panning correction is disabled.")

visualize = True  # @param {type:"boolean"}

# @markdown ---
# @markdown ## Segmentation Options
segmentation = False  # @param {type:"boolean"}
duration = 30  # @param {type:"slider", min:1, max:60, step:1}
time_unit = "Seconds"  # @param ["Milliseconds", "Seconds", "Minutes", "Hours"]

# === Auto Worker Configuration ===
num_workers = max(1, min(cpu_count // 2, 4)) if use_cuda_global else max(1, min(cpu_count, 8))

# === Directory Handling ===
format_subfolder = {'wav': 'wav', 'mp3': 'mp3', 'flac': 'flac', 'aiff': 'aiff'}[out_format]
export = os.path.join(output_path, format_subfolder)
os.makedirs(export, exist_ok=True)

# === Constants ===
TOP_DB = 40
DB_THRESH = -45
EDGE_SILENCE_THRESHOLD = 3e-3
VALID_FORMATS = ('.wav', '.mp3', '.flac', '.aiff')
MIN_FRAMES_FOR_RMS = 50
DEFAULT_HOP_LENGTH = 512

# === Loudness Normalization Helpers ===
MIN_BLOCK_SIZE = 0.050  # 50 ms
MIN_BLOCK_FLOOR = 0.001  # 1 ms

# === 1. Centralized Config ===
@dataclass(frozen=True)
class Config:
    sample_rate: int
    bit_depth: str
    channels: str
    normalization_profile: str
    TARGET_LUFS: float
    TARGET_PEAK_DB: float
    use_cuda: bool
    device: torch.device
    visualize: bool
    segmentation: bool
    duration_seconds: float
    panning_correction: bool
    mp3_bitrate: str

cfg = Config(
    sample_rate=int(sample_rate),
    bit_depth=bit_depth,
    channels=channels,
    normalization_profile=normalization_profile,
    TARGET_LUFS=TARGET_LUFS,
    TARGET_PEAK_DB=TARGET_PEAK_DB,
    use_cuda=use_cuda_global,
    device=device_global,
    visualize=visualize,
    segmentation=segmentation,
    duration_seconds=(
        duration if time_unit == "Seconds"
        else duration / 1000.0 if time_unit == "Milliseconds"
        else duration * 60 if time_unit == "Minutes"
        else duration * 3600
    ),
    panning_correction=panning_correction,
    mp3_bitrate=mp3_bitrate
)

def measure_loudness(y: np.ndarray, sr: int) -> dict:
    """
    Measure integrated LUFS and peak dB of an audio array.

    Parameters
    ----------
    y : np.ndarray
        Audio samples (mono or multichannel).
    sr : int
        Sample rate in Hz.

    Returns
    -------
    dict
        { 'lufs': float | None, 'peak': float }
    """
    BLOCK_SECONDS = 0.400
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y
    min_len = int(BLOCK_SECONDS * sr)
    if len(y_mono) < min_len:
        y_mono_padded = np.pad(y_mono, (0, min_len - len(y_mono)), mode="constant")
    else:
        y_mono_padded = y_mono
    meter = pyln.Meter(sr, block_size=BLOCK_SECONDS)
    try:
        lufs = meter.integrated_loudness(y_mono_padded)
    except:
        lufs = None
    peak_amp = np.max(np.abs(y))
    peak_db = 20 * np.log10(peak_amp) if peak_amp > 0 else -np.inf
    return {'lufs': lufs, 'peak': peak_db}

def normalize_loudness_true(y: np.ndarray, sr: int, log_message_func) -> tuple:
    """
    True-LUFS normalization with peak limiting fallback.

    Parameters
    ----------
    y : np.ndarray
    sr : int
    log_message_func : callable

    Returns
    -------
    tuple
        y_norm : np.ndarray
        report : dict
    """
    BLOCK_SECONDS = 0.400
    orig_peak_amp = np.max(np.abs(y))
    orig_peak_db = 20 * np.log10(orig_peak_amp) if orig_peak_amp > 0 else -np.inf
    log_message_func(f"[Normalize] Original Peak: {orig_peak_db:.2f} dBFS")
    y_mono = np.mean(y, axis=0) if y.ndim > 1 else y
    min_len = int(BLOCK_SECONDS * sr)
    if len(y_mono) < min_len:
        y_mono_padded = np.pad(y_mono, (0, min_len - len(y_mono)), mode="constant")
    else:
        y_mono_padded = y_mono
    meter = pyln.Meter(sr, block_size=BLOCK_SECONDS)
    try:
        orig_lufs = meter.integrated_loudness(y_mono_padded)
        log_message_func(f"[Normalize] Original LUFS: {orig_lufs:.2f} LUFS")
    except Exception as e:
        log_message_func(f"[Normalize] LUFS measurement failed: {e}")
        return normalize_by_peak_only(y, sr, log_message_func)
    # Use global TARGET_LUFS instead of cfg.TARGET_LUFS
    gain_lin = 10 ** ((TARGET_LUFS - orig_lufs) / 20)
    y_lufs = y * gain_lin
    log_message_func(f"[Normalize] Applied gain: {TARGET_LUFS - orig_lufs:.2f} dB to reach target LUFS")
    peak_amp_after = np.max(np.abs(y_lufs))
    peak_db_after = 20 * np.log10(peak_amp_after) if peak_amp_after > 0 else -np.inf
    log_message_func(f"[Normalize] Peak after LUFS: {peak_db_after:.2f} dBFS")
    # Use global TARGET_PEAK_DB instead of cfg.TARGET_PEAK_DB
    if peak_db_after > TARGET_PEAK_DB:
        scale = (10 ** (TARGET_PEAK_DB / 20)) / peak_amp_after
        y_norm = y_lufs * scale
        log_message_func(f"[Normalize] Applied peak limit to {TARGET_PEAK_DB:.1f} dBFS")
    else:
        y_norm = y_lufs
        log_message_func("[Normalize] No peak limiting needed")
    y_norm_mono = np.mean(y_norm, axis=0) if y_norm.ndim > 1 else y_norm
    if len(y_norm_mono) < min_len:
        y_norm_mono_padded = np.pad(y_norm_mono, (0, min_len - len(y_norm_mono)), mode="constant")
    else:
        y_norm_mono_padded = y_norm_mono
    try:
        final_lufs = meter.integrated_loudness(y_norm_mono_padded)
        log_message_func(f"[Normalize] Final LUFS: {final_lufs:.2f} LUFS")
    except:
        final_lufs = None
    final_peak_amp = np.max(np.abs(y_norm))
    final_peak_db = 20 * np.log10(final_peak_amp) if final_peak_amp > 0 else -np.inf
    log_message_func(f"[Normalize] Final Peak: {final_peak_db:.2f} dBFS")
    return y_norm, {
        'original_lufs':   orig_lufs,
        'original_peak':   orig_peak_db,
        'normalized_lufs': final_lufs,
        'normalized_peak': final_peak_db,
        'method':          'true_lufs',
        'duration':        len(y_norm) / sr
    }

def normalize_by_peak_only(y: np.ndarray, sr: int, log_message_func) -> tuple:
    """
    Peak-only normalization fallback.

    Parameters
    ----------
    y : np.ndarray
    sr : int
    log_message_func : callable

    Returns
    -------
    tuple
        y_norm : np.ndarray
        report : dict
    """
    log_message_func("[Normalize] Applying peak-only normalization")
    orig_peak = np.max(np.abs(y))
    orig_peak_db = 20 * np.log10(orig_peak) if orig_peak > 0 else -np.inf
    log_message_func(f"[Normalize] Original Peak: {orig_peak_db:.2f} dBFS")
    # Use global TARGET_PEAK_DB instead of cfg.TARGET_PEAK_DB
    if orig_peak > 0 and TARGET_PEAK_DB is not None:
        scale = (10 ** (TARGET_PEAK_DB / 20)) / orig_peak
        y_norm = y * scale
        final_peak_db = TARGET_PEAK_DB
    else:
        y_norm = y
        final_peak_db = orig_peak_db
    log_message_func(f"[Normalize] Final Peak (peak-only): {final_peak_db:.2f} dBFS")
    return y_norm, {
        'original_lufs':   None,
        'original_peak':   orig_peak_db,
        'normalized_lufs': None,
        'normalized_peak': final_peak_db,
        'method':          'peak_only',
        'duration':        len(y_norm) / sr
    }

def normalize_by_peak_only(y: np.ndarray, sr: int, log_message_func) -> tuple:
    """
    Peak-only normalization fallback.

    Parameters
    ----------
    y : np.ndarray
    sr : int
    log_message_func : callable

    Returns
    -------
    tuple
        y_norm : np.ndarray
        report : dict
    """
    log_message_func("[Normalize] Applying peak-only normalization")
    orig_peak = np.max(np.abs(y))
    orig_peak_db = 20 * np.log10(orig_peak) if orig_peak > 0 else -np.inf
    log_message_func(f"[Normalize] Original Peak: {orig_peak_db:.2f} dBFS")
    if orig_peak > 0 and cfg.TARGET_PEAK_DB is not None:
        scale = (10 ** (cfg.TARGET_PEAK_DB / 20)) / orig_peak
        y_norm = y * scale
        final_peak_db = cfg.TARGET_PEAK_DB
    else:
        y_norm = y
        final_peak_db = orig_peak_db
    log_message_func(f"[Normalize] Final Peak (peak-only): {final_peak_db:.2f} dBFS")
    return y_norm, {
        'original_lufs':   None,
        'original_peak':   orig_peak_db,
        'normalized_lufs': None,
        'normalized_peak': final_peak_db,
        'method':          'peak_only',
        'duration':        len(y_norm) / sr
    }

def pan_percent(left: np.ndarray, right: np.ndarray) -> tuple:
    """
    Compute percent power in left/right channels and deviation from center.
    """
    power_left = np.sum(left ** 2)
    power_right = np.sum(right ** 2)
    total = power_left + power_right
    if total < 1e-10:
        return 50.0, 50.0, 0.0
    left_pct = (power_left / total) * 100
    right_pct = (power_right / total) * 100
    return left_pct, right_pct, abs(left_pct - 50)

def normalize_panning(audio: np.ndarray, log_message_func) -> np.ndarray:
    """
    Apply corrective gain to right channel to match left RMS.

    Parameters
    ----------
    audio : np.ndarray
    log_message_func : callable

    Returns
    -------
    np.ndarray
    """
    if audio.ndim == 1:
        log_message_func("[Panning] Converting mono to stereo for correction")
        audio = np.stack([audio, audio], axis=0)
    left, right = audio[0], audio[1]
    rms_left = np.sqrt(np.mean(left ** 2))
    rms_right = np.sqrt(np.mean(right ** 2))
    if rms_right < 1e-6:
        return audio
    log_message_func("[Panning] Applying panning correction")
    return np.vstack((left, right * (rms_left / rms_right)))

def dbfs(amplitude: float) -> float:
    """
    Convert linear amplitude to dBFS.
    """
    return 20 * np.log10(amplitude) if amplitude > 0 else -np.inf

def soft_limiter(signal: np.ndarray, threshold_db: float=-3.00, ratio: float=10) -> np.ndarray:
    """
    Soft clip any samples above threshold with given ratio.

    Parameters
    ----------
    signal : np.ndarray
    threshold_db : float
    ratio : float

    Returns
    -------
    np.ndarray
    """
    threshold_lin = 10 ** (threshold_db / 20)
    if signal.ndim > 1:
        limited = np.zeros_like(signal)
        for c in range(signal.shape[0]):
            ch = signal[c]
            abs_ch = np.abs(ch)
            above = abs_ch > threshold_lin
            limited[c, ~above] = ch[~above]
            limited[c, above] = np.sign(ch[above]) * threshold_lin * ((abs_ch[above] / threshold_lin) ** (1/ratio))
        return limited
    else:
        abs_sig = np.abs(signal)
        limited = signal.copy()
        above = abs_sig > threshold_lin
        limited[above] = np.sign(signal[above]) * threshold_lin * ((abs_sig[above] / threshold_lin) ** (1/ratio))
        return limited

def format_duration(seconds: float) -> str:
    """
    Format a duration into the current time_unit.
    """
    if time_unit == "Milliseconds":
        ms = seconds * 1000
        return f"{ms:.2f} ms" if ms < 1000 else f"{seconds:.4f} s"
    if time_unit == "Seconds":
        return f"{seconds:.4f} s"
    if time_unit == "Minutes":
        return f"{seconds/60:.4f} min"
    if time_unit == "Hours":
        return f"{seconds/3600:.4f} hours"
    return f"{seconds:.4f} s"

def detect_clipping(y: np.ndarray, use_cuda: bool, device) -> tuple:
    """
    Detect sample clipping beyond ±0.999.

    Returns
    -------
    (clipped: bool, ratio: float)
    """
    if use_cuda:
        try:
            t = torch.tensor(y, device=device)
            mask = torch.abs(t) >= 0.999
            ratio = mask.float().mean().item()
            return ratio > 0.001, ratio
        except RuntimeError:
            mask = np.abs(y) >= 0.999
            ratio = mask.sum() / y.size
            return ratio > 0.001, ratio
    else:
        mask = np.abs(y) >= 0.999
        ratio = mask.sum() / y.size
        return ratio > 0.001, ratio

def attenuate_audio(y: np.ndarray, use_cuda: bool, device) -> np.ndarray:
    """
    Scale down waveform to avoid clipping, leaving 0.05 headroom.
    """
    if use_cuda:
        try:
            t = torch.tensor(y, device=device)
            peak = torch.max(torch.abs(t)).item()
            if peak == 0: return y
            factor = min(1.0 - 0.05, 1.0 / peak)
            return (t * factor).cpu().numpy()
        except RuntimeError:
            peak = np.max(np.abs(y))
            if peak == 0: return y
            factor = min(1.0 - 0.05, 1.0 / peak)
            return y * factor
    else:
        peak = np.max(np.abs(y))
        if peak == 0: return y
        factor = min(1.0 - 0.05, 1.0 / peak)
        return y * factor

def calculate_adaptive_hop_length(length: int) -> int:
    """
    Determine hop length for RMS frames based on total length.
    """
    return min(DEFAULT_HOP_LENGTH, max(32, length // MIN_FRAMES_FOR_RMS))

def auto_slice_audio(y: np.ndarray, sr: int, use_cuda: bool, device, log_message_func) -> tuple:
    """
    Rough silence trimming via frame-based RMS threshold.
    """
    L = len(y)
    if L < 128: return y, 0, L
    hop = calculate_adaptive_hop_length(L)
    frame_length = min(2048, L)
    if use_cuda:
        try:
            t = torch.tensor(y, device=device)
            frames = t.unfold(0, frame_length, hop)
            rms = torch.sqrt((frames ** 2).mean(dim=1))
            db = 20 * torch.log10(rms / rms.max() + 1e-7)
            db = db.cpu().numpy()
        except RuntimeError:
            log_message_func("[Silence] GPU OOM; falling back to CPU RMS")
            rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
            db = librosa.amplitude_to_db(rms, ref=np.max)
    else:
        rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
        db = librosa.amplitude_to_db(rms, ref=np.max)
    idx = np.where(db > DB_THRESH)[0]
    if idx.size == 0: return y, 0, L
    start = idx[0] * hop
    end = min(L, (idx[-1] + 1) * hop)
    return y[start:end], start, end

def slice_edge_silence(y: np.ndarray, thresh: float=EDGE_SILENCE_THRESHOLD) -> tuple:
    """
    Precise trim of very low-level edge silence.
    """
    abs_y = np.abs(y)
    nz = np.where(abs_y > thresh)[0]
    if nz.size == 0: return y, 0, len(y)
    return y[nz[0]: nz[-1] + 1], nz[0], nz[-1] + 1

def hard_slice_to_zero(y: np.ndarray, thresh: float=EDGE_SILENCE_THRESHOLD) -> np.ndarray:
    """
    Zero out samples before/after threshold crossings.
    """
    abs_y = np.abs(y)
    if not (abs_y > thresh).any(): return y
    s = np.argmax(abs_y > thresh)
    e = len(y) - np.argmax(abs_y[::-1] > thresh)
    y[:s] = 0; y[e:] = 0
    return y

def process_silence(y: np.ndarray, sr: int, use_cuda: bool, device, log_message_func) -> tuple:
    """
    Full two-stage silence trimming pipeline.
    """
    y_mono = librosa.to_mono(y) if y.ndim > 1 else y
    tmono, s0, e0 = auto_slice_audio(y_mono, sr, use_cuda, device, log_message_func)
    trimmed = (y[:, s0:e0] if y.ndim > 1 else y[s0:e0])
    fine_m, fs, fe = slice_edge_silence(tmono)
    fine = (trimmed[:, fs:fe] if y.ndim > 1 else trimmed[fs:fe])
    if fine.ndim > 1:
        proc = fine.copy()
        for c in range(proc.shape[0]):
            proc[c] = hard_slice_to_zero(fine[c], thresh=EDGE_SILENCE_THRESHOLD)
    else:
        proc = hard_slice_to_zero(fine.copy(), thresh=EDGE_SILENCE_THRESHOLD)
    proc[np.abs(proc) < 1e-6] = 0
    start = s0 + fs
    end = start + (fe - fs)
    pre = start / sr
    post = (len(y_mono) - end) / sr
    total = pre + post
    return proc, (pre, post, total), (start, end)

def plot_trim_boundaries(y: np.ndarray, sr: int, s0: int, e0: int, segments=None):
    """
    Plot waveform with trim boundaries (and optional segment markers).
    """
    if not cfg.visualize: return
    unit_dict = {"Milliseconds":("ms",1000),"Seconds":("s",1),
                 "Minutes":("min",1/60),"Hours":("hours",1/3600)}
    unit, fac = unit_dict.get(time_unit, ("s",1))
    times = np.arange(y.shape[-1]) / sr * fac
    plt.figure(figsize=(14,3))
    plt.plot(times, y.T if y.ndim>1 else y, alpha=0.8)
    plt.axvline(s0/sr*fac, linestyle='--', color='red', label='Trim Boundary')
    plt.axvline(e0/sr*fac, linestyle='--', color='red')
    if segments:
        for ss, ee in segments:
            plt.axvline(ss/sr*fac, linestyle='-', color='green', alpha=0.7)
            plt.axvline(ee/sr*fac, linestyle='-', color='green', alpha=0.7)
        plt.title(f"Waveform with Trim Boundaries (Red) and Segments (Green) ({time_unit})")
    else:
        plt.title(f"Waveform with Trim Boundaries (Red) ({time_unit})")
    plt.xlabel(f"Time ({unit})")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_zoomed_silence(y: np.ndarray, sr: int, s0: int, e0: int, zoom: float=0.05):
    """
    Plot a zoomed-in view of the silence before/after trim.
    """
    if not cfg.visualize: return
    zs = int(sr * zoom)
    unit_dict = {"Milliseconds":("ms",1000),"Seconds":("s",1),
                 "Minutes":("min",1/60),"Hours":("hours",1/3600)}
    unit, fac = unit_dict.get(time_unit, ("s",1))
    length = y.shape[-1] if y.ndim>1 else y.size
    if s0 > zs:
        pre = y[:,s0-zs:s0] if y.ndim>1 else y[s0-zs:s0]
        t0 = np.linspace(-zoom*fac,0,pre.shape[-1])
    else:
        pre = y[:s0] if y.ndim>1 else y[:s0]
        t0 = np.linspace(-pre.shape[-1]/sr*fac,0,pre.shape[-1])
    if e0+zs < length:
        post = y[:,e0:e0+zs] if y.ndim>1 else y[e0:e0+zs]
        t1 = np.linspace(0,zoom*fac,post.shape[-1])
    else:
        post = y[:,e0:] if y.ndim>1 else y[e0:]
        t1 = np.linspace(0,post.shape[-1]/sr*fac,post.shape[-1])
    fig, axs = plt.subplots(2,1,figsize=(14,4))
    if pre.shape[-1] > 0:
        axs[0].plot(t0, pre.T if pre.ndim>1 else pre)
        axs[0].set_xlim(t0[0], t0[-1])
    else:
        axs[0].text(0.5,0.5,"No silence before trim",ha='center',va='center')
        axs[0].set_xlim(-1,1)
    axs[0].set_title(f"Zoomed Silence Before Trim ({time_unit})")
    axs[0].set_xlabel(f"Time ({unit})")
    if post.shape[-1] > 0:
        axs[1].plot(t1, post.T if post.ndim>1 else post)
        axs[1].set_xlim(t1[0], t1[-1])
    else:
        axs[1].text(0.5,0.5,"No silence after trim",ha='center',va='center')
        axs[1].set_xlim(-1,1)
    axs[1].set_title(f"Zoomed Silence After Trim ({time_unit})")
    axs[1].set_xlabel(f"Time ({unit})")
    plt.tight_layout()
    plt.show()

def plot_normalized_waveform(y: np.ndarray, sr: int, title: str="Normalized Waveform"):
    """
    Plot the final processed or normalized waveform.
    """
    if not cfg.visualize: return
    length = y.shape[-1] if y.ndim>1 else y.size
    if length == 0: return
    unit_dict = {"Milliseconds":("ms",1000),"Seconds":("s",1),
                 "Minutes":("min",1/60),"Hours":("hours",1/3600)}
    unit, fac = unit_dict.get(time_unit, ("s",1))
    times = np.arange(length) / sr * fac
    plt.figure(figsize=(14,3))
    if y.ndim>1:
        for c in range(y.shape[0]):
            plt.plot(times, y[c], alpha=0.7, label=f'Channel {c+1}')
        plt.legend()
    else:
        plt.plot(times, y, alpha=0.8)
    plt.title(f"{title} ({time_unit})")
    plt.xlabel(f"Time ({unit})")
    plt.tight_layout()
    plt.show()

def get_all_audio_files(path: str) -> list:
    """
    Recursively collect all valid audio files under path.
    """
    files = []
    if os.path.isfile(path) and path.lower().endswith(VALID_FORMATS):
        files.append(path)
    else:
        for root, _, fnames in os.walk(path):
            for f in fnames:
                if f.lower().endswith(VALID_FORMATS):
                    files.append(os.path.join(root, f))
    return files

def get_export_path(file_path: str, input_dir: str, is_segmented: bool=False) -> str:
    """
    Compute and create export directory for a given file.
    """
    if os.path.isdir(input_dir):
        rel_path = os.path.relpath(file_path, input_dir)
        parts = rel_path.split(os.sep)
        volume_name = parts[0] if len(parts) > 1 else os.path.basename(input_dir)
    else:
        volume_name = os.path.splitext(os.path.basename(input_dir))[0]
    volume_name = volume_name.replace(' ', '_')
    base_path = os.path.join(export, volume_name)
    if is_segmented:
        file_name = os.path.splitext(os.path.basename(file_path))[0]
        base_path = os.path.join(base_path, f"{file_name}_segments")
    os.makedirs(base_path, exist_ok=True)
    return base_path

def export_audio(y: np.ndarray, sr: int, orig: str, fmt: str, idx: int=None) -> str:
    """
    Write processed audio to disk in the desired format.
    """
    is_segmented = idx is not None
    export_path = get_export_path(orig, input_path, is_segmented)
    base = os.path.splitext(os.path.basename(orig))[0]
    if cfg.channels == "mono" and y.ndim > 1:
        y = y.mean(axis=0)
    elif cfg.channels == "stereo" and y.ndim == 1:
        y = np.stack([y, y], axis=0)
    name = f"{base}_segment_{idx+1}.{fmt}" if is_segmented else f"{base}.{fmt}"
    out_path = os.path.join(export_path, name)
    if fmt.lower() == "mp3":
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()
        sf.write(tmp_path, y.T if y.ndim>1 else y, sr, subtype='FLOAT')
        try:
            cmd = [
                "ffmpeg", "-y", "-i", tmp_path,
                "-c:a", "libmp3lame",
                "-b:a", cfg.mp3_bitrate,
                "-ar", str(sr),
                out_path
            ]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            os.remove(tmp_path)
    else:
        actual_bit_depth = cfg.bit_depth
        if fmt == "flac" and cfg.bit_depth == "32":
            print("[Warning] 32-bit FLAC not supported; switching to 24-bit.")
            actual_bit_depth = "24"
        subtype = f"PCM_{actual_bit_depth}"
        sf.write(out_path, y.T if y.ndim>1 else y, sr, subtype=subtype)
    return out_path

def process_file(file: str, params: dict) -> tuple:
    """
    End-to-end processing of a single audio file, including trimming,
    normalization, segmentation, panning, and export.

    Returns
    -------
    tuple
        (file_path, results_dict)
    """
    file_results = {
        'messages': [],
        'export_paths': [],
        'plot_data': {},
        'total_silence': 0.0,
        'norm_reports': [],
        'original_metrics': None
    }
    def log_message(msg):
        file_results['messages'].append(msg)
    try:
        use_cuda = params['use_cuda']
        device = params['device']
        sample_rate_val = int(params['sample_rate'])
        current_segmentation = params['segmentation']
        duration_seconds = params['duration_seconds']
        time_unit_local = params['time_unit']
        normalize_audio_flag = params['normalize_audio']
        panning_correction_flag = params['panning_correction']

        log_message(f"[Info] Processing File: {file}")
        log_message(f"[Info] Processing Device: {'GPU' if use_cuda else 'CPU'}")

        try:
            y, sr = librosa.load(file, sr=sample_rate_val, mono=False)
        except Exception as e:
            log_message(f"[Error] Could not load audio: {e}")
            y, sr = librosa.load(file, sr=sample_rate_val, mono=True)

        log_message("")  # Blank line
        try:
            info = sf.info(file)
            log_message("[Info] Input Audio Details:")
            log_message(f"  Path: {file}")
            log_message(f"  Format: {os.path.splitext(file)[1][1:].upper()}")
            log_message(f"  Sample Rate: {info.samplerate} Hz")
            log_message(f"  Bit Depth: {info.subtype}")
            log_message(f"  Channels: {info.channels}")
            log_message(f"  Duration: {format_duration(info.duration)}")
            if y.ndim > 1 and y.shape[0] == 2:
                lp, rp, _ = pan_percent(y[0], y[1])
                log_message(f"  Pan Balance Input: Left = {lp:.2f}%, Right = {rp:.2f}%")
        except Exception as e:
            log_message(f"[Error] Could not read metadata: {e}")

        original_channels = "mono" if y.ndim == 1 else "stereo"
        log_message(f"[Info] Original audio channels: {original_channels}")

        if cfg.channels == "mono" and y.ndim > 1:
            log_message("[Info] Converting stereo → mono")
            y = librosa.to_mono(y)
        elif cfg.channels == "stereo" and y.ndim == 1:
            log_message("[Info] Converting mono → stereo")
            y = np.stack([y, y], axis=0)

        if y.ndim > 1 and y.shape[0] == 2:
            lp, rp, _ = pan_percent(y[0], y[1])
            log_message(f"[Info] Pan After Conversion: Left = {lp:.2f}%, Right = {rp:.2f}%")

        if panning_correction_flag:
            y = normalize_panning(y, log_message)
            if cfg.channels == "mono":
                y = librosa.to_mono(y)
                log_message("[Panning] Converted corrected stereo back to mono")
            else:
                lp, rp, _ = pan_percent(y[0], y[1])
                log_message(f"[Panning] Corrected Pan Balance: Left = {lp:.2f}%, Right = {rp:.2f}%")

        clipped, ratio = detect_clipping(y, use_cuda, device)
        if clipped:
            log_message(f"[Clipping] Detected {ratio:.2%} clipped → attenuating")
            y = attenuate_audio(y, use_cuda, device)

        y_proc, (pre, post, total_silence), (s_start, s_end) = process_silence(
            y, sr, use_cuda, device, log_message
        )
        log_message("")
        log_message("[Silence] Trim Results:")
        log_message(f"  Start silence: {format_duration(pre)}")
        log_message(f"  End silence: {format_duration(post)}")
        log_message(f"  Total silence removed: {format_duration(total_silence)}")
        file_results['total_silence'] = total_silence

        if not normalize_audio_flag:
            try:
                log_message("[Info] Measuring original loudness")
                orig_metrics = measure_loudness(y_proc, sr)
                file_results['original_metrics'] = orig_metrics
                if orig_metrics['lufs'] is not None:
                    log_message(f"  Original LUFS: {orig_metrics['lufs']:.2f} LUFS")
                log_message(f"  Original Peak: {orig_metrics['peak']:.2f} dBFS")
            except Exception as e:
                log_message(f"[Error] Loudness measure failed: {e}")

        norm_report = None
        if normalize_audio_flag and not current_segmentation:
            log_message("")
            log_message("[Normalize] Starting Loudness Normalization (full)")
            try:
                y_proc, norm_report = normalize_loudness_true(y_proc, sr, log_message)
                log_message("[Normalize] Completed full normalization")
            except Exception as e:
                log_message(f"[Normalize] Failed: {e}")
        file_results['norm_reports'].append(norm_report)

        processed_duration = y_proc.shape[-1] / sr
        export_paths = []
        segments = []
        seg_failed = False

        if current_segmentation:
            log_message(f"[Segmentation] Segment duration: {format_duration(duration_seconds)}")
            log_message(f"[Segmentation] Processed duration: {format_duration(processed_duration)}")

            if processed_duration < duration_seconds:
                log_message("[Error] Audio shorter than segment duration → skipping segmentation")
                seg_failed = True
                current_segmentation = False
            else:
                seg_len = int(sr * duration_seconds)
                total_len = y_proc.shape[-1]
                n_full = total_len // seg_len
                for i in range(n_full):
                    segments.append((i*seg_len, (i+1)*seg_len))
                rem = total_len % seg_len
                if rem > sr:
                    segments.append((n_full*seg_len, total_len))
                    log_message(f"[Segmentation] Final short segment: {format_duration(rem/sr)}")
                elif rem > 0:
                    log_message(f"[Segmentation] Skipping tiny remainder ({format_duration(rem/sr)})")
                log_message("")
                log_message("[Segmentation] Results:")
                log_message(f"  Total segments: {len(segments)}")
                log_message("")
                log_message("[Summary] Processing:")
                log_message(f"  Total silence removed: {format_duration(total_silence)}")
                log_message(f"  Total segments: {len(segments)}")
        else:
            log_message("")
            log_message("[Summary] Processing:")
            log_message(f"  Total silence removed: {format_duration(total_silence)}")
            log_message("  Exporting single file")

        file_results['plot_data'] = {
            'y': y,
            'y_proc': y_proc,
            'sr': sr,
            'samp_start': s_start,
            'samp_end': s_end,
            'segments': segments if segments else None,
            'segmentation_failed': seg_failed,
            'current_segmentation': current_segmentation,
            'total_len': y_proc.shape[-1] if y_proc is not None else 0
        }

        if current_segmentation and segments and not seg_failed:
            for i, (st, en) in enumerate(segments):
                seg_audio = y_proc[:, st:en] if y_proc.ndim > 1 else y_proc[st:en]
                seg_report = None
                if normalize_audio_flag:
                    log_message("")
                    log_message(f"[Normalize] Segment {i+1} normalization")
                    try:
                        seg_audio, seg_report = normalize_loudness_true(seg_audio, sr, log_message)
                        log_message(f"[Normalize] Completed segment {i+1}")
                    except Exception as e:
                        log_message(f"[Normalize] Segment {i+1} failed: {e}")
                file_results['norm_reports'].append(seg_report)
                path = export_audio(seg_audio, sr, file, out_format, i)
                export_paths.append(path)
                file_results['export_paths'].append({
                    'path': path,
                    'seg_index': i,
                    'seg_duration': seg_audio.shape[-1]/sr,
                    'norm_report': seg_report,
                    'original_metrics': file_results['original_metrics']
                })
        elif not seg_failed:
            path = export_audio(y_proc, sr, file, out_format)
            export_paths.append(path)
            file_results['export_paths'].append({
                'path': path,
                'seg_index': None,
                'seg_duration': processed_duration,
                'norm_report': norm_report,
                'original_metrics': file_results['original_metrics']
            })
        else:
            log_message("[Error] Export skipped due to segmentation failure")

        return file, file_results

    except Exception as e:
        file_results['messages'].append(f"[Error] Unexpected failure: {e}")
        return file, file_results

def run_tests():
    import numpy as np
    print("=== Running Basic Functional Tests ===")
    sr = 48000
    tone = 0.5 * np.sin(2 * np.pi * 440 * np.linspace(0,1,sr))
    silence = np.zeros(int(0.5 * sr))
    y = np.concatenate([silence, tone, silence]).astype(np.float32)

    y_trim, (pre,post,total), (s0,e0) = process_silence(y, sr, cfg.use_cuda, cfg.device, print)
    print(f"Trim results: pre={pre:.3f}s, post={post:.3f}s, total={total:.3f}s")

    stereo = np.vstack([y,y])
    lp,rp,dev = pan_percent(stereo[0], stereo[1])
    print(f"Pan percent: L={lp:.1f}%, R={rp:.1f}%, dev={dev:.1f}%")

    global TARGET_LUFS, TARGET_PEAK_DB
    sl,sp = cfg.TARGET_LUFS, cfg.TARGET_PEAK_DB
    TARGET_LUFS = -14.0; TARGET_PEAK_DB = -3.0
    try:
        y_norm, report = normalize_loudness_true(y_trim, sr, print)
        print(f"Norm report: {report}")
    finally:
        TARGET_LUFS, TARGET_PEAK_DB = sl, sp

    print("Basic functional tests completed.")

def main():
    tqdm.write(f"[Info] Export path: {export}")
    files = get_all_audio_files(input_path)
    tqdm.write(f"[Info] Processing {len(files)} audio file(s)")

    params = {
        'sample_rate': sample_rate,
        'segmentation': cfg.segmentation,
        'time_unit': time_unit,
        'duration': duration,
        'normalize_audio': normalize_audio,
        'panning_correction': cfg.panning_correction,
        'duration_seconds': cfg.duration_seconds,
        'use_cuda': cfg.use_cuda,
        'device': cfg.device
    }

    results = {}
    failed_files = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_file = {executor.submit(process_file, f, params): f for f in files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file),
                           total=len(files), desc="Processing files"):
            f = future_to_file[future]
            try:
                file, file_results = future.result()
                results[file] = file_results
            except Exception as e:
                tqdm.write(f"[Error] Processing {f} failed: {e}")
                failed_files.append(f)

    any_segmentation_failed = False

    for file, file_results in results.items():
        for msg in file_results['messages']:
            tqdm.write(msg)

        plot_data = file_results['plot_data']
        export_info = file_results['export_paths']

        if plot_data['segmentation_failed']:
            any_segmentation_failed = True

        if not plot_data['segmentation_failed']:
            if plot_data['current_segmentation'] and plot_data['segments']:
                plot_trim_boundaries(
                    plot_data['y_proc'],
                    plot_data['sr'],
                    0,
                    plot_data['total_len'],
                    plot_data['segments']
                )
            else:
                plot_trim_boundaries(
                    plot_data['y'],
                    plot_data['sr'],
                    plot_data['samp_start'],
                    plot_data['samp_end']
                )

            plot_zoomed_silence(
                plot_data['y'],
                plot_data['sr'],
                plot_data['samp_start'],
                plot_data['samp_end']
            )

            title = "Normalized Waveform" if normalize_audio else "Processed Waveform (without normalization)"
            plot_normalized_waveform(plot_data['y_proc'], plot_data['sr'], title)

        total_segments_count = (
            len(export_info) if cfg.segmentation and not plot_data['segmentation_failed']
            else (1 if export_info else 0)
        )

        for export_item in export_info:
            tqdm.write("")  # Blank line before Exported Audio Details
            if export_item['seg_index'] is not None:
                tqdm.write("Exported Segment Details:")
            else:
                tqdm.write("Exported Audio Details:")

            tqdm.write(f"  Path: {export_item['path']}")
            tqdm.write(f"  Format: {out_format.upper()}")

            try:
                info_exp = sf.info(export_item['path'])
                tqdm.write(f"  Sample Rate: {info_exp.samplerate} Hz")
                tqdm.write(f"  Bit Depth: {info_exp.subtype}")
                channels_exp = info_exp.channels
                ch_str = "Mono" if channels_exp == 1 else "Stereo"
                tqdm.write(f"  Channels: {ch_str}")
            except Exception as e:
                tqdm.write(f"  [Error] Could not read exported file metadata: {e}")

            if out_format.lower() == "mp3":
                tqdm.write(f"  MP3 Bitrate: {cfg.mp3_bitrate}")

            if cfg.segmentation and not plot_data['segmentation_failed']:
                tqdm.write(f"  Total Segments: {total_segments_count}")

            if cfg.segmentation:
                tqdm.write(f"  Time Unit: {time_unit}")

            tqdm.write(f"  Duration: {format_duration(export_item['seg_duration'])}")

            try:
                y_exp, sr_exp = librosa.load(export_item['path'], sr=None, mono=False)
                if y_exp.ndim > 1 and y_exp.shape[0] == 2:
                    l_pct_out, r_pct_out, _ = pan_percent(y_exp[0], y_exp[1])
                    tqdm.write(f"  Pan Balance Output: Left = {l_pct_out:.2f}%, Right = {r_pct_out:.2f}%")
                else:
                    tqdm.write("  Pan Balance Output: Left = 50.00%, Right = 50.00%")
            except Exception as e:
                tqdm.write(f"  [Error] Could not compute pan balance: {e}")

            if not normalize_audio and export_item['original_metrics'] is not None:
                metrics = export_item['original_metrics']
                if metrics['lufs'] is not None:
                    tqdm.write(f"  Original LUFS: {metrics['lufs']:.2f} LUFS")
                tqdm.write(f"  Original Peak: {metrics['peak']:.2f} dBFS")

            norm_report = export_item['norm_report']
            if norm_report is not None:
                if norm_report['original_lufs'] is not None:
                    tqdm.write(f"  Original LUFS: {norm_report['original_lufs']:.2f} LUFS")
                else:
                    tqdm.write("  Original LUFS: Not measured")
                if norm_report.get('original_peak') is not None:
                    tqdm.write(f"  Original PEAK: {norm_report['original_peak']:.2f} dBFS")
                else:
                    tqdm.write("  Original PEAK: N/A")
                if norm_report['normalized_lufs'] is not None:
                    tqdm.write(f"  Normalized LUFS: {norm_report['normalized_lufs']:.2f} LUFS")
                else:
                    tqdm.write("  Normalized LUFS: Not measured")
                tqdm.write(f"  Normalized Peak: {norm_report['normalized_peak']:.2f} dBFS")
            elif normalize_audio:
                try:
                    y_exp2, _ = librosa.load(export_item['path'], sr=None, mono=False)
                    peak_amp = np.max(np.abs(y_exp2))
                    peak_db = 20 * np.log10(peak_amp) if peak_amp > 0 else -np.inf
                    tqdm.write(f"  Measured Peak: {peak_db:.2f} dBFS")
                except:
                    tqdm.write("  [Error] Cannot measure peak")

            total_sil = file_results.get('total_silence', 0.0)
            tqdm.write(f"  Total silence removed: {format_duration(total_sil)}")

            device_str = "GPU" if cfg.use_cuda else "CPU"
            tqdm.write(f"  Processing Device: {device_str}")

    if failed_files:
        tqdm.write("[Error] Some files failed:")
        for fn in failed_files:
            tqdm.write(f"  - {fn}")

    if not any_segmentation_failed:
        tqdm.write(f"[Info] Processing complete! Processed {len(files)-len(failed_files)} file(s) successfully.")
    return results

# === RUN TESTS + MAIN, then DISPLAY AUDIO PLAYERS ===
run_tests()
results = main()

print("\n\n--- Preprocessed Audio Previews ---\n")
for file, file_results in results.items():
    for export_item in file_results['export_paths']:
        print("▶", export_item['path'])
        display(ipd.Audio(export_item['path']))
