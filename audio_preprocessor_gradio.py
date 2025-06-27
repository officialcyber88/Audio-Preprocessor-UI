import sys
import shutil
import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import torch
import tempfile
import warnings
import pyloudnorm as pyln
import subprocess
from dataclasses import dataclass
import gradio as gr
import traceback
from uuid import uuid4
import zipfile

# === Pre-flight Check: Install missing dependency if needed ===
try:
    import resampy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "resampy"])
    import resampy

# === Dependency Check ===
if shutil.which("ffmpeg") is None:
    sys.stderr.write("Missing dependency: ffmpeg\n")
    sys.exit(1)
warnings.filterwarnings("ignore", category=UserWarning)

# === Global Setup ===
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
OUTPUT_DIR = tempfile.mkdtemp(prefix="audio_preprocessor_")
print(f"Audio output will be saved in: {OUTPUT_DIR}")

# === Constants ===
DB_THRESH = -45
EDGE_SILENCE_THRESHOLD = 3e-3
VALID_FORMATS = ('.wav', '.mp3', '.flac', '.aiff', '.ogg', '.m4a')
MIN_FRAMES_FOR_RMS = 50
DEFAULT_HOP_LENGTH = 512

# === Config Dataclass ===
@dataclass(frozen=True)
class Config:
    sample_rate: int
    bit_depth: str
    channels: str
    target_lufs: float
    target_peak: float
    use_cuda: bool
    device: torch.device
    visualize: bool
    segmentation: bool
    duration: float
    panning: bool
    mp3_bitrate: str

# === Helpers ===
def pan_percent(l, r):
    pl, pr = np.sum(l**2), np.sum(r**2)
    t = pl + pr
    if t < 1e-10:
        return 50.0, 50.0, 0.0
    lp, rp = pl/t*100, pr/t*100
    return lp, rp, abs(lp-50)

def calculate_adaptive_hop_length(L):
    return min(DEFAULT_HOP_LENGTH, max(32, L//MIN_FRAMES_FOR_RMS))

def measure_loudness(y, sr):
    BLOCK = 0.4
    if y.size == 0:
        return {'lufs': None, 'peak': -np.inf}
    y_m = np.mean(y, axis=0) if y.ndim > 1 else y
    min_len = int(BLOCK * sr)
    if len(y_m) < min_len:
        y_m = np.pad(y_m, (0, min_len-len(y_m)))
    meter = pyln.Meter(sr, block_size=BLOCK)
    try:
        lufs = meter.integrated_loudness(y_m)
    except:
        lufs = None
    pk = np.max(np.abs(y))
    pdb = 20*np.log10(pk) if pk > 0 else -np.inf
    return {'lufs': lufs, 'peak': pdb}

def normalize_loudness(y, sr, log, tgt_lufs, tgt_peak):
    if y.size == 0:
        return y, {'method':'empty'}
    orig_pk = np.max(np.abs(y))
    orig_db = 20*np.log10(orig_pk) if orig_pk > 0 else -np.inf
    y_m = np.mean(y, axis=0) if y.ndim > 1 else y
    BLOCK = 0.4
    min_len = max(int(BLOCK*sr), int(0.001*sr))
    if len(y_m) < min_len:
        log(f"[Normalize] Padding {min_len-len(y_m)} samples")
        y_m = np.pad(y_m, (0, min_len-len(y_m)))
    meter = pyln.Meter(sr)
    try:
        orig_lufs = meter.integrated_loudness(y_m)
        log(f"[Normalize] Orig LUFS {orig_lufs:.2f}, Peak {orig_db:.2f}")
    except Exception as e:
        log(f"[Normalize] LUFS failed ({e}), peak-only")
        scale = (10**(tgt_peak/20)) / orig_pk if orig_pk > 0 else 1
        y_n = y * scale
        fp = 20*np.log10(np.max(np.abs(y_n))) if np.max(np.abs(y_n)) > 0 else -np.inf
        return y_n, {
            'original_lufs': None,
            'original_peak': orig_db,
            'normalized_lufs': None,
            'normalized_peak': fp,
            'method': 'peak_only'
        }
    gain = 10**((tgt_lufs - orig_lufs) / 20)
    y_l = y * gain
    pk_after = np.max(np.abs(y_l))
    if pk_after > 10**(tgt_peak/20):
        log("[Normalize] Limiting peak")
        y_n = y_l * (10**(tgt_peak/20) / pk_after)
    else:
        y_n = y_l
    fl = measure_loudness(y_n, sr)['lufs']
    fp = 20*np.log10(np.max(np.abs(y_n))) if np.max(np.abs(y_n)) > 0 else -np.inf
    log(f"[Normalize] Final LUFS {fl:.2f}, Peak {fp:.2f}")
    return y_n, {
        'original_lufs': orig_lufs,
        'original_peak': orig_db,
        'normalized_lufs': fl,
        'normalized_peak': fp,
        'method': 'true_lufs'
    }

def normalize_panning(a, log):
    if a.ndim != 2 or a.shape[0] != 2:
        return a
    lp, rp, _ = pan_percent(a[0], a[1])
    log(f"[Panning] Orig L{lp:.1f}% R{rp:.1f}%")
    r1, r2 = np.sqrt(np.mean(a[0]**2)), np.sqrt(np.mean(a[1]**2))
    if r1 < 1e-7 or r2 < 1e-7:
        return a
    corr = np.vstack((a[0], a[1] * (r1/r2)))
    lp2, rp2, _ = pan_percent(corr[0], corr[1])
    log(f"[Panning] Corr L{lp2:.1f}% R{rp2:.1f}%")
    return corr

def detect_clipping(y):
    if y.size == 0:
        return False, 0.0
    c = np.sum(np.abs(y) >= 0.999) / y.size
    return c > 0.001, c

def attenuate_clipped_audio(y, log):
    clip, ratio = detect_clipping(y)
    if clip:
        log(f"[Clipping] {ratio:.1%} clipped; attenuating")
        pk = np.max(np.abs(y))
        tgt = 10**(-1/20)
        if pk > 0:
            return y * (tgt/pk)
    return y

def auto_slice_audio(y, sr):
    if y.size == 0:
        return 0, 0
    L = y.shape[-1]
    if L < 128:
        return 0, L
    hop = calculate_adaptive_hop_length(L)
    frame = min(2048, L)
    rms = librosa.feature.rms(y=y, frame_length=frame, hop_length=hop)[0]
    db = librosa.amplitude_to_db(rms, ref=np.max)
    idx = np.where(db > DB_THRESH)[0]
    if idx.size == 0:
        return 0, L
    return idx[0] * hop, min(L, (idx[-1] + 1) * hop)

def process_silence(y, sr, log):
    if y.size == 0:
        return y, (0, 0, 0), (0, 0)
    L = y.shape[-1]
    y_m = np.mean(y, axis=0) if y.ndim > 1 else y
    s0, e0 = auto_slice_audio(y_m, sr)
    if e0 <= s0:
        log("[Silence] All silent")
        return np.array([]), (L/sr, 0, L/sr), (0, 0)
    t = y[..., s0:e0]
    tm = np.mean(t, axis=0) if t.ndim > 1 else t
    nz = np.where(np.abs(tm) > EDGE_SILENCE_THRESHOLD)[0]
    if nz.size == 0:
        log("[Silence] All trimmed")
        return np.array([]), (L/sr, 0, L/sr), (0, 0)
    fs, fe = nz[0], nz[-1] + 1
    final = t[..., fs:fe]
    rem = L - final.shape[-1]
    return final, ((s0+fs)/sr, (L-(s0+fe))/sr, rem/sr), (s0+fs, s0+fe)

def format_duration(s):
    return f"{s:.3f}s"

def plot_zoomed_silence(y, sr, s0, e0, zoom=0.05):
    zs = int(sr * zoom)
    fig, axs = plt.subplots(2, 1, figsize=(6, 4))
    pre = y[..., max(0, s0-zs):s0]
    t0 = np.linspace(-zoom, 0, pre.shape[-1])
    if pre.size > 0:
        axs[0].plot(t0, pre.T if pre.ndim>1 else pre)
    else:
        axs[0].text(0.5, 0.5, "No pre-silence", ha='center')
    axs[0].set_xlim(t0[0] if pre.size>0 else -zoom, 0)
    axs[0].set_title("Zoomed Silence Pre-trim")

    post = y[..., e0:e0+zs]
    t1 = np.linspace(0, zoom, post.shape[-1])
    if post.size > 0:
        axs[1].plot(t1, post.T if post.ndim>1 else post)
    else:
        axs[1].text(0.5, 0.5, "No post-silence", ha='center')
    axs[1].set_xlim(0, t1[-1] if post.size>0 else zoom)
    axs[1].set_title("Zoomed Silence Post-trim")

    plt.tight_layout()
    return fig

def plot_waveform(y, sr, title, time_unit_str="s", s0=None, e0=None, segments=None):
    fig, ax = plt.subplots(figsize=(6, 2.5))
    t = np.arange(y.shape[-1]) / sr
    if y.ndim > 1:
        for c in range(y.shape[0]):
            ax.plot(t, y[c], alpha=0.7, label=f'Ch{c+1}')
        ax.legend(fontsize="small")
    else:
        ax.plot(t, y)
    if s0 is not None and e0 is not None:
        ax.axvline(s0/sr, linestyle='--')
        ax.axvline(e0/sr, linestyle='--')
    if segments:
        for st, en in segments:
            ax.axvline(st/sr, linestyle='-', alpha=0.6)
            ax.axvline(en/sr, linestyle='-', alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel(f"Time ({time_unit_str})")
    ax.set_ylabel("Amplitude")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig

def get_all_audio_files(path):
    files = []
    if os.path.isfile(path) and path.lower().endswith(VALID_FORMATS):
        files.append(path)
    elif os.path.isdir(path):
        for root, _, fnames in os.walk(path):
            for f in fnames:
                if f.lower().endswith(VALID_FORMATS):
                    files.append(os.path.join(root, f))
    return files

def export_audio(y, sr, orig, fmt, cfg, exp_dir, idx=None):
    base = os.path.splitext(os.path.basename(orig))[0]
    name = f"{base}_segment_{idx+1}.{fmt}" if idx is not None else f"{base}.{fmt}"
    out = os.path.join(exp_dir, name)
    if y.size == 0:
        ch = 2 if cfg.channels == 'stereo' else 1
        y = np.zeros((ch,1)) if ch>1 else np.zeros(1)
    dat = y.T if y.ndim>1 else y
    if fmt == "mp3":
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, dat, sr, subtype="FLOAT")
            cmd = [
                "ffmpeg","-y","-i",tmp.name,
                "-c#a","libmp3lame","-b#a",cfg.mp3_bitrate,
                "-ar",str(sr), out
            ]
            subprocess.run(cmd, check=True, capture_output=True)
    else:
        styp = f"PCM_{cfg.bit_depth}"
        if fmt=="flac" and cfg.bit_depth=="32":
            styp="PCM_24"
        sf.write(out, dat, sr, subtype=styp)
    return out

def process_file(fp, cfg, params):
    res = {'messages': [], 'export_paths': [], 'plot_data': None}
    def log(m):
        res['messages'].append(m)
    try:
        log(f"--- Processing {os.path.basename(fp)} ---")
        log(f"Device: {'GPU' if cfg.use_cuda else 'CPU'}")
        y, orig_sr = librosa.load(fp, sr=None, mono=False)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        info = sf.info(fp)
        log(f"Input: {info.samplerate}Hz, {info.channels}ch, {format_duration(info.duration)}")
        # Panning
        if cfg.panning and y.shape[0] == 2:
            y = normalize_panning(y, log)
        # Resample
        if orig_sr != cfg.sample_rate:
            log(f"Resampling {orig_sr}→{cfg.sample_rate}")
            y = resampy.resample(y, orig_sr, cfg.sample_rate)
        if y.ndim == 1:
            y = y[np.newaxis, :]
        # Channels
        if cfg.channels == "mono" and y.shape[0] > 1:
            log("Converting to mono")
            y = np.mean(y, axis=0, keepdims=True)
        elif cfg.channels == "stereo" and y.shape[0] == 1:
            log("Converting to stereo")
            y = np.vstack([y, y])
        # Clipping
        y = attenuate_clipped_audio(y, log)
        y_pre = y.copy()
        # Silence trim
        y_proc, (pre, post, total), (s0, e0) = process_silence(y, cfg.sample_rate, log)
        log(f"Silence removed {format_duration(total)} (start {format_duration(pre)}, end {format_duration(post)})")
        if y_proc.size == 0:
            log("Empty after silence; skipping")
            return res
        # Prepare plot_data
        segments = []
        if cfg.segmentation:
            dur = y_proc.shape[-1] / cfg.sample_rate
            if dur < cfg.duration:
                log("⚠ shorter than segment; skipping export")
                return res
            else:
                ss = int(cfg.sample_rate * cfg.duration)
                nseg = int(np.ceil(dur / cfg.duration))
                segments = [(i*ss, min((i+1)*ss, y_proc.shape[-1])) for i in range(nseg)]
                log(f"Segmenting into {format_duration(cfg.duration)}, created {len(segments)} segments")
        else:
            segments = [(0, y_proc.shape[-1])]
        res['plot_data'] = {
            'y_pre': y_pre, 'y_proc': y_proc,
            'sr': cfg.sample_rate,
            's0': s0, 'e0': e0,
            'segments': segments
        }
        # Export & normalize
        normalize_flag = params['normalize'] != "No Normalization"
        for i, (st, en) in enumerate(segments):
            seg = y_proc[..., st:en]
            if normalize_flag:
                log(f"Normalizing {'segment '+str(i+1) if cfg.segmentation else 'file'}")
                seg, _ = normalize_loudness(seg, cfg.sample_rate, log, cfg.target_lufs, cfg.target_peak)
            out_path = export_audio(seg, cfg.sample_rate, fp, params['out_fmt'], cfg, OUTPUT_DIR, idx=(i if cfg.segmentation else None))
            res['export_paths'].append(out_path)
            log(f"✅ Exported {os.path.basename(out_path)}")
        return res

    except Exception:
        tb = traceback.format_exc()
        res['messages'].append(f"❌ ERROR:\n{tb}")
        return res

def gradio_process(input_mode, uploads, path_in,
                   out_fmt, sr, bd, ch, mp3_br,
                   norm, pan, seg, dur, tu, viz, zip_enable):
    logs = []

    # === Input Method & original file metadata ===
    logs.append("=== Input Method ===")
    if input_mode == "Path" and path_in.strip():
        logs.append(f"Mode: Path ➞ {path_in}")
        inputs = get_all_audio_files(path_in.strip())
    else:
        logs.append(f"Mode: Upload ➞ {len(uploads) if uploads else 0} file(s)")
        inputs = [f.name for f in uploads] if uploads else []

    # List each input file's original details
    logs.append("=== Original File Details ===")
    for fp in inputs:
        try:
            info = sf.info(fp)
            logs.append(f"{os.path.basename(fp)}: {info.samplerate}Hz, {info.channels}ch, {format_duration(info.duration)}")
        except:
            logs.append(f"{os.path.basename(fp)}: <could not read metadata>")
    logs.append("")

    # === Output Settings ===
    logs.append("=== Output Settings ===")
    logs.append(f"Format: {out_fmt}")
    logs.append(f"Sample Rate: {sr}")
    if out_fmt.lower() == "mp3":
        logs.append(f"MP3 Bitrate: {mp3_br}")
    else:
        logs.append(f"Bit Depth: {bd}")
    logs.append(f"Channels: {ch}")
    logs.append("")

    # === Processing Options ===
    logs.append("=== Processing Options ===")
    logs.append(f"Normalization Profile: {norm}")
    logs.append(f"Panning Correction: {pan}")
    logs.append(f"Segmentation: {'Yes' if seg else 'No'}" + (f", Duration {dur}{tu}" if seg else ""))
    logs.append(f"Show Visualizations: {'Yes' if viz else 'No'}")
    logs.append("")

    if not inputs:
        return "\n".join(logs + ["❌ No audio files provided."]), [], []

    tgt_lufs, tgt_peak = {"Spotify":(-14.0,-1.0), "Apple Music":(-16.0,-1.0)}.get(norm, (None,None))
    raw = float(dur)
    if tu == "Milliseconds":
        dsec = raw / 1000
    elif tu == "Minutes":
        dsec = raw * 60
    elif tu == "Hours":
        dsec = raw * 3600
    else:
        dsec = raw

    cfg = Config(
        sample_rate=int(sr.replace("Hz","")),
        bit_depth=bd,
        channels=ch,
        target_lufs=tgt_lufs,
        target_peak=tgt_peak,
        use_cuda=use_cuda,
        device=device,
        visualize=viz,
        segmentation=seg,
        duration=dsec,
        panning=(pan == "Yes"),
        mp3_bitrate=mp3_br
    )
    params = {'out_fmt': out_fmt, 'normalize': norm}

    gallery_images = []
    output_paths = []

    # Process each file
    for fp in inputs:
        r = process_file(fp, cfg, params)
        logs.extend(r['messages'])
        logs.append("")
        output_paths.extend(r['export_paths'])
        if viz and r.get('plot_data'):
            pd = r['plot_data']
            figs = [
                (plot_waveform(pd['y_pre'], pd['sr'], "Original w/ Trim", tu[0], pd['s0'], pd['e0'], pd['segments']), 'pre'),
                (plot_waveform(pd['y_proc'], pd['sr'], "Processed Output", tu[0]), 'post'),
                (plot_zoomed_silence(pd['y_pre'], pd['sr'], pd['s0'], pd['e0']), 'zoom')
            ]
            for fig, tag in figs:
                filename = os.path.join(OUTPUT_DIR, f"{uuid4().hex}_{tag}.png")
                fig.savefig(filename)
                gallery_images.append(filename)

    # === Handle ZIP option ===
    final_paths = output_paths
    if zip_enable:
        # Determine base name
        if input_mode == "Path" and path_in.strip():
            base = os.path.basename(path_in.rstrip("/"))
        else:
            base = os.path.splitext(os.path.basename(inputs[0]))[0] if inputs else "exported"
        zip_path = os.path.join(OUTPUT_DIR, f"{base}.zip")
        with zipfile.ZipFile(zip_path, 'w') as zf:
            for p in output_paths:
                zf.write(p, arcname=os.path.basename(p))
        final_paths = [zip_path]
        logs.append(f"Zipped exported files into: {os.path.basename(zip_path)}")

    # === Exported Files Summary & Settings Echo ===
    logs.append("=== Exported Files ===")
    logs.append(f"Count: {len(final_paths)}")
    logs.append("--- Used Settings ---")
    logs.append(f"Format: {out_fmt}")
    logs.append(f"Sample Rate: {sr}")
    if out_fmt.lower() == "mp3":
        logs.append(f"MP3 Bitrate: {mp3_br}")
    else:
        logs.append(f"Bit Depth: {bd}")
    logs.append(f"Channels: {ch}")
    logs.append(f"Normalization Profile: {norm}")
    logs.append(f"Panning Correction: {pan}")
    logs.append(f"Segmentation: {'Yes' if seg else 'No'}" + (f", Duration {dur}{tu}" if seg else ""))
    logs.append(f"Visualizations: {'Yes' if viz else 'No'}")
    logs.append("")
    for p in final_paths:
        logs.append(os.path.basename(p))

    return "\n".join(logs), gallery_images, final_paths

with gr.Blocks(title="Audio Preprocessor", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Audio Preprocessor")
    gr.Markdown(f"Outputs saved in `{OUTPUT_DIR}`")

    with gr.Row():
        with gr.Column(scale=2):
            input_mode = gr.Radio(["Upload","Path"], value="Upload", label="Input Method")
            file_uploader = gr.File(file_count="multiple", file_types=["audio"], label="Upload Audio Files")
            path_text     = gr.Textbox(placeholder="/path/to/dir", label="Or Enter Path", visible=False)
            input_mode.change(
                lambda m: (gr.update(visible=(m=="Upload")), gr.update(visible=(m=="Path"))),
                inputs=[input_mode],
                outputs=[file_uploader, path_text]
            )
        with gr.Column(scale=3):
            gr.Markdown("### Output Settings")
            with gr.Row():
                out_fmt     = gr.Dropdown(["wav","mp3","flac","aiff"], value="wav", label="Format")
                sample_rate = gr.Dropdown(["16000Hz","44100Hz","48000Hz","96000Hz","96000Hz"], value="44100Hz", label="Sample Rate")
                bit_depth   = gr.Dropdown(["16","24","32"], value="24", label="Bit Depth")
            with gr.Row():
                channels    = gr.Radio(["mono","stereo"], value="stereo", label="Channels")
                mp3_bitrate = gr.Dropdown(["128k","192k","256k","320k"], value="320k", label="MP3 Bitrate")

    with gr.Accordion("Processing Options", open=True):
        with gr.Row():
            norm_profile   = gr.Dropdown(["No Normalization","Spotify","Apple Music"], value="Spotify", label="Normalization Profile")
            panning_option = gr.Dropdown(["Yes","No"], value="Yes", label="Enable Panning Correction")
            visualize      = gr.Checkbox(value=True, label="Show Visualizations")
        with gr.Row():
            segmentation   = gr.Checkbox(value=False, label="Enable Segmentation")
            duration       = gr.Slider(minimum=1, maximum=60, step=1, value=15, label="Segment Duration")
            time_unit      = gr.Dropdown(["Milliseconds","Seconds","Minutes","Hours"], value="Seconds", label="Time Unit")
        with gr.Row():
            zip_enable     = gr.Checkbox(value=False, label="Save outputs as ZIP")

    process_btn = gr.Button("Process Audio", variant="primary")

    with gr.Tabs():
        with gr.TabItem("Logs"):
            logs_out = gr.Textbox(lines=15, label="Processing Logs", interactive=False)
        with gr.TabItem("Visualizations"):
            gr.Markdown("All waveform plots (3 per file)")
            gallery = gr.Gallery(label="Plots", columns=3, height="auto")
        with gr.TabItem("Output Files"):
            audio_out = gr.File(file_count="multiple", label="Processed Audio Files", interactive=False)

    process_btn.click(
        fn=gradio_process,
        inputs=[
            input_mode, file_uploader, path_text,
            out_fmt, sample_rate, bit_depth, channels, mp3_bitrate,
            norm_profile, panning_option, segmentation, duration, time_unit, visualize, zip_enable
        ],
        outputs=[logs_out, gallery, audio_out]
    )

if __name__ == "__main__":
    demo.launch(debug=True)
