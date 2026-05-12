import os
import platform
import re
import subprocess
import unicodedata
import warnings
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

import torch
import whisper
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

warnings.filterwarnings("ignore", category=UserWarning)

TEMP_AUDIO = "temp_clean_audio.wav"
FAST_AUDIO_FILTER = "highpass=f=200,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11"
HQ_AUDIO_FILTER = (
    "highpass=f=120,lowpass=f=7500,afftdn=nf=-28,"
    "anlmdn=s=0.0003,dynaudnorm=f=250:g=15,loudnorm=I=-18:TP=-1.5:LRA=11"
)
VI_SMART_AUDIO_FILTER = (
    "highpass=f=80,lowpass=f=7600,afftdn=nf=-22,"
    "anlmdn=s=0.0002,dynaudnorm=f=250:g=12,loudnorm=I=-16:TP=-1:LRA=11"
)

VI_SMART_CONFIG: Dict[str, Any] = {
    "temperature": 0.0,
    "best_of": 5,
    "language": "vi",
    "task": "transcribe",
    "word_timestamps": True,
    "condition_on_previous_text": False,
    "no_speech_threshold": 0.6,
    "compression_ratio_threshold": 2.2,
    "logprob_threshold": -0.8,
    "hallucination_silence_threshold": 1.5,
}

VAD_CONFIG: Dict[str, Any] = {
    "min_speech_duration_ms": 250,
    "min_silence_duration_ms": 350,
    "speech_pad_ms": 180,
    "merge_gap_ms": 300,
    "min_chunk_sec": 4.0,
    "max_chunk_sec": 12.0,
}

HALLUCINATION_THRESHOLDS: Dict[str, Any] = {
    "avg_logprob_min": -0.9,
    "no_speech_prob_max": 0.7,
    "compression_ratio_max": 2.4,
    "repeat_token_run": 4,
    "min_wps": 1.0,
    "max_wps": 8.0,
    "short_seg_sec": 1.2,
    "short_seg_max_chars": 12,
}

DIRECTOR_CUE_KEYWORDS = [
    "2, 3",
    "2 3",
    "thử lại",
    "1 lần nữa",
    "nói lại",
    "quay lại",
    "bị trùng",
    "đoạn tiếp theo",
    "ok chị",
    "chị nói lại",
    "cho em lại",
    "chỉ một chỗ",
    "em lấy",
    "cắt ra",
    "comment á",
]

FILLER_WORDS = {
    "nha",
    "nhé",
    "nhe",
    "ạ",
    "à",
    "ừ",
    "ừm",
    "thì",
    "là",
    "vậy",
    "mà",
    "ơi",
}

def get_detailed_hardware_info() -> Dict[str, str]:
    sys_os = platform.system() # Windows, Darwin (Mac), Linux
    machine = platform.machine()
    gpu_name = "Không tìm thấy"
    device_used = "CPU"

    if sys_os == "Darwin" and machine == "arm64":
        device_used = "MPS (Mac Silicon)"
        gpu_name = f"Apple M-Series GPU ({machine})"
    elif torch.cuda.is_available():
        device_used = "GPU (CUDA)"
        gpu_name = torch.cuda.get_device_name(0)
    else:
        device_used = "CPU"
        gpu_name = "N/A (Sử dụng chip xử lý)"

    os_display = "Windows" if sys_os == "Windows" else ("macOS" if sys_os == "Darwin" else sys_os)
    
    return {
        "os": os_display,
        "gpu": gpu_name,
        "device": device_used
    }

def set_status(msg: str):
    with open("progress.txt", "w", encoding="utf-8") as f:
        f.write(msg)
    print(msg)


def get_system_config() -> Tuple[str, str]:
    sys_os = platform.system()
    machine = platform.machine()

    if sys_os == "Darwin" and machine == "arm64":
        return "mac_silicon", "h264_videotoolbox"
    if torch.cuda.is_available():
        return "cuda", "h264_nvenc"
    return "cpu", "libx264"


def concat_multiple_videos(uploaded_files, output_path: str) -> str:
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    list_file_path = "concat_list.txt"

    with open(list_file_path, "w", encoding="utf-8") as list_file:
        for file in uploaded_files:
            list_file.write(f"file '{file.path}'\n")

    command = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        list_file_path,
        "-c",
        "copy",
        output_path,
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Lỗi khi ghép video bằng FFmpeg: {e}")

    if os.path.exists(list_file_path):
        os.remove(list_file_path)
    return output_path


def preprocess_audio_for_whisper(video_path: str, mode: str = "vi_smart") -> Tuple[str, str]:
    requested_mode = (mode or "vi_smart").lower()
    if requested_mode not in {"fast", "hq", "vi_smart"}:
        requested_mode = "vi_smart"

    mode_chain = {
        "fast": FAST_AUDIO_FILTER,
        "hq": HQ_AUDIO_FILTER,
        "vi_smart": VI_SMART_AUDIO_FILTER,
    }
    if requested_mode == "hq":
        modes_to_try = ["hq", "vi_smart", "fast"]
    elif requested_mode == "fast":
        modes_to_try = ["fast", "vi_smart"]
    else:
        modes_to_try = ["vi_smart", "hq", "fast"]

    for mode_to_try in modes_to_try:
        command = [
            "ffmpeg",
            "-y",
            "-i",
            video_path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-af",
            mode_chain[mode_to_try],
            TEMP_AUDIO,
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return TEMP_AUDIO, mode_to_try
        except subprocess.CalledProcessError:
            continue

    return video_path, "source"


def split_sentences(text: str) -> List[str]:
    normalized = re.sub(r"\s+", " ", text.replace("\r", "\n")).strip()
    if not normalized:
        return []

    primary_parts = re.split(r"(?<=[.!?])\s+|\n+", normalized)
    primary_sentences = [p.strip(" \t-•") for p in primary_parts if p.strip()]

    refined: List[str] = []
    for sent in primary_sentences:
        if len(sent.split()) > 18 and "," in sent:
            clauses = [c.strip(" \t-•") for c in sent.split(",") if c.strip()]
            usable_clauses = [c for c in clauses if len(c.split()) >= 4]
            if usable_clauses:
                refined.extend(usable_clauses)
                continue
        refined.append(sent)

    return [s for s in refined if len(s.split()) >= 4]


def check_repeating_words(segments: List[Dict[str, Any]]) -> bool:
    for seg in segments:
        clean_text = re.sub(r"[^\w\s]", "", seg["text"].lower())
        words = clean_text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i + 1] == words[i + 2]:
                return True
    return False


def check_identical_segments(segments: List[Dict[str, Any]], reference_script: str = "", similarity_threshold: float = 0.75) -> bool:
    if len(segments) < 2:
        return False

    ref_sentences = []
    if reference_script.strip():
        raw_sentences = split_sentences(reference_script)
        ref_sentences = [re.sub(r"[^\w\s]", "", s.lower()).strip() for s in raw_sentences]

    for i in range(1, len(segments)):
        prev_text = re.sub(r"[^\w\s]", "", segments[i - 1]["text"].lower()).strip()
        curr_text = re.sub(r"[^\w\s]", "", segments[i]["text"].lower()).strip()

        if len(curr_text.split()) < 2:
            continue

        if curr_text == prev_text:
            is_in_script = False
            if ref_sentences:
                for ref_sent in ref_sentences:
                    score = SequenceMatcher(None, curr_text, ref_sent).ratio()
                    if score >= similarity_threshold:
                        is_in_script = True
                        break
            if not is_in_script:
                return True
    return False


def _build_vad_clip_timestamps(audio_path: str) -> List[float]:
    try:
        audio = AudioSegment.from_file(audio_path)
    except Exception:
        return []

    duration_ms = len(audio)
    if duration_ms <= 0:
        return []

    silence_thresh = max(-45.0, float(audio.dBFS) - 16.0)
    nonsilent_ranges = detect_nonsilent(
        audio,
        min_silence_len=int(VAD_CONFIG["min_silence_duration_ms"]),
        silence_thresh=silence_thresh,
        seek_step=10,
    )
    if not nonsilent_ranges:
        return []

    padded_ranges: List[Tuple[int, int]] = []
    pad_ms = int(VAD_CONFIG["speech_pad_ms"])
    min_speech_ms = int(VAD_CONFIG["min_speech_duration_ms"])
    for start_ms, end_ms in nonsilent_ranges:
        start_ms = max(0, start_ms - pad_ms)
        end_ms = min(duration_ms, end_ms + pad_ms)
        if end_ms - start_ms >= min_speech_ms:
            padded_ranges.append((start_ms, end_ms))

    if not padded_ranges:
        return []

    merged_ranges: List[Tuple[int, int]] = [padded_ranges[0]]
    merge_gap_ms = int(VAD_CONFIG["merge_gap_ms"])
    for start_ms, end_ms in padded_ranges[1:]:
        prev_start, prev_end = merged_ranges[-1]
        if start_ms - prev_end <= merge_gap_ms:
            merged_ranges[-1] = (prev_start, max(prev_end, end_ms))
        else:
            merged_ranges.append((start_ms, end_ms))

    clip_timestamps: List[float] = []
    max_chunk_ms = int(float(VAD_CONFIG["max_chunk_sec"]) * 1000)
    min_chunk_ms = int(float(VAD_CONFIG["min_chunk_sec"]) * 1000)

    for start_ms, end_ms in merged_ranges:
        seg_start = start_ms
        while seg_start < end_ms:
            seg_end = min(seg_start + max_chunk_ms, end_ms)
            if seg_end - seg_start < min_chunk_ms and seg_end < end_ms:
                seg_end = min(end_ms, seg_start + min_chunk_ms)
            clip_timestamps.extend([seg_start / 1000.0, seg_end / 1000.0])
            seg_start = seg_end

    # Tránh edge-case của mlx_whisper khi điểm cuối trùng chính xác thời lượng audio.
    if clip_timestamps and abs(clip_timestamps[-1] - (duration_ms / 1000.0)) < 1e-3:
        clip_timestamps[-1] = max(clip_timestamps[-2] + 0.05, clip_timestamps[-1] - 0.05)

    return clip_timestamps


def _safe_to_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _has_repeated_token_run(text: str, run_len: int = 4) -> bool:
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return False
    streak = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            streak += 1
            if streak >= run_len:
                return True
        else:
            streak = 1
    return False


def _segment_quality_flags(seg: Dict[str, Any]) -> List[str]:
    flags: List[str] = []
    text = (seg.get("text") or "").strip()
    duration = max(0.001, _safe_to_float(seg.get("end"), 0.0) - _safe_to_float(seg.get("start"), 0.0))
    avg_logprob = _safe_to_float(seg.get("avg_logprob"), -99.0)
    no_speech_prob = _safe_to_float(seg.get("no_speech_prob"), 0.0)
    compression_ratio = _safe_to_float(seg.get("compression_ratio"), 0.0)

    if avg_logprob < float(HALLUCINATION_THRESHOLDS["avg_logprob_min"]):
        flags.append("low_logprob")
    if no_speech_prob > float(HALLUCINATION_THRESHOLDS["no_speech_prob_max"]) and len(text) > 6:
        flags.append("high_no_speech_prob")
    if compression_ratio > float(HALLUCINATION_THRESHOLDS["compression_ratio_max"]):
        flags.append("high_compression_ratio")
    if _has_repeated_token_run(text, run_len=int(HALLUCINATION_THRESHOLDS["repeat_token_run"])):
        flags.append("repeated_tokens")

    word_count = len(re.findall(r"\w+", text))
    wps = word_count / duration
    if word_count >= 2 and (
        wps < float(HALLUCINATION_THRESHOLDS["min_wps"])
        or wps > float(HALLUCINATION_THRESHOLDS["max_wps"])
    ):
        flags.append("abnormal_speech_rate")

    if (
        duration < float(HALLUCINATION_THRESHOLDS["short_seg_sec"])
        and len(text) > int(HALLUCINATION_THRESHOLDS["short_seg_max_chars"])
    ):
        flags.append("too_much_text_for_short_segment")

    return flags


def _annotate_and_filter_segments(segments: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    quality_stats = {"accept": 0, "review": 0, "reject": 0}
    cleaned: List[Dict[str, Any]] = []

    for seg in segments:
        enriched = dict(seg)
        flags = _segment_quality_flags(enriched)
        if len(flags) >= 2:
            decision = "REJECT"
            quality_stats["reject"] += 1
        elif len(flags) == 1:
            decision = "REVIEW"
            quality_stats["review"] += 1
        else:
            decision = "ACCEPT"
            quality_stats["accept"] += 1

        enriched["quality_flags"] = flags
        enriched["quality_decision"] = decision
        if decision != "REJECT":
            cleaned.append(enriched)

    return cleaned, quality_stats


def _transcribe_with_openai_whisper(
    model: Any,
    audio_path: str,
    decode_kwargs: Dict[str, Any],
    fp16: bool,
) -> Dict[str, Any]:
    kwargs = dict(decode_kwargs)
    kwargs["fp16"] = fp16
    try:
        return model.transcribe(audio_path, **kwargs)
    except TypeError:
        # Fallback cho các phiên bản whisper cũ thiếu một số tham số.
        kwargs.pop("hallucination_silence_threshold", None)
        kwargs.pop("best_of", None)
        return model.transcribe(audio_path, **kwargs)


def _transcribe_with_mlx_whisper(
    mlx_whisper_module: Any,
    audio_path: str,
    decode_kwargs: Dict[str, Any],
    prompt_moi: str,
) -> Dict[str, Any]:
    base_kwargs = dict(decode_kwargs)
    base_kwargs["initial_prompt"] = prompt_moi
    base_kwargs["path_or_hf_repo"] = "mlx-community/whisper-large-v3-turbo"
    try:
        return mlx_whisper_module.transcribe(audio_path, **base_kwargs)
    except TypeError:
        for key in ["task", "best_of", "hallucination_silence_threshold", "clip_timestamps"]:
            base_kwargs.pop(key, None)
        return mlx_whisper_module.transcribe(audio_path, **base_kwargs)


def transcribe_audio(
    video_path: str,
    reference_script: str = "",
    transcribe_mode: str = "vi_smart",
) -> Tuple[List[Dict[str, Any]], bool, str]:
    device, _ = get_system_config()
    set_status(f"Bắt đầu xử lý âm thanh... Đang khởi tạo mô hình trên {device.upper()}")
    processed_audio, transcribe_mode_used = preprocess_audio_for_whisper(video_path, mode=transcribe_mode)
    prompt_moi = "Ngữ cảnh: hội thoại tiếng Việt. Từ khóa tham khảo: " + reference_script[:300]

    decode_kwargs = dict(VI_SMART_CONFIG)
    decode_kwargs["initial_prompt"] = prompt_moi

    clip_timestamps = _build_vad_clip_timestamps(processed_audio)
    if clip_timestamps:
        decode_kwargs["clip_timestamps"] = clip_timestamps
        set_status(f"Đã phát hiện {len(clip_timestamps) // 2} cụm thoại bằng VAD, đang bóc băng...")
    else:
        set_status("Không tách được cụm thoại từ VAD, đang bóc băng toàn bộ audio...")

    if device == "mac_silicon":
        try:
            import mlx_whisper
        except ImportError:
            raise Exception("Chưa cài đặt mlx-whisper!")
        result = _transcribe_with_mlx_whisper(
            mlx_whisper_module=mlx_whisper,
            audio_path=processed_audio,
            decode_kwargs=decode_kwargs,
            prompt_moi=prompt_moi,
        )
    else:
        model = whisper.load_model("large", device=device)
        use_fp16 = device == "cuda"
        result = _transcribe_with_openai_whisper(
            model=model,
            audio_path=processed_audio,
            decode_kwargs=decode_kwargs,
            fp16=use_fp16,
        )

    raw_segments = result.get("segments", [])

    # Nếu VAD quá gắt làm rỗng hoặc lọc quá mạnh, fallback transcribe toàn file.
    used_full_audio_fallback = False
    if clip_timestamps and len(raw_segments) == 0:
        used_full_audio_fallback = True
        decode_no_vad = dict(decode_kwargs)
        decode_no_vad.pop("clip_timestamps", None)
        if device == "mac_silicon":
            import mlx_whisper

            result = _transcribe_with_mlx_whisper(
                mlx_whisper_module=mlx_whisper,
                audio_path=processed_audio,
                decode_kwargs=decode_no_vad,
                prompt_moi=prompt_moi,
            )
        else:
            result = _transcribe_with_openai_whisper(
                model=model,
                audio_path=processed_audio,
                decode_kwargs=decode_no_vad,
                fp16=use_fp16,
            )
        raw_segments = result.get("segments", [])

    try:
        audio_full = AudioSegment.from_file(video_path)
        for seg in raw_segments:
            start_ms = int(_safe_to_float(seg.get("start"), 0.0) * 1000)
            end_ms = int(_safe_to_float(seg.get("end"), 0.0) * 1000)
            audio_chunk = audio_full[max(0, start_ms):max(0, end_ms)]
            loudness = audio_chunk.dBFS
            seg["loudness_dBFS"] = loudness if loudness != float("-inf") else -100.0
    except Exception:
        for seg in raw_segments:
            seg["loudness_dBFS"] = 0.0

    cleaned_segments = [
        seg for seg in raw_segments if seg.get("loudness_dBFS", 0.0) != -100.0
    ]
    final_cleaned_segments, quality_stats = _annotate_and_filter_segments(cleaned_segments)
    set_status(
        "Đã bóc băng xong. "
        f"ACCEPT={quality_stats['accept']}, REVIEW={quality_stats['review']}, REJECT={quality_stats['reject']}."
    )

    has_repeating_words = check_repeating_words(final_cleaned_segments)
    has_identical_segments = check_identical_segments(final_cleaned_segments, reference_script)
    is_hallucinating = (
        quality_stats["review"] > 0
        or quality_stats["reject"] > 0
        or has_repeating_words
        or has_identical_segments
    )

    mode_suffix = []
    if clip_timestamps:
        mode_suffix.append("vad")
    if used_full_audio_fallback:
        mode_suffix.append("fallback_full_audio")
    mode_name = transcribe_mode_used
    if mode_suffix:
        mode_name = f"{transcribe_mode_used} ({', '.join(mode_suffix)})"

    if os.path.exists(TEMP_AUDIO):
        os.remove(TEMP_AUDIO)

    return final_cleaned_segments, is_hallucinating, mode_name


def format_segments_to_text(segments: List[Dict[str, Any]]) -> str:
    lines = []
    for seg in segments:
        loudness = seg.get("loudness_dBFS", 0.0)
        lines.append(f"[{seg['start']:.2f} - {seg['end']:.2f} | {loudness:.2f} dBFS] {seg['text'].strip()}")
    return "\n".join(lines)


def _parse_transcript_line(line: str) -> Optional[Dict[str, Any]]:
    pattern = re.compile(
        r"^\s*\[(?P<start>\d+(?:\.\d+)?)\s*-\s*(?P<end>\d+(?:\.\d+)?)(?:\s*\|\s*(?P<loud>-?\d+(?:\.\d+)?)\s*dBFS)?\]\s*(?P<text>.*)$"
    )
    match = pattern.match(line.strip())
    if not match:
        return None

    start = float(match.group("start"))
    end = float(match.group("end"))
    if end <= start:
        return None

    text = (match.group("text") or "").strip()
    if not text:
        return None

    loud_str = match.group("loud")
    loudness = float(loud_str) if loud_str is not None else 0.0

    return {
        "start": start,
        "end": end,
        "text": text,
        "loudness_dBFS": loudness,
    }


def parse_transcript_text(transcript_text: str) -> List[Dict[str, Any]]:
    parsed: List[Dict[str, Any]] = []
    for line in transcript_text.splitlines():
        item = _parse_transcript_line(line)
        if item:
            parsed.append(item)

    parsed.sort(key=lambda s: (s["start"], s["end"]))
    return parsed


def _normalize_for_match(text: str) -> str:
    lowered = text.lower()
    lowered = re.sub(r"[^\w\s]", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    tokens = [tok for tok in lowered.split() if tok not in FILLER_WORDS]
    return " ".join(tokens)


def _strip_diacritics(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def _is_director_cue(text: str) -> bool:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    return any(keyword in normalized for keyword in DIRECTOR_CUE_KEYWORDS)


def preprocess_transcript_segments(
    segments: List[Dict[str, Any]],
    min_duration: float = 0.6,
    min_words: int = 2,
    max_gap: float = 0.8,
) -> List[Dict[str, Any]]:
    kept: List[Dict[str, Any]] = []
    for seg in segments:
        duration = seg["end"] - seg["start"]
        word_count = len(seg["text"].split())

        if _is_director_cue(seg["text"]):
            continue
        if duration < min_duration and word_count < min_words:
            continue

        kept.append(seg)

    if not kept:
        return []

    merged: List[Dict[str, Any]] = []
    current = dict(kept[0])

    for seg in kept[1:]:
        gap = seg["start"] - current["end"]
        # Không gộp 2 đoạn quá giống nhau để giữ retake tách biệt cho bước Last Best Take.
        if gap <= max_gap and not _is_retake_like_pair(current["text"], seg["text"]):
            old_duration = current["end"] - current["start"]
            new_duration = seg["end"] - seg["start"]
            total_duration = old_duration + new_duration
            if total_duration > 0:
                weighted_loudness = (
                    current.get("loudness_dBFS", 0.0) * old_duration
                    + seg.get("loudness_dBFS", 0.0) * new_duration
                ) / total_duration
            else:
                weighted_loudness = current.get("loudness_dBFS", 0.0)

            current["end"] = seg["end"]
            current["text"] = f"{current['text']} {seg['text']}".strip()
            current["loudness_dBFS"] = weighted_loudness
        else:
            merged.append(current)
            current = dict(seg)

    merged.append(current)
    return merged


def _partial_ratio(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 0.0

    if len(text_a) <= len(text_b):
        shorter, longer = text_a, text_b
    else:
        shorter, longer = text_b, text_a

    if shorter == longer:
        return 1.0
    if shorter in longer:
        return 1.0

    matcher = SequenceMatcher(None, shorter, longer)
    blocks = matcher.get_matching_blocks()
    best = 0.0
    short_len = len(shorter)
    long_len = len(longer)

    for block in blocks:
        start = max(0, block.b - block.a)
        end = min(long_len, start + short_len)
        start = max(0, end - short_len)
        window = longer[start:end]
        if not window:
            continue
        score = SequenceMatcher(None, shorter, window).ratio()
        if score > best:
            best = score

    if best == 0.0:
        best = SequenceMatcher(None, shorter, longer).ratio()
    return best


def _similarity(a: str, b: str) -> float:
    norm_a = _normalize_for_match(a)
    norm_b = _normalize_for_match(b)
    if not norm_a or not norm_b:
        return 0.0

    norm_a_ascii = _strip_diacritics(norm_a)
    norm_b_ascii = _strip_diacritics(norm_b)

    seq_score = max(
        SequenceMatcher(None, norm_a, norm_b).ratio(),
        SequenceMatcher(None, norm_a_ascii, norm_b_ascii).ratio(),
    )
    partial_score = max(
        _partial_ratio(norm_a, norm_b),
        _partial_ratio(norm_a_ascii, norm_b_ascii),
    )

    tokens_a = norm_a_ascii.split()
    tokens_b = norm_b_ascii.split()
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    if not set_a or not set_b:
        token_score = 0.0
        token_recall = 0.0
    else:
        overlap = len(set_a & set_b)
        token_score = (2 * overlap) / (len(set_a) + len(set_b))
        token_recall = overlap / len(set_a)

    shorter_is_script = len(tokens_a) <= len(tokens_b)
    if shorter_is_script:
        # Script ngắn hơn chunk: ưu tiên partial + recall để bắt "câu nằm trong chunk".
        return (
            0.20 * seq_score
            + 0.45 * partial_score
            + 0.35 * max(token_recall, token_score)
        )

    return 0.45 * seq_score + 0.25 * partial_score + 0.30 * token_score


def _is_retake_like_pair(text_a: str, text_b: str, threshold: float = 0.82) -> bool:
    return _similarity(text_a, text_b) >= threshold


def align_sequence_needleman_wunsch(
    script_sentences: List[str],
    transcript_chunks: List[Dict[str, Any]],
    match_threshold: float = 0.45,
    weak_match_threshold: float = 0.25,
    gap_sentence_penalty: float = -0.18,
    gap_chunk_penalty: float = -0.04,
    mismatch_penalty: float = -0.25,
) -> List[Tuple[int, int, float]]:
    m = len(script_sentences)
    n = len(transcript_chunks)
    if m == 0 or n == 0:
        return []

    scores = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            scores[i][j] = _similarity(script_sentences[i], transcript_chunks[j]["text"])

    dp = [[0.0] * (n + 1) for _ in range(m + 1)]
    trace = [[""] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = dp[i - 1][0] + gap_sentence_penalty
        trace[i][0] = "U"
    for j in range(1, n + 1):
        dp[0][j] = dp[0][j - 1] + gap_chunk_penalty
        trace[0][j] = "L"

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            sim = scores[i - 1][j - 1]
            diag_score = dp[i - 1][j - 1] + (sim if sim >= weak_match_threshold else mismatch_penalty)
            up_score = dp[i - 1][j] + gap_sentence_penalty
            left_score = dp[i][j - 1] + gap_chunk_penalty

            if diag_score >= up_score and diag_score >= left_score:
                dp[i][j] = diag_score
                trace[i][j] = "D"
            elif up_score >= left_score:
                dp[i][j] = up_score
                trace[i][j] = "U"
            else:
                dp[i][j] = left_score
                trace[i][j] = "L"

    matches: List[Tuple[int, int, float]] = []
    i, j = m, n
    while i > 0 and j > 0:
        move = trace[i][j]
        if move == "D":
            sim = scores[i - 1][j - 1]
            if sim >= match_threshold:
                matches.append((i - 1, j - 1, sim))
            i -= 1
            j -= 1
        elif move == "U":
            i -= 1
        else:
            j -= 1

    matches.reverse()
    return matches


def apply_last_best_take(
    script_sentences: List[str],
    chunks: List[Dict[str, Any]],
    matches: List[Tuple[int, int, float]],
    match_threshold: float = 0.45,
    retake_cluster_gap_sec: float = 5.0,
) -> List[Tuple[int, int, float]]:
    if not matches:
        return []

    refined: List[Tuple[int, int, float]] = []

    for idx, (script_idx, chunk_idx, original_score) in enumerate(matches):
        left_bound = matches[idx - 1][1] + 1 if idx > 0 else 0
        right_bound = matches[idx + 1][1] - 1 if idx < len(matches) - 1 else len(chunks) - 1

        if left_bound > right_bound:
            refined.append((script_idx, chunk_idx, original_score))
            continue

        candidates: List[Tuple[int, float]] = []
        cutoff = max(match_threshold - 0.07, 0.63)
        for k in range(left_bound, right_bound + 1):
            score = _similarity(script_sentences[script_idx], chunks[k]["text"])
            if score >= cutoff:
                candidates.append((k, score))

        if not candidates:
            refined.append((script_idx, chunk_idx, original_score))
            continue

        clusters: List[List[Tuple[int, float]]] = []
        for cand in candidates:
            if not clusters:
                clusters.append([cand])
                continue

            prev_idx = clusters[-1][-1][0]
            time_gap = chunks[cand[0]]["start"] - chunks[prev_idx]["end"]
            if time_gap <= retake_cluster_gap_sec:
                clusters[-1].append(cand)
            else:
                clusters.append([cand])

        target_cluster = None
        for cluster in clusters:
            if any(c[0] == chunk_idx for c in cluster):
                target_cluster = cluster
                break
        if target_cluster is None:
            target_cluster = min(clusters, key=lambda c: abs(c[-1][0] - chunk_idx))

        cluster_best = max(score for _, score in target_cluster)
        acceptable = [c for c in target_cluster if c[1] >= cluster_best - 0.10]
        chosen_idx, chosen_score = acceptable[-1] if acceptable else target_cluster[-1]

        refined.append((script_idx, chosen_idx, chosen_score))

    return refined


def _seconds_to_timecode(seconds: float, fps: int = 25) -> str:
    total_frames = max(0, int(round(seconds * fps)))
    hh = total_frames // (3600 * fps)
    mm = (total_frames % (3600 * fps)) // (60 * fps)
    ss = (total_frames % (60 * fps)) // fps
    ff = total_frames % fps
    return f"{hh:02d}:{mm:02d}:{ss:02d}:{ff:02d}"


def build_edl(timeline: List[Dict[str, Any]], title: str = "AUTO_CUT", fps: int = 25) -> str:
    lines = [f"TITLE: {title}", "FCM: NON-DROP FRAME", ""]
    record_cursor = 0.0

    event_number = 1
    for item in timeline:
        source_in = item["start"]
        source_out = item["end"]
        duration = max(0.0, source_out - source_in)
        if duration < 0.05:
            continue

        rec_in = record_cursor
        rec_out = rec_in + duration

        lines.append(
            f"{event_number:03d}  AX       V     C        "
            f"{_seconds_to_timecode(source_in, fps)} {_seconds_to_timecode(source_out, fps)} "
            f"{_seconds_to_timecode(rec_in, fps)} {_seconds_to_timecode(rec_out, fps)}"
        )
        lines.append(f"* FROM CLIP NAME: Script line {item['script_index'] + 1}")
        lines.append(f"* COMMENT: {item['script_text']}")

        record_cursor = rec_out
        event_number += 1

    return "\n".join(lines).strip() + "\n"


def build_simple_timeline_xml(timeline: List[Dict[str, Any]]) -> str:
    lines = ['<?xml version="1.0" encoding="UTF-8"?>', "<timeline>"]
    for item in timeline:
        lines.append(
            "  <clip "
            f"script_index=\"{item['script_index'] + 1}\" "
            f"start=\"{item['start']:.3f}\" "
            f"end=\"{item['end']:.3f}\" "
            f"score=\"{item['score']:.3f}\">"
        )
        lines.append(f"    <script_text>{_escape_xml(item['script_text'])}</script_text>")
        lines.append(f"    <matched_text>{_escape_xml(item['matched_text'])}</matched_text>")
        lines.append("  </clip>")
    lines.append("</timeline>")
    return "\n".join(lines) + "\n"


def _escape_xml(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))


def _token_coverage(script_text: str, transcript_text: str) -> float:
    script_tokens = set(_strip_diacritics(_normalize_for_match(script_text)).split())
    transcript_tokens = set(_strip_diacritics(_normalize_for_match(transcript_text)).split())
    if not script_tokens:
        return 0.0
    return len(script_tokens & transcript_tokens) / len(script_tokens)


def _loudness_norm(loudness_dbfs: float) -> float:
    return _clamp((loudness_dbfs + 30.0) / 10.0, 0.0, 1.0)


def _final_match_score(similarity: float, token_coverage: float, loudness_dbfs: float) -> float:
    return (
        0.72 * similarity
        + 0.15 * _loudness_norm(loudness_dbfs)
        + 0.13 * token_coverage
    )


def _is_candidate_accepted(similarity: float, loudness_dbfs: float, final_score: float) -> bool:
    similarity_ok = similarity >= 0.40 or (similarity >= 0.33 and loudness_dbfs >= -20.0)
    return similarity_ok and final_score >= 0.44


def _build_candidate_options_for_sentence(
    script_idx: int,
    script_sentence: str,
    chunks: List[Dict[str, Any]],
    retake_cluster_gap_sec: float = 5.0,
) -> List[Dict[str, Any]]:
    accepted_candidates: List[Dict[str, Any]] = []
    for chunk_idx, chunk in enumerate(chunks):
        similarity = _similarity(script_sentence, chunk["text"])
        loudness_dbfs = float(chunk.get("loudness_dBFS", -100.0))
        coverage = _token_coverage(script_sentence, chunk["text"])
        score = _final_match_score(similarity, coverage, loudness_dbfs)
        if _is_candidate_accepted(similarity, loudness_dbfs, score):
            accepted_candidates.append(
                {
                    "script_index": script_idx,
                    "chunk_index": chunk_idx,
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "script_text": script_sentence,
                    "matched_text": chunk["text"],
                    "loudness_dBFS": loudness_dbfs,
                    "similarity": similarity,
                    "token_coverage": coverage,
                    "score": score,
                }
            )

    if not accepted_candidates:
        return []

    accepted_candidates.sort(key=lambda x: (x["start"], x["end"]))
    clusters: List[List[Dict[str, Any]]] = []
    for cand in accepted_candidates:
        if not clusters:
            clusters.append([cand])
            continue
        prev = clusters[-1][-1]
        if cand["start"] - prev["end"] <= retake_cluster_gap_sec:
            clusters[-1].append(cand)
        else:
            clusters.append([cand])

    cluster_ranked: List[Dict[str, Any]] = []
    for cluster_id, cluster in enumerate(clusters):
        cluster_best = max(c["score"] for c in cluster)
        cluster_end = max(c["end"] for c in cluster)
        near_best = [c for c in cluster if c["score"] >= (cluster_best - 0.08)]
        near_best.sort(key=lambda c: (c["start"], c["end"]))
        preferred = near_best[-1]

        # Ưu tiên take cuối trong nhóm near-best, sau đó đến các candidate còn lại theo điểm.
        near_best_desc = list(reversed(near_best))
        remaining = [c for c in cluster if c not in near_best]
        remaining.sort(key=lambda c: (c["score"], c["start"], c["end"]), reverse=True)
        ordered_cluster_candidates = near_best_desc + remaining

        cluster_ranked.append(
            {
                "cluster_id": cluster_id,
                "cluster_best": cluster_best,
                "cluster_end": cluster_end,
                "preferred": preferred,
                "ordered_candidates": ordered_cluster_candidates,
            }
        )

    cluster_ranked.sort(key=lambda x: (x["cluster_best"], x["cluster_end"]), reverse=True)

    options: List[Dict[str, Any]] = []
    seen_chunk_indices = set()
    for cluster_info in cluster_ranked:
        for cand in cluster_info["ordered_candidates"]:
            if cand["chunk_index"] in seen_chunk_indices:
                continue
            seen_chunk_indices.add(cand["chunk_index"])
            options.append(cand)

    return options


def _resolve_one_to_one_candidates(
    script_candidate_options: Dict[int, List[Dict[str, Any]]],
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    selected_option_idx: Dict[int, int] = {
        script_idx: 0 for script_idx, opts in script_candidate_options.items() if opts
    }
    dedupe_conflicts_resolved = 0

    while True:
        chunk_to_scripts: Dict[int, List[int]] = {}
        for script_idx, option_idx in selected_option_idx.items():
            option = script_candidate_options[script_idx][option_idx]
            chunk_to_scripts.setdefault(option["chunk_index"], []).append(script_idx)

        conflict_groups = [scripts for scripts in chunk_to_scripts.values() if len(scripts) > 1]
        if not conflict_groups:
            break

        progressed = False
        for scripts in conflict_groups:
            winner = max(
                scripts,
                key=lambda s: (
                    script_candidate_options[s][selected_option_idx[s]]["score"],
                    script_candidate_options[s][selected_option_idx[s]]["similarity"],
                    -s,
                ),
            )

            for loser in scripts:
                if loser == winner or loser not in selected_option_idx:
                    continue

                current_idx = selected_option_idx[loser]
                next_idx = current_idx + 1
                found_next = False
                while next_idx < len(script_candidate_options[loser]):
                    next_option = script_candidate_options[loser][next_idx]
                    if next_option["chunk_index"] != script_candidate_options[loser][current_idx]["chunk_index"]:
                        found_next = True
                        break
                    next_idx += 1

                dedupe_conflicts_resolved += 1
                if found_next:
                    selected_option_idx[loser] = next_idx
                else:
                    del selected_option_idx[loser]
                progressed = True

        if not progressed:
            break

    resolved = {
        script_idx: script_candidate_options[script_idx][option_idx]
        for script_idx, option_idx in selected_option_idx.items()
    }
    return resolved, dedupe_conflicts_resolved


def _build_match_row(
    script_idx: int,
    chunk_idx: int,
    script_sentence: str,
    chunk: Dict[str, Any],
) -> Dict[str, Any]:
    similarity = _similarity(script_sentence, chunk["text"])
    token_coverage = _token_coverage(script_sentence, chunk["text"])
    loudness = float(chunk.get("loudness_dBFS", -100.0))
    score = _final_match_score(similarity, token_coverage, loudness)
    return {
        "script_index": script_idx,
        "chunk_index": chunk_idx,
        "start": float(chunk["start"]),
        "end": float(chunk["end"]),
        "script_text": script_sentence,
        "matched_text": chunk["text"],
        "loudness_dBFS": loudness,
        "similarity": similarity,
        "token_coverage": token_coverage,
        "score": score,
    }


def _expand_n_to_one_matches(
    script_sentences: List[str],
    chunks: List[Dict[str, Any]],
    matched_rows: Dict[int, Dict[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], int]:
    if not matched_rows:
        return matched_rows, 0

    used_chunk_indices = {row["chunk_index"] for row in matched_rows.values()}
    unmatched_script_indices = [i for i in range(len(script_sentences)) if i not in matched_rows]

    for script_idx in unmatched_script_indices:
        sentence = script_sentences[script_idx]
        best_row: Optional[Dict[str, Any]] = None

        for chunk_idx in used_chunk_indices:
            chunk = chunks[chunk_idx]
            candidate = _build_match_row(script_idx, chunk_idx, sentence, chunk)
            if (
                candidate["similarity"] < 0.35
                or candidate["token_coverage"] < 0.50
                or candidate["score"] < 0.43
            ):
                continue

            if best_row is None or (
                candidate["score"],
                candidate["similarity"],
                candidate["token_coverage"],
            ) > (
                best_row["score"],
                best_row["similarity"],
                best_row["token_coverage"],
            ):
                best_row = candidate

        if best_row is not None:
            matched_rows[script_idx] = best_row

    shared_groups = {}
    for row in matched_rows.values():
        shared_groups.setdefault(row["chunk_index"], []).append(row["script_index"])
    n_to_one_shared_chunks = sum(1 for script_indices in shared_groups.values() if len(script_indices) > 1)

    return matched_rows, n_to_one_shared_chunks


def _split_shared_chunk_intervals(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    grouped: Dict[int, List[int]] = {}
    for idx, row in enumerate(rows):
        chunk_idx = int(row.get("chunk_index", -1))
        grouped.setdefault(chunk_idx, []).append(idx)

    output = [dict(row) for row in rows]
    for chunk_idx, row_indices in grouped.items():
        if chunk_idx < 0 or len(row_indices) <= 1:
            continue

        ordered = sorted(row_indices, key=lambda i: output[i]["script_index"])
        chunk_start = min(float(output[i]["start"]) for i in ordered)
        chunk_end = max(float(output[i]["end"]) for i in ordered)
        total_duration = max(0.001, chunk_end - chunk_start)

        weights: List[float] = []
        for i in ordered:
            w = len(_normalize_for_match(output[i]["script_text"]).split())
            weights.append(max(1.0, float(w)))
        weight_sum = sum(weights)
        if weight_sum <= 0:
            weights = [1.0] * len(ordered)
            weight_sum = float(len(ordered))

        cursor = chunk_start
        for pos, i in enumerate(ordered):
            if pos == len(ordered) - 1:
                seg_start = cursor
                seg_end = chunk_end
            else:
                share = total_duration * (weights[pos] / weight_sum)
                seg_start = cursor
                seg_end = min(chunk_end, seg_start + max(0.05, share))
                cursor = seg_end

            output[i]["start"] = seg_start
            output[i]["end"] = seg_end
            output[i]["shared_chunk_index"] = chunk_idx
            output[i]["shared_chunk_group_size"] = len(ordered)

    return output


def deterministic_filter_pipeline(
    reference_script: str,
    transcript_text: str,
    chunk_gap_sec: float = 1.0,
) -> Dict[str, Any]:
    script_sentences = split_sentences(reference_script)
    raw_segments = parse_transcript_text(transcript_text)
    processed_chunks = preprocess_transcript_segments(raw_segments, max_gap=chunk_gap_sec)

    if not script_sentences or not processed_chunks:
        return {
            "filtered_script": "",
            "filtered_rows": [],
            "timeline_script_order": [],
            "timeline_time_order": [],
            "timeline": [],
            "unmatched_sentences": [
                {"script_index": i, "script_text": sentence}
                for i, sentence in enumerate(script_sentences)
            ],
            "edl": build_edl([], title="AUTO_CUT"),
            "timeline_xml": build_simple_timeline_xml([]),
            "stats": {
                "script_sentence_count": len(script_sentences),
                "raw_segment_count": len(raw_segments),
                "chunk_count": len(processed_chunks),
                "match_count": 0,
                "match_rate": 0.0,
                "dedupe_conflicts_resolved": 0,
                "low_loudness_selected_count": 0,
                "nw_match_count": 0,
                "last_best_take_count": 0,
                "n_to_one_shared_chunks": 0,
            },
        }

    nw_matches = align_sequence_needleman_wunsch(
        script_sentences=script_sentences,
        transcript_chunks=processed_chunks,
        match_threshold=0.42,
        weak_match_threshold=0.24,
        gap_sentence_penalty=-0.16,
        gap_chunk_penalty=-0.05,
        mismatch_penalty=-0.24,
    )
    refined_matches = apply_last_best_take(
        script_sentences=script_sentences,
        chunks=processed_chunks,
        matches=nw_matches,
        match_threshold=0.42,
    )

    resolved_map: Dict[int, Dict[str, Any]] = {}
    for script_idx, chunk_idx, _ in refined_matches:
        chunk = processed_chunks[chunk_idx]
        resolved_map[script_idx] = _build_match_row(
            script_idx=script_idx,
            chunk_idx=chunk_idx,
            script_sentence=script_sentences[script_idx],
            chunk=chunk,
        )

    resolved_map, n_to_one_shared_chunks = _expand_n_to_one_matches(
        script_sentences=script_sentences,
        chunks=processed_chunks,
        matched_rows=resolved_map,
    )

    filtered_rows = list(resolved_map.values())
    filtered_rows.sort(key=lambda x: x["script_index"])
    filtered_rows = _split_shared_chunk_intervals(filtered_rows)

    for row in filtered_rows:
        row["loudness_dBFS"] = round(float(row["loudness_dBFS"]), 2)
        row["similarity"] = round(float(row["similarity"]), 4)
        row["token_coverage"] = round(float(row["token_coverage"]), 4)
        row["score"] = round(float(row["score"]), 4)
        row["start"] = round(float(row["start"]), 3)
        row["end"] = round(float(row["end"]), 3)

    timeline_script_order = sorted((dict(row) for row in filtered_rows), key=lambda x: x["script_index"])
    timeline_time_order = sorted((dict(row) for row in filtered_rows), key=lambda x: (x["start"], x["end"]))

    filtered_lines = []
    for item in timeline_script_order:
        filtered_lines.append(
            f"[{item['start']:.2f} - {item['end']:.2f} | {item['loudness_dBFS']:.2f} dBFS | "
            f"sim {item['similarity']:.4f} | score {item['score']:.4f}] "
            f"SCRIPT: {item['script_text']} || RAW: {item['matched_text']}"
        )

    matched_script_indices = {item["script_index"] for item in timeline_script_order}
    unmatched = [
        {"script_index": i, "script_text": sentence}
        for i, sentence in enumerate(script_sentences)
        if i not in matched_script_indices
    ]
    low_loudness_selected_count = len([item for item in timeline_script_order if item["loudness_dBFS"] < -20.0])

    return {
        "filtered_script": "\n".join(filtered_lines),
        "filtered_rows": filtered_rows,
        "timeline_script_order": timeline_script_order,
        "timeline_time_order": timeline_time_order,
        "timeline": timeline_script_order,  # backward compatibility
        "unmatched_sentences": unmatched,
        "edl": build_edl(timeline_script_order, title="AUTO_CUT"),
        "timeline_xml": build_simple_timeline_xml(timeline_script_order),
        "stats": {
            "script_sentence_count": len(script_sentences),
            "raw_segment_count": len(raw_segments),
            "chunk_count": len(processed_chunks),
            "match_count": len(timeline_script_order),
            "match_rate": round((len(timeline_script_order) / len(script_sentences)) * 100, 2) if script_sentences else 0.0,
            "dedupe_conflicts_resolved": 0,
            "low_loudness_selected_count": low_loudness_selected_count,
            "nw_match_count": len(nw_matches),
            "last_best_take_count": len(refined_matches),
            "n_to_one_shared_chunks": n_to_one_shared_chunks,
        },
    }


def render_video_from_timeline(
    source_video_path: str,
    timeline: List[Dict[str, Any]],
    output_path: str,
    temp_dir: str,
) -> str:
    if not os.path.exists(source_video_path):
        raise Exception("Không tìm thấy video nguồn để cắt dựng.")
    if not timeline:
        raise Exception("Timeline rỗng. Không có đoạn nào để xuất video.")

    os.makedirs(temp_dir, exist_ok=True)
    clips_dir = os.path.join(temp_dir, "clips")
    if os.path.exists(clips_dir):
        for name in os.listdir(clips_dir):
            path = os.path.join(clips_dir, name)
            if os.path.isfile(path):
                os.remove(path)
    else:
        os.makedirs(clips_dir, exist_ok=True)

    clip_paths: List[str] = []
    for i, item in enumerate(timeline):
        start = max(0.0, float(item.get("start", 0.0)))
        end = max(start, float(item.get("end", 0.0)))
        if end - start < 0.05:
            continue

        clip_path = os.path.join(clips_dir, f"clip_{i:04d}.mp4")
        clip_paths.append(clip_path)
        command = [
            "ffmpeg",
            "-y",
            "-ss",
            f"{start:.3f}",
            "-to",
            f"{end:.3f}",
            "-i",
            source_video_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            clip_path,
        ]
        try:
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Lỗi cắt clip thứ {i + 1}: {e}")

    if not clip_paths:
        raise Exception("Không tạo được clip hợp lệ từ timeline.")

    concat_list_path = os.path.join(temp_dir, "concat_clips.txt")
    with open(concat_list_path, "w", encoding="utf-8") as f:
        for clip_path in clip_paths:
            f.write(f"file '{os.path.abspath(clip_path)}'\n")

    concat_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        concat_list_path,
        "-c",
        "copy",
        output_path,
    ]
    try:
        subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        # Fallback nối lại bằng re-encode nếu copy thất bại do mismatch metadata.
        fallback_cmd = [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-movflags",
            "+faststart",
            output_path,
        ]
        try:
            subprocess.run(fallback_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Lỗi khi nối các clip đã cắt: {e}")

    return output_path
