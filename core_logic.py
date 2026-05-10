import whisper
import torch
import subprocess
import os
import re
import platform
import warnings
import difflib
from pydub import AudioSegment
import requests

warnings.filterwarnings("ignore", category=UserWarning)

TEMP_AUDIO = "temp_clean_audio.wav"

def set_status(msg):
    with open("progress.txt", "w", encoding="utf-8") as f:
        f.write(msg)
    print(msg) # Vẫn in ra terminal để dự phòng

def get_system_config():
    sys_os = platform.system()
    machine = platform.machine()
    
    if sys_os == "Darwin" and machine == "arm64":
        return "mac_silicon", "h264_videotoolbox"
    elif torch.cuda.is_available():
        return "cuda", "h264_nvenc" 
    else:
        return "cpu", "libx264"

def filter_segments_with_llm(reference_script, transcript_text, model_name="gemma4:e4b"):
    prompt = f"""
    Bạn là một trợ lý chỉnh sửa video chuyên nghiệp.
    Nhiệm vụ của bạn là đối chiếu 'Phụ đề Video thực tế' với 'Kịch bản Chuẩn'.
    Người diễn thuyết có thể nói vấp, nói lại nhiều lần một đoạn. Hãy chọn ra những đoạn nói (takes) tốt nhất, trôi chảy nhất, không bị vấp, và sát với Kịch bản Chuẩn nhất để ghép thành một video hoàn chỉnh.

    KỊCH BẢN CHUẨN:
    {reference_script}

    PHỤ ĐỀ VIDEO THỰC TẾ (Định dạng: [start - end | loudness] text):
    {transcript_text}

    YÊU CẦU ĐẦU RA:
    1. CHỈ trả về danh sách các đoạn cần giữ lại.
    2. GIỮ NGUYÊN định dạng timestamp [start - end] ở đầu mỗi dòng.
    3. KHÔNG giải thích, KHÔNG thêm văn bản phụ trợ, KHÔNG dùng markdown code block.
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        raise Exception(f"Lỗi khi kết nối với Ollama: {e}")

def concat_multiple_videos(uploaded_files, output_path):
    # Sort files by name (uploaded_files là list các object FakeFile từ main.py)
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    temp_filenames = []
    list_file_path = "concat_list.txt"
    
    with open(list_file_path, "w", encoding="utf-8") as list_file:
        for file in uploaded_files:
            # File đã được lưu sẵn trong thư mục temp_uploads bởi FastAPI
            temp_filenames.append(file.path)
            list_file.write(f"file '{file.path}'\n")
            
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file_path, "-c", "copy", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(f"Lỗi khi ghép video bằng FFmpeg: {e}")
        
    if os.path.exists(list_file_path): os.remove(list_file_path)
    return output_path

def preprocess_audio_for_whisper(video_path):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-af", "highpass=f=200,afftdn=nf=-25,loudnorm=I=-16:TP=-1.5:LRA=11",
        TEMP_AUDIO
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return TEMP_AUDIO
    except subprocess.CalledProcessError:
        return video_path 

def split_sentences(text):
    text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def check_repeating_words(segments):
    for seg in segments:
        clean_text = re.sub(r'[^\w\s]', '', seg['text'].lower())
        words = clean_text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True 
    return False

def check_identical_segments(segments, reference_script="", similarity_threshold=0.75):
    if len(segments) < 2: return False
        
    ref_sentences = []
    if reference_script.strip():
        raw_sentences = split_sentences(reference_script)
        ref_sentences = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in raw_sentences]

    for i in range(1, len(segments)):
        prev_text = re.sub(r'[^\w\s]', '', segments[i-1]['text'].lower()).strip()
        curr_text = re.sub(r'[^\w\s]', '', segments[i]['text'].lower()).strip()
        
        if len(curr_text.split()) < 2: continue
            
        if curr_text == prev_text:
            is_in_script = False
            if ref_sentences:
                for ref_sent in ref_sentences:
                    score = difflib.SequenceMatcher(None, curr_text, ref_sent).ratio()
                    if score >= similarity_threshold:
                        is_in_script = True
                        break
            if not is_in_script:
                return True
    return False

def transcribe_audio(video_path, reference_script=""):
    device, _ = get_system_config()
    set_status(f"Bắt đầu xử lý âm thanh... Đang khởi tạo mô hình trên {device.upper()}")
    processed_audio = preprocess_audio_for_whisper(video_path)
    prompt_moi = "Đây là video tiếng Việt. Từ vựng tham khảo: " + reference_script[:500]
    
    max_retries = 3
    final_cleaned_segments = []
    
    for attempt in range(max_retries):
        if attempt > 0: print(f"Phát hiện ảo giác. Đang bóc băng lại lần {attempt + 1}...")
        current_temp = 0.0 if attempt == 0 else 0.2 + (attempt * 0.2)
        
        if device == "mac_silicon":
            try:
                import mlx_whisper
            except ImportError:
                raise Exception("Chưa cài đặt mlx-whisper!")
            result = mlx_whisper.transcribe(
                processed_audio, path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language="vi", word_timestamps=True, temperature=current_temp, initial_prompt=prompt_moi
            )
        else:
            model = whisper.load_model("large", device=device)
            use_fp16 = True if device == "cuda" else False
            result = model.transcribe(
                processed_audio, fp16=use_fp16, language="vi", 
                word_timestamps=True, temperature=current_temp, initial_prompt=prompt_moi
            )
        
        raw_segments = result['segments']
        
        try:
            audio_full = AudioSegment.from_file(video_path)
            for seg in raw_segments:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                audio_chunk = audio_full[start_ms:end_ms]
                loudness = audio_chunk.dBFS
                seg['loudness_dBFS'] = loudness if loudness != float('-inf') else -100.0
        except Exception:
            for seg in raw_segments: seg['loudness_dBFS'] = 0.0

        cleaned_segments = [seg for seg in raw_segments if seg.get('loudness_dBFS', 0.0) != -100.0 and (seg['end'] - seg['start']) >= 1.0]
        final_cleaned_segments = cleaned_segments
        
        has_repeating_words = check_repeating_words(final_cleaned_segments)
        has_identical_segments = check_identical_segments(final_cleaned_segments, reference_script)
        
        if not has_repeating_words and not has_identical_segments:
            break 

    if os.path.exists(TEMP_AUDIO): os.remove(TEMP_AUDIO)
    is_hallucinating = has_repeating_words or has_identical_segments
    
    return final_cleaned_segments, is_hallucinating

def format_segments_to_text(segments):
    lines = []
    for seg in segments:
        loudness = seg.get('loudness_dBFS', 0.0)
        lines.append(f"[{seg['start']:.2f} - {seg['end']:.2f} | {loudness:.2f} dBFS] {seg['text'].strip()}")
    return "\n".join(lines)