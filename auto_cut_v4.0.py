import streamlit as st
import whisper
import torch
import subprocess
import os
import re
import platform
import warnings
import difflib
import time  
from pydub import AudioSegment
import requests
import json

warnings.filterwarnings("ignore", category=UserWarning)

def filter_segments_with_llm(reference_script, transcript_text, model_name="gemma4:e4b"):
    """
    Gửi prompt cho LLM để phân tích và chọn lọc các câu nói tốt nhất.
    """
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
    
    Kết quả mong đợi:
    [0.00 - 2.50] Nội dung câu nói...
    [5.00 - 8.20] Nội dung câu nói tiếp theo...
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1 # Giữ temperature thấp để LLM không tự sáng tạo thêm từ
        }
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except Exception as e:
        return f"Lỗi khi kết nối với Ollama: {e}\n(Hãy đảm bảo Ollama đang chạy và đã tải model {model_name})"

# --- CẤU HÌNH CƠ BẢN ---
OUTPUT_VIDEO = "output_video.mp4"
TEMP_AUDIO = "temp_clean_audio.wav"
COMBINED_VIDEO = "temp_input.mp4"

# --- TỐI ƯU HÓA ĐA NỀN TẢNG ---
def get_system_config():
    sys_os = platform.system()
    machine = platform.machine()
    
    if sys_os == "Darwin" and machine == "arm64":
        return "mac_silicon", "h264_videotoolbox"
    elif torch.cuda.is_available():
        return "cuda", "h264_nvenc" 
    else:
        return "cpu", "libx264"

# --- CÁC HÀM XỬ LÝ LÕI ---

def concat_multiple_videos(uploaded_files, output_path):
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    temp_filenames = []
    list_file_path = "concat_list.txt"
    
    with open(list_file_path, "w", encoding="utf-8") as list_file:
        for i, file in enumerate(uploaded_files):
            ext = os.path.splitext(file.name)[1]
            temp_name = f"temp_part_{i}{ext}"
            with open(temp_name, "wb") as temp_f:
                temp_f.write(file.getbuffer())
            temp_filenames.append(temp_name)
            list_file.write(f"file '{temp_name}'\n")
            
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file_path, "-c", "copy", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        st.error(f"Lỗi khi ghép video. Vui lòng đảm bảo các video có cùng định dạng và độ phân giải. Lỗi: {e}")
        st.stop()
        
    for temp_name in temp_filenames:
        if os.path.exists(temp_name): os.remove(temp_name)
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

def check_repeating_words(segments):
    """Kiểm tra xem có từ nào bị lặp lại quá 2 lần (vd: han han han) trong 1 segment không"""
    for seg in segments:
        clean_text = re.sub(r'[^\w\s]', '', seg['text'].lower())
        words = clean_text.split()
        for i in range(len(words) - 2):
            if words[i] == words[i+1] == words[i+2]:
                return True 
    return False

def check_identical_segments(segments, reference_script="", similarity_threshold=0.75):
    """
    Kiểm tra ảo giác dựa trên 2 điều kiện:
    1. Hai câu liền kề nhau có nội dung giống hệt nhau.
    2. Nội dung đó KHÔNG nằm trong (hoặc không gần giống) kịch bản chuẩn.
    """
    if len(segments) < 2:
        return False
        
    # Tiền xử lý kịch bản chuẩn: Tách thành các câu và làm sạch (bỏ dấu, in thường)
    ref_sentences = []
    if reference_script.strip():
        raw_sentences = split_sentences(reference_script)
        ref_sentences = [re.sub(r'[^\w\s]', '', s.lower()).strip() for s in raw_sentences]

    # Duyệt từ segment thứ 2 để so sánh với segment ngay trước nó (liền kề)
    for i in range(1, len(segments)):
        prev_text = re.sub(r'[^\w\s]', '', segments[i-1]['text'].lower()).strip()
        curr_text = re.sub(r'[^\w\s]', '', segments[i]['text'].lower()).strip()
        
        # Bỏ qua các segment quá ngắn (dưới 2 từ)
        if len(curr_text.split()) < 2:
            continue
            
        # ĐIỀU KIỆN 2: Hai câu nằm liền kề nhau có nội dung giống nhau
        if curr_text == prev_text:
            is_in_script = False
            
            # ĐIỀU KIỆN 1: Kiểm tra nội dung này có trong kịch bản chuẩn không
            if ref_sentences:
                for ref_sent in ref_sentences:
                    # Tính độ tương đồng giữa segment và câu trong kịch bản
                    score = difflib.SequenceMatcher(None, curr_text, ref_sent).ratio()
                    # Nếu độ giống nhau vượt ngưỡng threshold (VD: 75%) -> Bỏ qua
                    if score >= similarity_threshold:
                        is_in_script = True
                        break
            
            # Nếu 2 câu liền kề giống nhau mà KHÔNG có trong kịch bản -> Ảo giác
            if not is_in_script:
                return True
                
    return False

def transcribe_audio(video_path, reference_script=""):
    device, _ = get_system_config()
    st.info("🎵 Đang làm sạch và chuẩn hóa âm thanh...")
    processed_audio = preprocess_audio_for_whisper(video_path)

    # Tạo prompt mồi từ kịch bản chuẩn (lấy khoảng 500 ký tự đầu cho nhẹ)
    prompt_moi = "Đây là video tiếng Việt. Từ vựng tham khảo: " + reference_script[:500]
    
    max_retries = 3
    final_cleaned_segments = []
    
    for attempt in range(max_retries):
        if attempt > 0:
            st.warning(f"⚠️ Phát hiện Whisper bị ảo giác (lặp từ hoặc lặp câu). Đang tự động bóc băng lại lần {attempt + 1}/{max_retries}...")
        
        current_temp = 0.0 if attempt == 0 else 0.2 + (attempt * 0.2)
        
        # --- 1. BÓC BĂNG ---
        if device == "mac_silicon":
            if attempt == 0: st.info("🚀 Đang bóc băng bằng **MLX-Whisper**...")
            try:
                import mlx_whisper
            except ImportError:
                st.error("❌ Chưa cài đặt thư viện MLX! Hãy chạy: pip install mlx-whisper")
                st.stop()
                
            result = mlx_whisper.transcribe(
                processed_audio,
                path_or_hf_repo="mlx-community/whisper-large-v3-turbo",
                language="vi",
                word_timestamps=True,
                temperature=current_temp,
                initial_prompt=prompt_moi
            )
        else:
            if attempt == 0: st.info(f"🚀 Đang bóc băng bằng **OpenAI-Whisper** trên hệ thống {device.upper()}...")
            model = whisper.load_model("large", device=device)
            use_fp16 = True if device == "cuda" else False
            result = model.transcribe(
                processed_audio, 
                fp16=use_fp16, 
                language="vi", 
                word_timestamps=True,
                temperature=current_temp,
                initial_prompt=prompt_moi
            )
        
        raw_segments = result['segments']
        
        # --- 2. ĐO CƯỜNG ĐỘ ÂM THANH ---
        if attempt == 0: st.info("📊 Đang phân tích cường độ âm thanh cho từng đoạn...")
        try:
            audio_full = AudioSegment.from_file(video_path)
            for seg in raw_segments:
                start_ms = int(seg['start'] * 1000)
                end_ms = int(seg['end'] * 1000)
                audio_chunk = audio_full[start_ms:end_ms]
                loudness = audio_chunk.dBFS
                seg['loudness_dBFS'] = loudness if loudness != float('-inf') else -100.0
        except Exception as e:
            for seg in raw_segments:
                seg['loudness_dBFS'] = 0.0

        # --- 3. LÀM SẠCH DỮ LIỆU ---
        cleaned_segments = []
        for seg in raw_segments:
            duration = seg['end'] - seg['start']
            if seg.get('loudness_dBFS', 0.0) == -100.0:
                continue
            if duration < 1.0:
                continue
            cleaned_segments.append(seg)
            
        final_cleaned_segments = cleaned_segments
        
        # --- 4. KIỂM TRA ẢO GIÁC (LẶP TỪ & LẶP CÂU) ---
        has_repeating_words = check_repeating_words(final_cleaned_segments)
        has_identical_segments = check_identical_segments(final_cleaned_segments, reference_script)
        
        if not has_repeating_words and not has_identical_segments:
            break 
        elif attempt == max_retries - 1:
            st.error("⚠️ Đã bóc băng lại 3 lần nhưng dữ liệu vẫn còn hiện tượng ảo giác. Giữ nguyên kết quả tốt nhất hiện tại.")

    if os.path.exists(TEMP_AUDIO):
        os.remove(TEMP_AUDIO)
        
    # Gom 2 trạng thái lỗi lại. Nếu 1 trong 2 là True -> có ảo giác
    is_hallucinating = has_repeating_words or has_identical_segments
    
    return final_cleaned_segments, is_hallucinating

def format_segments_to_text(segments):
    lines = []
    for seg in segments:
        loudness = seg.get('loudness_dBFS', 0.0)
        lines.append(f"[{seg['start']:.2f} - {seg['end']:.2f} | {loudness:.2f} dBFS] {seg['text'].strip()}")
    return "\n".join(lines)

def parse_text_to_segments(text):
    segments = []
    for line in text.strip().split('\n'):
        match = re.match(r"\[([\d\.]+) - ([\d\.]+)(?:.*?)\] (.*)", line)
        if match:
            segments.append({
                'start': float(match.group(1)),
                'end': float(match.group(2)),
                'text': match.group(3)
            })
    return segments

def split_sentences(text):
    text = re.sub(r'([.!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def merge_overlapping_segments(segments):
    if not segments: return []
    merged = [segments[0].copy()]
    for current in segments[1:]:
        previous = merged[-1]
        if current['start'] <= previous['end'] + 0.2:
            previous['end'] = max(previous['end'], current['end'])
            previous['text'] += " " + current['text'].strip()
        else:
            merged.append(current.copy())
    return merged

# --- GIAO DIỆN STREAMLIT ---

st.set_page_config(page_title="AI Video Auto Cutter", layout="wide")
st.title("✂️ AI Video Auto Cutter")

device, encoder = get_system_config()

device_display = {
    "mac_silicon": "Apple M-Series (MLX: GPU + ANE)",
    "cuda": "NVIDIA GPU (CUDA)",
    "cpu": "CPU"
}.get(device, "Unknown")

st.caption(f"⚙️ **Phần cứng phát hiện được:** `{device_display}` | Bộ mã hóa FFmpeg: `{encoder}`")

# Khởi tạo Session State
if "step" not in st.session_state: st.session_state.step = 0
if "video_path" not in st.session_state: st.session_state.video_path = None
if "segments" not in st.session_state: st.session_state.segments = []
if "good_segments" not in st.session_state: st.session_state.good_segments = []
if "match_details" not in st.session_state: st.session_state.match_details = []
if "reference_script" not in st.session_state: st.session_state.reference_script = ""
if "threshold" not in st.session_state: st.session_state.threshold = 0.5
if "processing_time" not in st.session_state: st.session_state.processing_time = 0
if "has_hallucination" not in st.session_state: st.session_state.has_hallucination = False

# --- BƯỚC 0: TẢI VIDEO & NHẬP KỊCH BẢN ---
if st.session_state.step == 0:
    st.header("1. Cung cấp Dữ liệu Đầu vào")
    
    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Chọn các file video (.mp4, .mov)", 
            type=["mp4", "mov"], 
            accept_multiple_files=True
        )
        if uploaded_files:
            st.success(f"Đã chọn {len(uploaded_files)} video. Các file sẽ được ghép theo thứ tự tên (A-Z).")
            
    with col2:
        st.session_state.reference_script = st.text_area(
            "Kịch bản Chuẩn (Reference Script)", 
            value=st.session_state.reference_script, 
            height=200, 
            placeholder="Dán kịch bản chuẩn của bạn vào đây.\nHệ thống sẽ tự động phân tách các câu nói dựa trên dấu chấm (.), chấm hỏi (?) hoặc chấm than (!)."
        )

    if uploaded_files and st.session_state.reference_script.strip():
        if st.button("Bắt đầu ghép và xử lý Video", type="primary"):
            start_time = time.time()
            
            with st.spinner("Đang chuẩn bị video (ghép file và bóc băng âm thanh). Vui lòng đợi..."):
                st.session_state.video_path = concat_multiple_videos(uploaded_files, COMBINED_VIDEO)
                st.session_state.segments, st.session_state.has_hallucination = transcribe_audio(st.session_state.video_path, st.session_state.reference_script)
                
                st.session_state.processing_time = time.time() - start_time
                st.session_state.step = 1
                st.rerun()
    elif uploaded_files:
        st.warning("Vui lòng nhập Kịch bản chuẩn trước khi bắt đầu!")

# --- BƯỚC 1: RÀ SOÁT & TIẾN HÀNH LỌC BẰNG AI ---
elif st.session_state.step == 1:
    st.header("2. Rà soát & Lọc kịch bản bằng AI (Gemma)")
    
    mins = int(st.session_state.processing_time // 60)
    secs = int(st.session_state.processing_time % 60)
    st.success(f"⏱️ **Quá trình Tiền xử lý (Ghép File + Bóc Băng + Làm Sạch) hoàn tất trong: {mins} phút {secs} giây**")
    
    if st.session_state.has_hallucination:
        st.warning("⚠️ **Lưu ý:** Dữ liệu vẫn còn dấu hiệu lặp từ/lặp câu do ảo giác của Whisper. Hãy kiểm tra phụ đề bên dưới.")
    else:
        st.success("✅ **Dữ liệu sạch:** Không phát hiện hiện tượng ảo giác ở lần bóc băng cuối cùng.")

    col1, col2 = st.columns(2)
    with col1:
        raw_text = format_segments_to_text(st.session_state.segments)
        edited_text = st.text_area("Phụ đề từ Video (Raw Transcript)", value=raw_text, height=350)
        
        if st.button("🔄 Trích xuất lại phụ đề"):
            start_time = time.time()
            with st.spinner("Đang trích xuất lại..."):
                st.session_state.segments, st.session_state.has_hallucination = transcribe_audio(st.session_state.video_path, st.session_state.reference_script)
                st.session_state.processing_time = time.time() - start_time
                st.rerun()
    
    with col2:
        st.session_state.reference_script = st.text_area("Kịch bản Chuẩn (Reference Script)", value=st.session_state.reference_script, height=350)

    st.divider()
    
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⬅️ Làm lại từ đầu"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    with col_btn2:
        if st.button("🧠 Yêu cầu Gemma lọc các đoạn tốt nhất ➡️", type="primary"):
            st.session_state.edited_transcript = edited_text
            st.session_state.step = 2
            st.rerun()

# --- BƯỚC 2: KẾT QUẢ TỪ LLM ---
elif st.session_state.step == 2:
    st.header("3. Kết quả lọc kịch bản từ AI")
    
    with st.spinner("Đang gửi dữ liệu cho mô hình gemma4:e4b phân tích. Vui lòng đợi..."):
        # Gọi hàm LLM
        final_script = filter_segments_with_llm(
            reference_script=st.session_state.reference_script,
            transcript_text=st.session_state.edited_transcript,
            model_name="gemma4:e4b"
        )
    
    if "Lỗi khi kết nối" in final_script:
        st.error(final_script)
        if st.button("⬅️ Quay lại"):
            st.session_state.step = 1
            st.rerun()
    else:
        st.success("🎉 AI đã hoàn tất việc lọc các đoạn (takes) tốt nhất!")
        st.text_area("Kịch bản & Timestamp đã lọc", value=final_script, height=400)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Quay lại điều chỉnh"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            st.download_button(
                label="💾 Tải Script (.txt)",
                data=final_script,
                file_name="filtered_script_with_timestamps.txt",
                mime="text/plain"
            )