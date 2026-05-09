import streamlit as st
import whisper
import torch
import subprocess
import os
import re
import platform
import warnings
import difflib
from pydub import AudioSegment

warnings.filterwarnings("ignore", category=UserWarning)

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
    """Sắp xếp các file theo tên và dùng FFmpeg nối lại thành 1 file duy nhất"""
    # Sắp xếp file theo tên (A-Z)
    uploaded_files = sorted(uploaded_files, key=lambda x: x.name)
    
    temp_filenames = []
    list_file_path = "concat_list.txt"
    
    # Lưu tạm các file đã upload ra ổ cứng và tạo file danh sách cho FFmpeg
    with open(list_file_path, "w", encoding="utf-8") as list_file:
        for i, file in enumerate(uploaded_files):
            # Lấy đuôi file gốc (vd: .mp4, .mov)
            ext = os.path.splitext(file.name)[1]
            temp_name = f"temp_part_{i}{ext}"
            
            with open(temp_name, "wb") as temp_f:
                temp_f.write(file.getbuffer())
                
            temp_filenames.append(temp_name)
            # Ghi vào danh sách theo cú pháp FFmpeg yêu cầu
            list_file.write(f"file '{temp_name}'\n")
            
    # Gọi FFmpeg ghép file không re-encode (-c copy) cho tốc độ siêu nhanh
    command = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", 
        "-i", list_file_path, "-c", "copy", output_path
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        st.error(f"Lỗi khi ghép video. Vui lòng đảm bảo các video có cùng định dạng và độ phân giải. Lỗi: {e}")
        st.stop()
        
    # Dọn dẹp rác
    for temp_name in temp_filenames:
        if os.path.exists(temp_name):
            os.remove(temp_name)
    if os.path.exists(list_file_path):
        os.remove(list_file_path)
        
    return output_path

def preprocess_audio_for_whisper(video_path):
    command = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000",
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        TEMP_AUDIO
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return TEMP_AUDIO
    except subprocess.CalledProcessError:
        return video_path 

def transcribe_audio(video_path):
    device, _ = get_system_config()
    st.info("🎵 Đang làm sạch và chuẩn hóa âm thanh...")
    processed_audio = preprocess_audio_for_whisper(video_path)
    
    if device == "mac_silicon":
        st.info("🚀 Đang bóc băng bằng **MLX-Whisper** (Tối ưu siêu tốc cho Apple Silicon GPU + ANE)...")
        try:
            import mlx_whisper
        except ImportError:
            st.error("❌ Chưa cài đặt thư viện MLX! Hãy mở Terminal và chạy lệnh: `pip install mlx-whisper`")
            st.stop()
            
        result = mlx_whisper.transcribe(
            processed_audio,
            path_or_hf_repo="mlx-community/whisper-large-v3-mlx",
            language="vi",
            word_timestamps=True,
            temperature=0
        )
    else:
        st.info(f"🚀 Đang bóc băng bằng **OpenAI-Whisper** trên hệ thống {device.upper()}...")
        model = whisper.load_model("large", device=device)
        use_fp16 = True if device == "cuda" else False
        result = model.transcribe(
            processed_audio, 
            fp16=use_fp16, 
            language="vi", 
            word_timestamps=True 
        )
    
    segments = result['segments']
    
    st.info("📊 Đang phân tích cường độ âm thanh cho từng đoạn...")
    try:
        audio_full = AudioSegment.from_file(video_path)
        for seg in segments:
            start_ms = int(seg['start'] * 1000)
            end_ms = int(seg['end'] * 1000)
            audio_chunk = audio_full[start_ms:end_ms]
            
            loudness = audio_chunk.dBFS
            seg['loudness_dBFS'] = loudness if loudness != float('-inf') else -100.0
    except Exception as e:
        st.warning(f"Lỗi khi dùng Pydub phân tích âm thanh: {e}")
        for seg in segments:
            seg['loudness_dBFS'] = 0.0

    if os.path.exists(TEMP_AUDIO):
        os.remove(TEMP_AUDIO)
    return segments

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
    if not segments:
        return []
    
    merged = [segments[0].copy()]
    for current in segments[1:]:
        previous = merged[-1]
        if current['start'] <= previous['end'] + 0.2:
            previous['end'] = max(previous['end'], current['end'])
            previous['text'] += " " + current['text'].strip()
        else:
            merged.append(current.copy())
    return merged

def match_segments_with_script(reference_script, segments, threshold=0.5, max_window=8):
    ref_lines = split_sentences(reference_script)
    selected_indices = set()
    match_details = []

    for ref_line in ref_lines:
        best_score = 0
        best_block = None
        
        for i in range(len(segments)):
            accumulated_text = ""
            for j in range(i, min(i + max_window, len(segments))):
                accumulated_text += " " + segments[j]['text'].strip()
                accumulated_text = accumulated_text.strip()
                
                score = difflib.SequenceMatcher(None, ref_line.lower(), accumulated_text.lower()).ratio()
                
                if score >= best_score and score >= threshold:
                    best_score = score
                    best_block = {
                        "start_idx": i,
                        "end_idx": j,
                        "text": accumulated_text,
                        "score": score
                    }
        
        if best_block is not None:
            for idx in range(best_block["start_idx"], best_block["end_idx"] + 1):
                selected_indices.add(idx)
                
            start_time = segments[best_block["start_idx"]]['start']
            end_time = segments[best_block["end_idx"]]['end']
            
            match_details.append({
                "ref_line": ref_line,
                "matched_text": best_block["text"],
                "score": best_block["score"],
                "start": start_time,
                "end": end_time
            })

    raw_final_segments = [segments[i] for i in sorted(list(selected_indices))]
    final_segments = merge_overlapping_segments(raw_final_segments)
    
    return final_segments, match_details

def build_ffmpeg_concat_file(good_segments):
    filter_script = ""
    for i, seg in enumerate(good_segments):
        start = seg['start']
        end = seg['end']
        filter_script += f"[0:v]trim=start={start}:end={end},setpts=PTS-STARTPTS[v{i}]; "
        filter_script += f"[0:a]atrim=start={start}:end={end},asetpts=PTS-STARTPTS[a{i}]; "
    
    concat_str = "".join([f"[v{i}][a{i}]" for i in range(len(good_segments))])
    filter_script += f"{concat_str}concat=n={len(good_segments)}:v=1:a=1[outv][outa]"
    return filter_script


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

# --- BƯỚC 0: TẢI VIDEO & NHẬP KỊCH BẢN ---
if st.session_state.step == 0:
    st.header("1. Cung cấp Dữ liệu Đầu vào")
    
    col1, col2 = st.columns(2)
    with col1:
        # Cập nhật: Cho phép upload nhiều file
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
            with st.spinner("Đang chuẩn bị video (ghép file và bóc băng âm thanh). Vui lòng đợi..."):
                # Gọi hàm ghép video và lưu đường dẫn vào session state
                st.session_state.video_path = concat_multiple_videos(uploaded_files, COMBINED_VIDEO)
                st.session_state.segments = transcribe_audio(st.session_state.video_path)
                st.session_state.step = 1
                st.rerun()
    elif uploaded_files:
        st.warning("Vui lòng nhập Kịch bản chuẩn trước khi bắt đầu!")

# --- BƯỚC 1: ĐIỀU CHỈNH KỊCH BẢN & ĐỘ NHẠY ---
elif st.session_state.step == 1:
    st.header("2. Rà soát & Đối chiếu kịch bản")
    
    col1, col2 = st.columns(2)
    with col1:
        raw_text = format_segments_to_text(st.session_state.segments)
        edited_text = st.text_area("Phụ đề từ Video đã ghép (Có thể sửa lỗi chính tả Whisper)", value=raw_text, height=350)
        
        if st.button("🔄 Trích xuất lại phụ đề (Không cần upload lại)"):
            with st.spinner("Đang trích xuất lại từ file video đã lưu..."):
                st.session_state.segments = transcribe_audio(st.session_state.video_path)
                st.rerun()
    
    with col2:
        st.session_state.reference_script = st.text_area("Kịch bản Chuẩn (Có thể sửa lại nếu cần)", value=st.session_state.reference_script, height=280)
        st.info("Hệ thống tự động nối các đoạn nhỏ lại với nhau để tối đa hóa độ trùng khớp.")
        st.session_state.threshold = st.slider("Độ chính xác tối thiểu (Threshold)", 0.1, 1.0, st.session_state.threshold, 0.05)

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("⬅️ Làm lại từ đầu (Xóa hết Video & Kịch bản)"):
            for key in list(st.session_state.keys()): del st.session_state[key]
            st.rerun()
    with col_btn2:
        if st.button("Tiến hành Đối chiếu & Xem kết quả ➡️", type="primary"):
            st.session_state.segments = parse_text_to_segments(edited_text)
            final_segs, details = match_segments_with_script(st.session_state.reference_script, st.session_state.segments, st.session_state.threshold)
            st.session_state.good_segments = final_segs
            st.session_state.match_details = details
            st.session_state.step = 2
            st.rerun()

# --- BƯỚC 2: XÁC NHẬN & RENDER ---
elif st.session_state.step == 2:
    st.header("3. Kết quả đối chiếu")
    
    if len(st.session_state.good_segments) == 0:
        st.warning("Không tìm thấy đoạn nào khớp. Hãy quay lại hạ độ nhạy hoặc kiểm tra kịch bản có đủ dấu chấm câu chưa.")
        if st.button("⬅️ Quay lại điều chỉnh"):
            st.session_state.step = 1
            st.rerun()
    else:
        st.success(f"Tìm thấy tổ hợp hoàn chỉnh. Đã gộp thành {len(st.session_state.good_segments)} cảnh quay lớn.")
        with st.expander("Xem chi tiết các đoạn sẽ được giữ lại", expanded=True):
            for detail in st.session_state.match_details:
                st.write(f"**Gốc:** {detail['ref_line']} \n\n **→ Tổ hợp Video:** `[{detail['start']:.2f} - {detail['end']:.2f}]` {detail['matched_text']} *(Độ khớp: {detail['score']:.2f})*")
                st.divider()
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("⬅️ Quay lại điều chỉnh (Sửa phụ đề / Hạ độ nhạy)"):
                st.session_state.step = 1
                st.rerun()
        with col2:
            if st.button("🎬 Bắt đầu Cắt Video", type="primary"):
                with st.spinner("Đang Render. Quá trình này có thể mất vài phút..."):
                    filter_complex = build_ffmpeg_concat_file(st.session_state.good_segments)
                    subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", st.session_state.video_path, 
                            "-filter_complex", filter_complex, 
                            "-map", "[outv]", "-map", "[outa]", 
                            "-c:v", encoder, 
                            "-pix_fmt", "yuv420p", 
                            "-c:a", "aac", OUTPUT_VIDEO
                        ], 
                        check=True
                    )
                    st.session_state.step = 3
                    st.rerun()

# --- BƯỚC 3: HOÀN TẤT & ĐIỀU CHỈNH LẠI ---
elif st.session_state.step == 3:
    st.header("4. Hoàn tất!")
    if os.path.exists(OUTPUT_VIDEO):
        st.video(OUTPUT_VIDEO)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            with open(OUTPUT_VIDEO, "rb") as file:
                st.download_button("💾 Tải Video Xuất Ra", data=file, file_name="edited_video.mp4", mime="video/mp4")
        with col2:
            if st.button("🔄 Quay lại điều chỉnh mức độ cắt (Không mất dữ liệu)"):
                st.session_state.step = 1
                st.session_state.good_segments = []
                st.session_state.match_details = []
                st.rerun()
        with col3:
            if st.button("🗑️ Bắt đầu Dự án Mới"):
                for key in list(st.session_state.keys()): del st.session_state[key]
                st.rerun()