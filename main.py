from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil

from core_logic import (
    concat_multiple_videos, 
    transcribe_audio, 
    filter_segments_with_llm,
    format_segments_to_text,
    set_status # Import thêm hàm trạng thái
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TEMP_DIR = "temp_uploads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/")
async def serve_frontend():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>Không tìm thấy file index.html</h1>", status_code=404)

# API TRẢ VỀ TRẠNG THÁI HIỆN TẠI
@app.get("/api/status")
def api_status():
    try:
        with open("progress.txt", "r", encoding="utf-8") as f:
            return {"message": f.read()}
    except:
        return {"message": "Hệ thống đang chuẩn bị..."}

# XÓA TỪ KHÓA 'async' Ở ĐÂY ĐỂ TRÁNH NGHẼN SERVER
@app.post("/api/transcribe")
def api_transcribe(
    files: List[UploadFile] = File(...),
    reference_script: str = Form(...)
):
    try:
        set_status("Đang tiếp nhận file video từ trình duyệt...")
        saved_files = []
        for file in files:
            file_path = os.path.join(TEMP_DIR, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            class FakeFile:
                def __init__(self, name, path):
                    self.name = name
                    self.path = path
                def getbuffer(self):
                    with open(self.path, "rb") as f: return f.read()
            saved_files.append(FakeFile(file.filename, file_path))

        set_status("Đang nối các file video lại với nhau...")
        combined_video_path = os.path.join(TEMP_DIR, "temp_input.mp4")
        concat_path = concat_multiple_videos(saved_files, combined_video_path)
        
        segments, has_hallucination = transcribe_audio(concat_path, reference_script)
        
        set_status("Hoàn tất bóc băng! Đang chuẩn bị dữ liệu trả về...")
        raw_text = format_segments_to_text(segments)

        # LỌC SẠCH LỖI NaN BẰNG CÁCH CHỈ LẤY CÁC TRƯỜNG CẦN THIẾT
        clean_segments = []
        for seg in segments:
            clean_segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", ""),
                "loudness_dBFS": seg.get("loudness_dBFS", 0.0)
            })

        return {
            "status": "success",
            "segments": clean_segments, # Trả về mảng đã được làm sạch
            "raw_text": raw_text,
            "has_hallucination": has_hallucination
        }
    except Exception as e:
        set_status("Lỗi hệ thống!")
        raise HTTPException(status_code=500, detail=str(e))

# XÓA TỪ KHÓA 'async' Ở ĐÂY
@app.post("/api/filter")
def api_filter(
    reference_script: str = Form(...),
    edited_transcript: str = Form(...)
):
    try:
        set_status("Đang gửi dữ liệu cho AI (Ollama) phân tích. Vui lòng đợi...")
        final_script = filter_segments_with_llm(
            reference_script=reference_script,
            transcript_text=edited_transcript,
            model_name="gemma4:e4b"
        )
        set_status("AI đã hoàn tất việc lọc kịch bản!")
        return {"status": "success", "filtered_script": final_script}
    except Exception as e:
        set_status("Lỗi khi gọi AI!")
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn
if __name__ == "__main__":
    print("🚀 Đang khởi động AI Video Auto Cutter Server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)