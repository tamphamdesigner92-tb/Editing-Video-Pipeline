from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import os
import shutil
import json

from core_logic import (
    concat_multiple_videos, 
    transcribe_audio, 
    deterministic_filter_pipeline,
    render_video_from_timeline,
    format_segments_to_text,
    set_status,
    get_detailed_hardware_info
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

@app.get("/api/hardware-info")
def api_hardware_info():
    try:
        info = get_detailed_hardware_info()
        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

@app.post("/api/reset-project")
def api_reset_project():
    try:
        removed_items = 0
        if os.path.exists(TEMP_DIR):
            for name in os.listdir(TEMP_DIR):
                path = os.path.join(TEMP_DIR, name)
                try:
                    if os.path.isfile(path) or os.path.islink(path):
                        os.remove(path)
                    elif os.path.isdir(path):
                        shutil.rmtree(path)
                    removed_items += 1
                except Exception:
                    continue

        set_status("Đã dọn dẹp dữ liệu tạm. Sẵn sàng cho dự án mới.")
        return {"status": "success", "removed_items": removed_items}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# XÓA TỪ KHÓA 'async' Ở ĐÂY ĐỂ TRÁNH NGHẼN SERVER
@app.post("/api/transcribe")
def api_transcribe(
    files: List[UploadFile] = File(...),
    reference_script: str = Form(...),
    transcribe_mode: str = Form("vi_smart"),
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
        
        segments, has_hallucination, transcribe_mode_used = transcribe_audio(
            concat_path,
            reference_script,
            transcribe_mode=transcribe_mode,
        )

        # [MỚI] LƯU TOÀN BỘ DATA (GỒM WORD-TIMESTAMPS) VÀO LOCAL FILE
        session_file = os.path.join(TEMP_DIR, "session_segments.json")
        with open(session_file, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False)
        
        set_status("Hoàn tất bóc băng! Đang chuẩn bị dữ liệu trả về...")
        raw_text = format_segments_to_text(segments)

        # LỌC SẠCH LỖI NaN BẰNG CÁCH CHỈ LẤY CÁC TRƯỜNG CẦN THIẾT
        clean_segments = []
        for seg in segments:
            clean_segments.append({
                "start": seg.get("start", 0.0),
                "end": seg.get("end", 0.0),
                "text": seg.get("text", ""),
                "loudness_dBFS": seg.get("loudness_dBFS", 0.0),
                "quality_decision": seg.get("quality_decision", "ACCEPT"),
                "quality_flags": seg.get("quality_flags", []),
            })

        return {
            "status": "success",
            "segments": clean_segments, # Trả về mảng đã được làm sạch
            "raw_text": raw_text,
            "has_hallucination": has_hallucination,
            "transcribe_mode_used": transcribe_mode_used,
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
        set_status("Đang chạy pipeline lọc transcript: tiền xử lý, so khớp mờ và chọn take cuối...")
        result = deterministic_filter_pipeline(
            reference_script=reference_script,
            transcript_text=edited_transcript,
        )
        set_status("Đã hoàn tất lọc kịch bản và tạo timeline EDL/XML.")
        return {
            "status": "success",
            "filtered_script": result["filtered_script"],
            "filtered_rows": result["filtered_rows"],
            "timeline_script_order": result["timeline_script_order"],
            "timeline_time_order": result["timeline_time_order"],
            "timeline": result["timeline"],  # backward compatibility
            "unmatched_sentences": result["unmatched_sentences"],
            "stats": result["stats"],
            "edl": result["edl"],
            "timeline_xml": result["timeline_xml"],
        }
    except Exception as e:
        set_status("Lỗi khi chạy pipeline lọc transcript!")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export-video")
def api_export_video(
    timeline_json: str = Form(...),
):
    try:
        set_status("Đang chuẩn bị cắt video theo timeline đã lọc...")
        source_video_path = os.path.join(TEMP_DIR, "temp_input.mp4")
        if not os.path.exists(source_video_path):
            raise HTTPException(status_code=400, detail="Không tìm thấy video nguồn. Hãy chạy bước bóc băng trước.")

        timeline = json.loads(timeline_json)
        if not isinstance(timeline, list):
            raise HTTPException(status_code=400, detail="Dữ liệu timeline không hợp lệ.")

        output_path = os.path.join(TEMP_DIR, "final_cut.mp4")
        render_video_from_timeline(
            source_video_path=source_video_path,
            timeline=timeline,
            output_path=output_path,
            temp_dir=TEMP_DIR,
        )
        set_status("Đã hoàn tất cắt dựng video.")

        return FileResponse(
            path=output_path,
            media_type="video/mp4",
            filename="final_cut.mp4",
        )
    except HTTPException:
        raise
    except Exception as e:
        set_status("Lỗi khi xuất video hoàn chỉnh!")
        raise HTTPException(status_code=500, detail=str(e))

import uvicorn
if __name__ == "__main__":
    print("🚀 Đang khởi động AI Video Auto Cutter Server...")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
