#!/bin/bash

# 1. Chuyển hướng vào thư mục chứa script
cd "$(dirname "$0")"

echo "=========================================="
echo "Khởi động AI Video Auto Cutter (FastAPI)"
echo "Sử dụng môi trường ảo .venv"
echo "=========================================="

# 2. Kích hoạt môi trường ảo .venv (nếu tồn tại)
if [ -d ".venv" ]; then
    echo "🔄 Đang kích hoạt môi trường ảo .venv..."
    source .venv/bin/activate
else
    echo "⚠️ Không tìm thấy thư mục .venv. Đang dùng Python hệ thống..."
fi

# 3. Đảm bảo FastAPI đã được cài trong môi trường này
# (Lệnh này chạy rất nhanh nếu đã cài rồi)
pip install fastapi uvicorn python-multipart --quiet

# 4. Tự động mở trình duyệt sau 3 giây (chạy ngầm)
(sleep 3 && open http://127.0.0.1:8000) &

# 5. Khởi động server bằng lệnh python (lúc này đã là python trong .venv)
python main.py

# Giữ cửa sổ không bị tắt ngay nếu có lỗi xảy ra
echo "------------------------------------------"
echo "Server đã dừng hoạt động."
read -p "Nhấn phím Enter để thoát..."