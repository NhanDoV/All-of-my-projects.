FROM python:3.11-slim

WORKDIR /app

# Cài đặt một số package hệ thống cần thiết (tránh lỗi khi build một số thư viện)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để tận dụng cache
COPY requirements.txt .

# Cài dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ source code + ảnh
COPY . .

# Streamlit mặc định chạy ở port 8501
EXPOSE 8501

# Chạy Streamlit
CMD ["streamlit", "run", "launch.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]