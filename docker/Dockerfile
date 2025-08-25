FROM python:3.10-slim

# 安裝系統依賴
RUN apt-get update && \
    apt-get install -y --no-install-recommends cron jq supervisor && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 安裝 Python 依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案程式碼
COPY . .

# 讓 shell script 可執行
RUN chmod +x utils/*.sh

# 複製 supervisord 配置
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# 暴露 Dash 與 FastAPI Port
EXPOSE 8050 8000

# 使用 supervisord 啟動所有服務
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
