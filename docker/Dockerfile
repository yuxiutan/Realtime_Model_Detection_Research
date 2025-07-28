FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y --no-install-recommends cron jq && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN chmod +x utils/*.sh

RUN echo "0 0 * * * /bin/bash /app/utils/clear_log.sh" > mycron && \
    echo "*/5 * * * * /bin/bash /app/utils/wazuh_api.sh" >> mycron && \
    crontab mycron

EXPOSE 8050

CMD ["sh", "-c", "cron && python Model/inference.py & python app.py"]
