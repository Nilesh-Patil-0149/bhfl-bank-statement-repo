FROM python:3.12-slim
WORKDIR /app
COPY . /app/
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libpq-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENTRYPOINT ["python", "bhfl_stmt.py"]
