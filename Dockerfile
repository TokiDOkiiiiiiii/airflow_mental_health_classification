FROM apache/airflow:2.9.0-python3.11

USER root

# Install CUDA runtime libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg2 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copy requirements file
COPY requirements.txt .

# Install PyTorch (kept separate due to the custom index URL)
RUN pip install --no-cache-dir \
    torch==2.2.2+cu121 \
    torchvision==0.17.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Install standard ML deps from file
RUN pip install --no-cache-dir -r requirements.txt
