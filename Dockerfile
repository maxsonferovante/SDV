FROM apache/spark:latest

USER root

# Upgrade pip and packaging tools
RUN pip install --upgrade pip setuptools build wheel

# Install PyTorch CPU-only first to avoid massive CUDA library downloads (>5GB)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

WORKDIR /app
COPY . /app

# Install SDV
RUN pip install .

# Install pyarrow, pyspark, and kagglehub
RUN pip install pyarrow pyspark kagglehub
