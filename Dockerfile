FROM apache/spark:latest

USER root

# Upgrade pip and packaging tools
RUN pip install --upgrade pip setuptools build wheel

WORKDIR /app
COPY . /app

# Install SDV
RUN pip install .

# Install pyarrow, pyspark, and kagglehub
RUN pip install pyarrow pyspark kagglehub
