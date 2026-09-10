FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 TORCH_HOME=/app/torch_cache OMP_NUM_THREADS=2 YITU_RUNTIME=cloud_primary
RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir flask gunicorn Pillow numpy PyYAML requests python-dotenv
COPY . /app
EXPOSE 8000
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "2", "--timeout", "180", "--access-logfile", "-", "cloud_entry:app"]
