# WT-PM command center + SCADA bridge
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8100 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=1
COPY pyproject.toml README.md ARCHITECTURE.md ./
COPY src ./src
COPY platform ./platform
RUN pip install --no-cache-dir . ./platform \
    && useradd --create-home --uid 10001 wtpm \
    && chown wtpm:wtpm /app
USER wtpm
EXPOSE 8100
ENV WTPM_CONFIG=/app/wt-pm.yaml
CMD ["wt-pm", "serve", "--days", "6"]
