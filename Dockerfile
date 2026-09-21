# WT-PM command center + SCADA bridge
FROM python:3.11-slim
WORKDIR /app
COPY pyproject.toml README.md ARCHITECTURE.md ./
COPY src ./src
COPY platform ./platform
RUN pip install --no-cache-dir -e . -e platform
EXPOSE 8100
ENV WTPM_CONFIG=/app/wt-pm.yaml
CMD ["wt-pm", "serve", "--port", "8100", "--days", "6"]
