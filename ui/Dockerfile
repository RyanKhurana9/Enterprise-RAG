# Thin Streamlit client: needs only streamlit, pandas and httpx (no torch / DB drivers).
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && pip install "streamlit==1.67.0" "pandas>=2.2" "httpx>=0.27"
WORKDIR /app
COPY ui/streamlit_app.py ./streamlit_app.py
RUN useradd --create-home app
USER app
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
