FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application
COPY src/ ./src/
COPY run.sh .
COPY .env .

# Create data directory for state persistence
RUN mkdir -p /app/data

# Set environment
ENV PYTHONPATH=/app
ENV LONGRUN_STATE_FILE=/app/data/longrun_state.json
# Memory optimization for Raspberry Pi
ENV PYTHONMALLOC=malloc
ENV MALLOC_TRIM_THRESHOLD_=65536

# Keep container alive - ofelia scheduler handles the actual execution
CMD ["tail", "-f", "/dev/null"]
