FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for plotly/kaleido
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libnss3 \
    libgconf-2-4 \
    libfontconfig1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
COPY ../requirements.txt ./main_requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir -r main_requirements.txt

# Copy bot code
COPY bot/ ./bot/

# Copy parent src/ code (existing analysis modules)
COPY ../src/ ./src/

# Copy Schwab client config
COPY ../schwab_client.json ./schwab_client.json

# Create logs directory
RUN mkdir -p logs

# Run the bot
CMD ["python", "-m", "bot.main"]
