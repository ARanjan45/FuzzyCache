# Use a lightweight Python base image
FROM python:3.10-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install them
COPY requirements.txt .
# Install build dependencies required for HDBSCAN/UMAP, then clean up
RUN apt-get update && apt-get install -y build-essential \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y build-essential \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

# Copy the rest of the application code
COPY . .

# Expose port 8000
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
