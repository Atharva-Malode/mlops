# Use the official Ultralytics CPU image
FROM ultralytics/ultralytics:latest-cpu

# Ensure stdout/stderr are unbuffered
ENV PYTHONUNBUFFERED=1

# Set working directory inside the container
WORKDIR /app

# Copy only extra requirements (if any)
COPY requirements.txt .
# If you don't have extra deps, you can skip these two lines
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and any weights/configs
COPY app ./app

# Expose FastAPI port
EXPOSE 8000

# Launch the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
