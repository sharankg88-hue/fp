FROM python:3.10

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg

# Create App Directory
WORKDIR /app

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for render
EXPOSE 10000

# Start Gunicorn
CMD ["gunicorn", "-b", "0.0.0.0:10000", "app:app"]
