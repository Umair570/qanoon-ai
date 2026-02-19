# 1. Use a lightweight base image
FROM python:3.10-slim

# 2. Set environment variables to keep Python snappy and clean
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
# Render will provide the PORT, but we set a default just in case
ENV PORT=10000 

WORKDIR /app

# 3. Cache dependencies: Copy only requirements first
# This way, if you change your code but not your libraries, 
# Docker skips the slow 'pip install' step.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy the rest of the app
COPY . .

# 5. Expose the port (informative only)
EXPOSE 10000

# 6. Use the $PORT variable in the start command
# Render requires binding to 0.0.0.0 and the assigned $PORT
CMD ["sh", "-c", "gunicorn -w 1 -b 0.0.0.0:${PORT:-7860} --timeout 600 --graceful-timeout 300 --preload app:app"]