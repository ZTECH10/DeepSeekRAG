# Use Python 3.12 as the base image
FROM python:3.12.0

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Install dependencies
RUN pip3 install -r requirements.txt

# Expose FastAPI's default port
EXPOSE 8000

# Run FastAPI with uvicorn, listening on all interfaces (0.0.0.0) and on port 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

