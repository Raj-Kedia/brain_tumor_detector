# Base image
FROM python:3.11-slim

# Create working directory
WORKDIR /app

# Copy all project files
COPY . .

# Install bash (necessary if your script uses bash features)
RUN apt-get update && apt-get install -y bash

# Make the build script executable
RUN chmod +x ./build.sh

# Run the script
RUN ./build.sh

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run migrations and collect static files
RUN python manage.py migrate && python manage.py collectstatic --no-input

CMD ["gunicorn", "brain_tumor_detector.wsgi:application"]
