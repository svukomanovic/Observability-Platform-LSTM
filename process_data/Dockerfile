# Dockerfile

FROM python:3.9-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY process_server.py .
COPY index.html .

# MODIFICATION: Create the directory AND set permissions
# The python base image may run as user 1000. This makes the dir writable.
RUN mkdir -p /state && chown -R 1000:1000 /state

EXPOSE 5000
CMD ["python", "process_server.py"]