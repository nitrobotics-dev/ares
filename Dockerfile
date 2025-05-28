FROM python:3.10

WORKDIR /ares

# 1. Prepare prerequisites
RUN apt-get update && \
    apt-get install -y gnupg curl lsb-release

# 2. Add MongoDB’s public GPG key
RUN curl -fsSL https://pgp.mongodb.com/server-7.0.asc | \
   gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg \
   --dearmor

# 3. Add the MongoDB Database Tools repo
RUN echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" \
| tee /etc/apt/sources.list.d/mongodb-org-7.0.list

# 4. Install only the Database Tools (no mongod server)
RUN apt-get update && \
    apt-get install -y mongodb-org-tools


# Create cache directory and set environment variable
ENV TRANSFORMERS_CACHE=/cache/huggingface
RUN mkdir -p /cache/huggingface

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    htop \
    wkhtmltopdf

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --retries 10

COPY . .

RUN pip install -e .

# start in bash for interactive containers
CMD ["/bin/bash"]