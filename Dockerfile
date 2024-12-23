FROM python:3.10

WORKDIR /ares

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