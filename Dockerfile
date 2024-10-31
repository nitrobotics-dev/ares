FROM python:3.10

WORKDIR /ares

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt --retries 10

COPY . .

RUN pip install -e .

# RUN apt-get update && apt-get install -y htop

# start in bash for interactive containers
CMD ["/bin/bash"]