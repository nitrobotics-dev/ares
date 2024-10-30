FROM python:3.10

WORKDIR /ares

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

RUN pip install -e .

# start in bash for interactive containers
CMD ["/bin/bash"]