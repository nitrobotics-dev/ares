FROM python:3.10

WORKDIR /ares


# set work directory and pythonpath for imports
WORKDIR /ares
ENV PYTHONPATH="${PYTHONPATH}:/ares/openvla"

COPY . .
RUN pip install -e openvla/ && pip install -r requirements.txt && pip install tensorflow_graphics

# start in bash for interactive containers
CMD ["/bin/bash"]