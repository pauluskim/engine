FROM nvcr.io/nvidia/tritonserver:23.08-py3

ENV WORK_DIR=/server
ENV MODEL_DIR=${WORK_DIR}/models
RUN mkdir -p $MODEL_DIR
WORKDIR $WORK_DIR

COPY requirements.txt $WORK_DIR/
RUN apt-get install -y --no-install-recommends python3 python3-pip
RUN pip3 install --upgrade pip  \
    && pip3 install --upgrade wheel setuptools &&  \
    pip3 install -r requirements.txt

COPY models $MODEL_DIR



# Docker build
# docker build -t server:test .

# Docker run
# docker run --rm --shm-size=1gb -p8000:8000 -p8001:8001 -p8002:8002 server:test tritonserver --model-repository=/server/models --model-control-mode=explicit --load-model=ebr

# Docker run without launching
# 1. docker run --rm --shm-size=1gb -p8000:8000 -p8001:8001 -p8002:8002 -v /Users/jack/engine/learning_spoons_lec/server/models:/server/models -t -i server:test bash
# 2. In the docker, run the below command to launch the triton server
# tritonserver --model-repository=/server/models --model-control-mode=explicit --load-model=ebr