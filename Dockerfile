FROM nvcr.io/nvidia/pytorch:19.08-py3

WORKDIR /workspace/super-mario-bros-dqn

COPY requirements.txt /workspace/super-mario-bros-dqn
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

RUN apt-get update && \
    apt-get install -y ffmpeg

# Workaround to squelch eroneous errors thrown by NES-Py
RUN sed -i "s/raise ValueError('env has already been closed.')/return/g" \
    /opt/conda/lib/python3.6/site-packages/nes_py/nes_env.py

COPY . /workspace/super-mario-bros-dqn

ENTRYPOINT ["python", "train.py"]
