FROM verlai/verl:app-verl0.4-vllm0.8.5-mcore0.12.2-te2.2
COPY requirements.txt /myworkspace/requirements.txt
RUN pip install -r /myworkspace/requirements.txt
RUN pip install -U nvidia-pytriton
RUN git clone https://github.com/yuekaizhang/verl.git /myworkspace/verl -b thread && cd /myworkspace/verl && pip install --no-deps -e .
RUN git clone https://github.com/yuekaizhang/PytritonSenseVoice.git /myworkspace/PytritonSenseVoice && cd /myworkspace/PytritonSenseVoice && pip install -e .