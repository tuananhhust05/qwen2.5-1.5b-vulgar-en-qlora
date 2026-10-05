FROM nvcr.io/nvidia/pytorch:24.10-py3

WORKDIR /workspace/llm-finetune

ENV PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/workspace/hf_cache \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY requirements-ngc.txt pyproject.toml ./
RUN pip install --no-cache-dir -r requirements-ngc.txt

COPY . .
RUN pip install --no-cache-dir -e . && chmod +x scripts/*.sh

CMD ["bash", "scripts/train.sh"]
