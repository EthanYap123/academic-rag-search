FROM python:3.12-slim

# Hugging Face Spaces run containers as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    DATA_DIR=/home/user/data
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Bake the embedding model into the image so the first request doesn't download it
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2 as E; E()(['warm up'])"

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
