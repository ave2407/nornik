FROM node:20-bookworm-slim AS frontend

WORKDIR /build/panorama_app/frontend
COPY panorama_app/frontend/package.json ./
RUN npm install
COPY panorama_app/frontend/ ./
RUN npm run build

FROM python:3.10-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PANORAMA_DATA_ROOT=/app/data
ENV PANORAMA_MODELS_ROOT=/app/models
ENV TALC_ONNX_MODEL=/app/models/talc_unetpp_effb3_768.onnx
ENV PANORAMA_BATCH_SIZE=4

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
      libgomp1 \
      libgl1 \
      ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY panorama_app/requirements.txt /app/panorama_app/requirements.txt
RUN python -m pip install --no-cache-dir --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/panorama_app/requirements.txt

COPY panorama_app /app/panorama_app
COPY --from=frontend /build/panorama_app/frontend/dist /app/panorama_app/frontend/dist
COPY models /app/models
COPY demo_images /app/demo_images
COPY PROJECT_CONTEXT.md /app/PROJECT_CONTEXT.md
COPY SUBMISSION_README.md /app/SUBMISSION_README.md

RUN mkdir -p /app/data/projects /app/logs

EXPOSE 7860

CMD ["uvicorn", "panorama_app.backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
