# Как запустить проект локально (без GPU, например на Mac)

Этот файл — не для сдачи, а для проверки перед `git push`. Основной способ запуска
для проверяющих описан в `SUBMISSION_README.md` (Docker + GPU-сервер). Здесь —
что делать, если под рукой только обычный ноутбук без NVIDIA-видеокарты.

## Способ 1 — Docker без GPU (рекомендуется)

Плюсы: использует настоящий `Dockerfile`, ничего не подменяет и не редактирует —
самая близкая к реальному деплою проверка. Минус: первая сборка занимает несколько минут.

```bash
cd nornik
docker build -t nornik-ore-analysis .
docker run --rm -p 7860:7860 -v "$(pwd)/data:/app/data" nornik-ore-analysis
```

Открыть: **http://127.0.0.1:7860** (фронтенд и бэкенд на одном порту — так собран Dockerfile).

Без GPU модель сама переключится на CPU (в `inference.py` это уже предусмотрено:
`CUDAExecutionProvider` добавляется только если доступен, иначе `CPUExecutionProvider`).
Работать будет чуть медленнее, но корректно.

**Важно:** не запускать `docker compose up` на локальной машине без GPU — в
`docker-compose.yml` жёстко прописано `runtime: nvidia`, без GPU-рантайма контейнер
не стартует. Compose — только для сервера команды с настоящей видеокартой.

## Способ 2 — Dev-режим (venv + npm), без Docker

Плюсы: запускается за секунды, удобно для быстрой правки кода и просмотра
изменений на лету (hot reload). Минус: нужно руками поставить зависимости, и на
Mac без GPU **нельзя** ставить `onnxruntime-gpu` из `requirements.txt` как есть —
для него просто нет сборки под macOS.

```bash
# Бэкенд
cd nornik/panorama_app
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install fastapi uvicorn python-multipart numpy opencv-python-headless pillow pydantic onnx onnxruntime
uvicorn panorama_app.backend.main:app --host 127.0.0.1 --port 7860 &

# Фронтенд (в отдельном терминале)
cd nornik/panorama_app/frontend
npm install   # если ещё не делали
npm run dev -- --host 127.0.0.1 --port 5173
```

Открыть: **http://127.0.0.1:5173**

**Не редактировать `requirements.txt` под это** — там правильные версии для
GPU-сервера (`onnxruntime-gpu`, `opencv-python-headless`), они должны остаться
как есть для реального деплоя. Здесь ставим CPU-версии только в локальный venv,
файл в репозитории не трогаем.

## Способ 3 — Docker + Compose с GPU (для сервера команды)

Именно так это будет запускать сервер/проверяющий — описано в `SUBMISSION_README.md`:

```bash
docker compose up --build
```

Требует NVIDIA Container Toolkit и настоящую видеокарту — **на Mac не заработает**,
привожу только для полноты картины.

## Что выбрать

| Хотите | Способ |
| --- | --- |
| Максимально надёжно проверить перед пушем, не спешите | **1 — Docker без GPU** |
| Быстро посмотреть/поправить что-то в коде прямо сейчас | **2 — Dev-режим** |
| Проверить именно на сервере команды, как для сдачи | **3 — Compose с GPU** (не на Mac) |

Если сомневаетесь — начните со **способа 1**: он ближе всего к тому, что увидят
проверяющие, и не требует ничего дополнительно настраивать вручную.
