# Как запустить проект локально (без GPU)

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