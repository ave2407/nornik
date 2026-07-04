# Nornik Talc Segmentation: Project Context

## 1. Общая цель

Нужно собрать рабочий прототип для анализа микрофотографий и панорам руд:

- сегментировать области талька бинарной маской;
- обрабатывать большие панорамы через patch inference с перекрытием;
- давать web-интерфейс для просмотра, threshold-настройки и ручной правки маски;
- считать статистику по итоговой маске;
- экспортировать результаты;
- оставить интерфейсные заглушки под будущую модель классификации типа руды.

Сейчас главный inference-контур строится вокруг модели сегментации `UnetPlusPlus + efficientnet-b3`, обученной на ручных масках талька.

## 2. Данные

### Исходные данные

Исходная папка:

```text
C:\Users\Vladimir\Yandex.Disk\Загрузки\Задача 3. Скажи мне, кто твой шлиф
```

Внутри были:

- `Панорамы`
- `Фото руд по сортам. ч1`
- `Фото руд по сортам. ч2`

### Очищенная копия данных

Сделана отдельная нормализованная копия без exact-дубликатов:

```text
C:\Users\Vladimir\PycharmProjects\nornik\remaining_dataset_clean
/home/team117/nornik/remaining_dataset_clean
```

Структура:

```text
remaining_dataset_clean/
  images_by_class/
    talc/          126
    ordinary/      540
    refractory/     67
    thin/          391
  panoramas/        13
  annotations_blue_lines/
    talc_regions/   42
  manifest.csv
  duplicates_removed_sha256.csv
  summary.csv
```

Скрипт подготовки:

```text
prepare_remaining_dataset.py
```

### Датасет сегментации

Правильный датасет для сегментации собран из Label Studio JSON:

```text
C:\Users\Vladimir\PycharmProjects\nornik\talc_unetpp_dataset_from_json
/home/team117/nornik/talc_unetpp_dataset_from_json
```

Содержит:

- `images/` — 42 изображения
- `masks/` — 42 бинарные маски
- `train/` — 34 пары
- `val/` — 8 пар
- `debug/`, `overlays/`
- `manifest.csv`

Важно: ранний dataset, собранный простым сопоставлением по порядку, был неправильным. Использовать нужно именно `talc_unetpp_dataset_from_json`.

## 3. Что уже сделано по ML

### EDA

Сделан первичный анализ:

- количество изображений по классам;
- размеры изображений;
- exact duplicates по SHA256;
- похожие пары по perceptual hash.

Артефакты:

```text
EDA.ipynb
dataset_metadata.csv
exact_duplicate_groups_sha256.csv
similar_pairs_phash.csv
```

### Подготовка масок

Были две ветки:

1. Автоматическое извлечение областей из синих линий:
   - `extract_talc_masks.py`
   - оказалось недостаточно надежно для областей, касающихся краев.
2. Ручные маски из Label Studio:
   - `talc_masks_out/real_masks`
   - `talc1-5d08d959.json`
   - `prepare_real_masks_unetpp.py`
   - это стало основной разметкой.

### Обучение

Сделаны ноутбуки:

```text
train_unetpp_wandb.ipynb
train_unetpp_3fold_wandb.ipynb
train_final_unetpp_effb3_768_wandb.ipynb
```

Результат 3-fold для `UnetPlusPlus + efficientnet-b3`:

```text
fold 0: Dice 0.643, IoU 0.497
fold 1: Dice 0.725, IoU 0.595
fold 2: Dice 0.696, IoU 0.545
mean Dice: 0.688 ± 0.042
mean IoU:  0.545 ± 0.049
```

Финальная модель обучена на всех 42 масках:

```text
UnetPlusPlus
encoder: efficientnet-b3
img_size: 768
batch_size: 4
epochs: 25
```

Checkpoint на сервере:

```text
/home/team117/nornik/runs_unetpp_final/20260703-214420_final-unetpp-efficientnet-b3-768-all42/last_unetpp_effb3_768_all42.pt
```

ONNX экспорт:

```text
/home/team117/nornik/models/talc_unetpp_effb3_768.onnx
```

ONNX Runtime работает с `CUDAExecutionProvider`, если перед созданием session импортировать `torch`, чтобы подгрузить CUDA/cuDNN shared libraries из venv.

## 4. Сервер

Сервер:

```text
IP: 103.76.55.96
user: team117
key: C:\Users\Vladimir\.ssh\team117_id_rsa
project: /home/team117/nornik
venv: /home/team117/nornik/.venv
```

GPU фактически:

```text
NVIDIA L4
```

Не T4, как изначально говорили.

SSH:

```powershell
ssh -i "$env:USERPROFILE\.ssh\team117_id_rsa" team117@103.76.55.96
```

Туннель для web-приложения запускать на Windows, не внутри сервера:

```powershell
ssh -i "$env:USERPROFILE\.ssh\team117_id_rsa" -L 5173:127.0.0.1:5173 -L 7860:127.0.0.1:7860 team117@103.76.55.96
```

После этого открыть:

```text
http://127.0.0.1:5173
```

## 5. Web-приложение

Код:

```text
panorama_app/
  backend/
  frontend/
  requirements.txt
  start_server.sh
```

Backend:

- FastAPI
- ONNX Runtime
- OpenCV
- файловое хранение проектов

Frontend:

- React
- Vite
- TypeScript
- OpenSeadragon
- lucide-react

Запуск на сервере:

```bash
cd /home/team117/nornik
bash panorama_app/start_server.sh
```

## Latest panorama performance patch

Applied on 2026-07-04:

- Frontend threshold slider now updates local UI immediately, but sends the backend request with debounce and commits once on pointer release. This prevents a large panorama mask recomputation on every tiny slider movement.
- Backend mask tile endpoint now crops `mask_final.png` first and creates RGBA only for the requested tile crop. It no longer allocates a full-panorama RGBA array per mask tile request.
- Backend skips `overlay_preview.jpg` regeneration for images larger than `20_000_000` pixels. Large panoramas are displayed through image/mask tiles, so full preview regeneration is unnecessary during threshold/edit/reset.
- These changes target UI freezes during pan/zoom and threshold adjustment on panorama projects.

Сервисы:

```text
Backend:  http://127.0.0.1:7860
Frontend: http://127.0.0.1:5173
```

Логи:

```bash
tail -f /home/team117/nornik/logs/panorama_backend.log
tail -f /home/team117/nornik/logs/panorama_frontend.log
```

## 6. Backend API

Основные endpoints:

```text
GET  /api/health
GET  /api/projects
POST /api/projects
GET  /api/projects/{id}
POST /api/projects/{id}/infer
PATCH /api/projects/{id}/threshold
POST /api/projects/{id}/edits
POST /api/projects/{id}/export
GET  /api/classification/{project_id}
GET  /api/projects/{id}/source
GET  /api/projects/{id}/mask
GET  /api/projects/{id}/overlay
GET  /api/projects/{id}/tiles/image/{z}/{x}/{y}.jpg
GET  /api/projects/{id}/tiles/mask/{z}/{x}/{y}.png
```

Хранение проекта:

```text
data/projects/{project_id}/
  project.json
  source.*
  probability.npy
  mask_base.png
  mask_add.png
  mask_erase.png
  mask_final.png
  stats.json
  inference_meta.json
  overlay_preview.jpg
  exports/
```

Финальная маска:

```text
base = probability >= threshold
final = (base OR mask_add) AND NOT mask_erase
```

Это нужно, чтобы threshold можно было менять без потери ручных правок.

## 7. Что уже работает в приложении

Проверено end-to-end на сервере:

```text
upload -> infer -> ready -> stats -> export zip
```

Проверенный результат:

```text
status: ready
fill_percent: 19.68%
component_count: 216
export zip: 15.5 MB
```

Работает:

- upload изображения;
- создание проекта;
- ONNX inference на GPU;
- patch inference `768` с overlap `192`;
- weighted stitching;
- сохранение `probability.npy`;
- пересчет маски по threshold;
- ручные слои `mask_add`, `mask_erase`;
- итоговая `mask_final`;
- статистика:
  - total pixels;
  - mask pixels;
  - fill percent;
  - connected components, filtered by minimum component size;
  - largest component pixels;
  - largest component bbox;
- export ZIP;
- classification stub:
  - `class_name = unknown`;
  - `confidence = null`;
  - `model_version = stub`;
- frontend viewer;
- threshold slider;
- opacity slider;
- brush add;
- erase;
- stats panel;
- export button.

## 8. Последняя проблема и фикс

Проблема:

- большая панорама обрабатывалась слишком долго;
- во время/после переключения между проектами изображение и маска могли пропасть;
- сервер временно перестал отвечать по SSH, вероятно из-за тяжелой обработки панорамы.

Уже примененный фикс:

- inference переведен в отдельный `ThreadPoolExecutor(max_workers=1)`;
- `/api/projects/{id}/infer` теперь быстро возвращает статус `running`, а обработка идет фоном;
- в `project.json` добавлен `inference_progress`;
- probability stitching теперь потоковый batch-by-batch, без хранения всех tile predictions в RAM;
- `overlay_preview.jpg` ограничен `max_side=4096`, чтобы не кодировать огромную панораму целиком;
- frontend mask layer обновляется через `maskRevision`, без потери слоя при переключении threshold/edit;
- UI показывает прогресс inference.

Дополнительный фикс после теста UI:

- edge tiles теперь всегда отдаются как `256x256`, а недостающая часть edge tile заполняется пустым canvas; это убирает растянутые/размытые края изображений;
- добавлен отдельный инструмент `Pan`; при `Add`/`Erase` OpenSeadragon mouse navigation отключается, поэтому drag уходит в brush stroke, а не в перемещение фотографии;
- `threshold` и `edit` запрещены для проектов не в статусе `ready`, API возвращает `409`, а не `500`;
- добавлен endpoint `POST /api/projects/{id}/cancel`;
- зависшая старая панорама `f3fc4a2304e2` переведена в `cancelled`;
- tile endpoints используют LRU cache для `cv2.imread`, чтобы панорама не декодировалась с диска заново на каждый тайл.

Проверено на сервере:

```text
edge image tile: 256x256
edge mask tile:  256x256 RGBA
edit changed mask_pixels: 1050701 -> 1060123
threshold on cancelled project: 409 Mask is not ready yet
```

Component statistics note:

- `mask_pixels` and `fill_percent` are computed over the full final mask.
- `component_count` and `largest_component_*` ignore small components below `PANORAMA_MIN_COMPONENT_PIXELS`.
- Default minimum component size: `500` pixels.

Файлы фикса:

```text
panorama_app/backend/schemas.py
panorama_app/backend/storage.py
panorama_app/backend/inference.py
panorama_app/backend/main.py
panorama_app/frontend/src/api.ts
panorama_app/frontend/src/App.tsx
```

## 9. Что надо сделать дальше

### Срочно

1. После изменений кода перезапустить приложение:

```bash
cd /home/team117/nornik
bash panorama_app/start_server.sh
```

2. Проверить:

```bash
curl http://127.0.0.1:7860/api/health
tail -f logs/panorama_backend.log
```

3. Повторить тест:
   - открыть UI;
   - загрузить небольшое изображение;
   - запустить inference;
   - подвигать threshold;
   - переключиться на другой проект;
   - убедиться, что image/mask не пропадают.

### Для панорам

1. На первой панораме начать с batch size `1` или `2`, если снова будет тяжело.
2. Добавить настройку batch size/tile overlap из UI или backend config.
3. Сохранять preview не только overlay, но и downsample probability для быстрого просмотра.
4. Подумать про tiled storage для `probability`, потому что `.npy` полной панорамы может быть большим.

### Для UI

1. Добавить кнопку cancel inference.
2. Добавить undo/redo stroke history на backend, а не только в UI.
3. Добавить режим polygon/lasso для крупных правок.
4. Добавить горячие клавиши:
   - `B` brush;
   - `E` erase;
   - `[` / `]` brush size;
   - `Ctrl+Z` undo.
5. Улучшить отображение прогресса по тайлам.

### Для ML

1. Подобрать threshold по fold validation, а не держать только `0.5`.
2. Проверить `img_size=768` на валидационных fold-ах, если будет время.
3. Добавить test-time augmentation для inference, если качество на панорамах слабое.
4. Обучить классификационную модель на `remaining_dataset_clean/images_by_class`.
5. Подключить классификатор вместо `DummyClassifierService`.

### Для деплоя

1. Сделать `systemd` user service или supervisor script.
2. Добавить `.env` для портов, model path, batch size.
3. Добавить ротацию логов.
4. Добавить простой health dashboard в UI.

## 10. Команды диагностики

Процессы:

```bash
ps -eo pid,stat,pcpu,pmem,etime,cmd | grep -E "uvicorn|vite|npm|python|onnx" | grep -v grep
```

GPU:

```bash
nvidia-smi
```

Логи:

```bash
tail -200 logs/panorama_backend.log
tail -100 logs/panorama_frontend.log
tail -f logs/panorama_backend.log
```

Остановить приложение:

```bash
pkill -f "uvicorn panorama_app.backend.main:app"
pkill -f "vite.*5173"
pkill -f "npm run dev"
```

Перезапустить:

```bash
cd /home/team117/nornik
bash panorama_app/start_server.sh
```
