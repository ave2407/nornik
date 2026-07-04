# Nornik Ore Analysis

Web-система для анализа микрофотографий и панорам руд: классифицирует сорт руды, сегментирует область оталькования, показывает вспомогательные фазовые маски и считает статистики по изображению.

## Что решает система

Итоговая задача - быстро загрузить `jpg/png/tif/bmp`, получить класс руды, маски и численные показатели, которые можно показать экспертам или экспортировать.

Система работает с 3 классами:

- `talc` - оталькованная руда;
- `ordinary` - рядовая руда;
- `difficult` - труднообогатимая руда.

В исходных данных было 4 папки, но `thin` и `refractory` объединены в один класс `difficult`.

## Главные возможности

- **Классификация руды ONNX-моделью**: EfficientNet-B3, 3 класса, вход `512x512`.
- **Сегментация талька**: U-Net++ с encoder `efficientnet-b3`, ONNX, patch inference для больших панорам.
- **Панорамы**: изображение режется на overlapping tiles `768x768`, вероятности сшиваются weighted blending.
- **Интерактивный viewer**: OpenSeadragon, zoom/pan, тайловая подгрузка изображения и масок.
- **Ручная правка талька**: кисть `Add talc`, ластик `Erase talc`, отдельные слои правок.
- **Threshold без потери правок**: итоговая маска считается как `(base OR mask_add) AND NOT mask_erase`.
- **Фазовые подсказки**: синий - тальк, зелёный - крупные сульфидные срастания, красный - тонкие/рассеянные срастания.
- **Экспорт ZIP**: исходник, маски, overlay, вероятности, статистики и классификация.
- **Docker-пакет для сдачи**: backend и frontend запускаются одним сервисом на порту `7860`.

## Как устроена логика

### 1. Сегментация талька

Модель:

```text
models/talc_unetpp_effb3_768.onnx
architecture: UnetPlusPlus + efficientnet-b3
input: 1 x 3 x 768 x 768
output: probability mask
```

Для больших изображений:

```text
tile_size = 768
overlap = 192
stride = 576
batch_size = 4
```

После inference сохраняется:

```text
probability.npy
mask_base.png
mask_add.png
mask_erase.png
mask_final.png
stats.json
```

Финальная маска:

```text
base = probability >= threshold
final = (base OR mask_add) AND NOT mask_erase
```

Так можно двигать threshold и не терять ручные правки.

### 2. Классификация руды

Модель:

```text
models/ore_classifier_3class_effb3.onnx
metadata: models/ore_classifier_3class_effb3.json
architecture: efficientnet-b3
input: 512 x 512 RGB
classes: ordinary, difficult, talc
```

В UI всегда показывается сырой вывод модели:

- `Model class`
- `Model confidence`
- `Model ordinary`
- `Model difficult`
- `Model talc`

И отдельно итоговый класс `Final class`.

### 3. Экспертная логика

Экспертное правило применено поверх модели:

```text
если talc_percent > 10% -> Final class = talc
иначе Final class = Model class
```

Причина: тальк технологически критичен, и при высокой доле образец автоматически относится к оталькованной руде.

Важно: фазовые маски сульфидов сейчас эвристические, поэтому они используются как подсказка и визуальное объяснение, но не скрывают вывод классификатора.

### 4. Фазовые статистики

Файл:

```text
panorama_app/backend/phase_analysis.py
```

Считаются:

- `talc_percent`
- `sulfide_percent`
- `gangue_percent`
- `ordinary_intergrowth_area_percent`
- `thin_intergrowth_area_percent`
- `fine_component_count`
- `coarse_component_count`
- `largest_sulfide_component_pixels`

Эвристика:

- тальк берётся из `mask_final.png`;
- сульфиды ищутся как яркие/металлические области по `Lab/HSV`;
- крупные connected components считаются обычными срастаниями;
- мелкие/раздробленные components считаются тонкими срастаниями;
- всё остальное - нерудная матрица.

## Интерфейс

Левая панель:

- загрузка изображения;
- запуск inference;
- отмена inference;
- список проектов.

Верхняя панель:

- `Pan`
- `Add talc`
- `Erase talc`
- `Talc`
- `Phases`
- `Reset all`
- `Export`

Правая панель:

- статус проекта;
- `View controls`: brush size, threshold, opacity;
- итоговая классификация;
- вероятности модели;
- фазовые статистики;
- статистика маски талька.

## Структура проекта

```text
panorama_app/
  backend/
    main.py              FastAPI API и static frontend serving
    inference.py         ONNX U-Net++ inference
    classifier.py        ONNX classifier + expert rule
    phase_analysis.py    фазовые эвристики и статистики
    storage.py           хранение проектов на диске
    stats.py             статистика mask_final
    exporter.py          экспорт ZIP/JSON/PNG/JPG
    schemas.py           Pydantic schemas
  frontend/
    src/App.tsx          React UI
    src/api.ts           API client
    src/styles.css       стили

models/
  talc_unetpp_effb3_768.onnx
  ore_classifier_3class_effb3.onnx
  ore_classifier_3class_effb3.json

demo_images/
  demo_talc_00001.jpg
  demo_ordinary_00001.jpg
  demo_difficult_thin_00001.jpg
  demo_difficult_refractory_00001.jpg
  demo_panorama_00001.jpg

ore_3class/
  manifest.csv
  summary.csv
  README.md
```

## Запуск через Docker

Нужен Docker. Для GPU желательно установить NVIDIA Container Toolkit, но приложение имеет CPU fallback.

Сборка и запуск:

```bash
docker compose up --build
```

Открыть:

```text
http://localhost:7860
```

Остановить:

```bash
docker compose down
```

Запуск уже собранного образа:

```bash
docker compose up
```

Логи:

```bash
docker compose logs -f ore-analysis
```

Если запускаете без Compose:

```bash
docker build -t nornik-ore-analysis:latest .
docker run --gpus all -p 7860:7860 -v ./data:/app/data nornik-ore-analysis:latest
```

Если GPU недоступен:

```bash
docker run -p 7860:7860 -v ./data:/app/data nornik-ore-analysis:latest
```

Рабочие проекты сохраняются в:

```text
./data
```

## Запуск без Docker

На сервере:

```bash
cd /home/team117/nornik
bash panorama_app/start_server.sh
```

Backend:

```text
http://127.0.0.1:7860
```

Frontend dev server:

```text
http://127.0.0.1:5173
```

При Docker-сборке frontend отдаётся самим FastAPI на `7860`, поэтому отдельный Vite-сервер не нужен.

## Demo flow

1. Открыть `http://localhost:7860`.
2. Нажать `Upload image`.
3. Выбрать файл из `demo_images/`.
4. Нажать `Run inference`.
5. После статуса `ready` посмотреть:
   - `Final class`;
   - `Model class`;
   - вероятности модели;
   - `Talc` overlay;
   - `Phases` overlay;
   - статистики справа.
6. При необходимости поправить тальк кистью.
7. Нажать `Export`.

## API

Основные endpoints:

```text
GET  /api/health
GET  /api/projects
POST /api/projects
GET  /api/projects/{id}
POST /api/projects/{id}/infer
POST /api/projects/{id}/cancel
POST /api/projects/{id}/reset
PATCH /api/projects/{id}/threshold
POST /api/projects/{id}/edits
POST /api/projects/{id}/analyze
GET  /api/classification/{id}
POST /api/projects/{id}/export
GET  /api/projects/{id}/tiles/image/{z}/{x}/{y}.jpg
GET  /api/projects/{id}/tiles/mask/{z}/{x}/{y}.png
GET  /api/projects/{id}/tiles/phases/{z}/{x}/{y}.png
```

## Что входит в export ZIP

```text
source image
project.json
stats.json
phase_stats.json
classification.json
inference_meta.json
mask_base.png
mask_add.png
mask_erase.png
mask_final.png
phase_overlay.jpg
overlay_preview.jpg
phase_masks.npz
probability.npy
```

## Ограничения и честные замечания

- Маска талька может ошибаться на чёрном фоне панорам: модель училась на тёмных областях, а чёрный фон визуально похож.
- Фазовые маски сульфидов являются эвристикой, а не обученной сегментацией.
- Поэтому в UI отдельно сохранён вывод ONNX-классификатора, который не зависит напрямую от качества красно-зелёных фазовых масок.

Быстрый следующий шаг для улучшения талька: добавить `sample/background mask`, которая зануляет probability талька вне области образца и убирает чёрный фон, связанный с краями изображения.
