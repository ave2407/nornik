# Nornik Ore Analysis: текущий контекст проекта

## Цель

Система анализирует микрофотографии и панорамы руд:

- сегментирует область оталькования;
- классифицирует сорт руды в 3 класса;
- оценивает фазовые признаки: тальк, сульфидные включения, нерудная матрица;
- показывает результат в web-интерфейсе с тайловым просмотром, threshold, ручной правкой маски и экспортом.

Итоговые классы:

- `talc` - оталькованная руда;
- `ordinary` - рядовая руда;
- `difficult` - труднообогатимая руда.

Папки `thin` и `refractory` объединены в один класс `difficult`.

## Где что лежит

Локально:

```text
C:\Users\Vladimir\PycharmProjects\nornik
```

На сервере:

```text
/home/team117/nornik
```

Основные файлы:

```text
panorama_app/
  backend/
    main.py              FastAPI endpoints
    inference.py         ONNX U-Net++ talc inference
    classifier.py        ONNX ore classifier + expert rules
    phase_analysis.py    heuristic phase masks and statistics
    storage.py           filesystem project storage
    stats.py             talc mask connected components
    exporter.py          ZIP/JSON/PNG/JPG export
    schemas.py           API schemas
  frontend/
    src/App.tsx          React UI
    src/api.ts           frontend API client
    src/styles.css       UI styles
  start_server.sh
  requirements.txt

ore_3class/
  manifest.csv           3-class image classification manifest
  summary.csv            class counts
  README.md              class mapping

prepare_ore_3class_manifest.py
train_ore_classifier_3class_wandb.ipynb
```

Модели на сервере:

```text
models/talc_unetpp_effb3_768.onnx
models/ore_classifier_3class_effb3.onnx
models/ore_classifier_3class_effb3.json
models/ore_classifier_3class_effb3.pt
```

Данные:

```text
remaining_dataset_clean/
  images_by_class/
    ordinary/      540
    talc/          126
    thin/          391
    refractory/     67
  panoramas/        13
  annotations_blue_lines/talc_regions/ 42

talc_unetpp_dataset_from_json/
  images/ 42
  masks/  42
```

3-классовый manifest:

```text
ordinary   -> ordinary   = 540
talc       -> talc       = 126
thin       -> difficult  = 391
refractory -> difficult  = 67

Итого difficult = 458
```

## Inference талька

Модель:

```text
UnetPlusPlus
encoder: efficientnet-b3
input: 3 x 768 x 768
output: probability mask
runtime: ONNX Runtime
```

Панорамы обрабатываются patch-wise:

```text
tile_size = 768
overlap = 192
stride = 576
batch_size = 4
```

Схема:

1. Изображение режется на перекрывающиеся патчи.
2. Края добиваются padding.
3. Патчи батчами проходят через ONNX U-Net++.
4. Вероятности сшиваются weighted blending.
5. Сохраняется `probability.npy`.
6. Binary mask строится по threshold:

```text
base = probability >= threshold
final = (base OR mask_add) AND NOT mask_erase
```

Файлы проекта:

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
  classification.json
  phase_stats.json
  phase_masks.npz
  phase_overlay.jpg
  overlay_preview.jpg
  exports/
```

Ручные правки хранятся отдельно:

- `mask_add.png` - добавленные кистью области;
- `mask_erase.png` - стёртые области;
- `mask_final.png` - итог после threshold и правок.

Это позволяет менять threshold без потери ручных правок.

## Классификация руды

Используется ONNX-классификатор:

```text
models/ore_classifier_3class_effb3.onnx
input size: 512 x 512
classes: ordinary, difficult, talc
preprocess: RGB, resize 512, ImageNet normalize
provider: CPUExecutionProvider
```

CPU provider выбран специально: классификатор маленький, а CUDA provider на сервере для этой ONNX-модели ругался на `libcudnn.so.9`. Для скорости UI это не критично.

Backend всегда возвращает вывод модели:

```text
model_class_name
model_display_name
model_confidence
model_probs
```

Итоговый класс `class_name` считается так:

1. Сначала считается фазовая статистика и `talc_percent`.
2. Если `talc_percent > 10%`, итоговый класс принудительно `talc`.
3. Если `talc_percent <= 10%`, итоговый класс берётся из ONNX-классификатора.
4. Фазовые маски обычных/тонких срастаний остаются как вспомогательное объяснение, потому что сейчас они эвристические и могут быть неточными.

Это важно: даже если фазовое выделение красных/зелёных масок плохое, UI всё равно показывает реальный вывод модели.

Поля ответа `/api/classification/{project_id}`:

```text
class_name              итоговый класс после expert rule
display_name            русское имя итогового класса
confidence              confidence итогового класса
probs                   вероятности модели
model_class_name        сырой класс ONNX-модели
model_display_name      русское имя класса модели
model_confidence        confidence модели
model_probs             вероятности модели по 3 классам
model_version           ore-effb3-onnx-512
rule_version            версия экспертного правила
decision_reason         текстовое объяснение
phase_stats             статистики фаз
```

## Фазовый анализ

Фазовый анализ находится в:

```text
panorama_app/backend/phase_analysis.py
```

Он нужен для объяснения и статистик, но не должен полностью заменять классификатор.

Текущие эвристики:

- тальк - `mask_final.png` из U-Net++ и ручных правок;
- сульфидные включения - bright/metallic regions по `Lab L`, `HSV V/S`;
- обычные срастания - крупные connected components сульфидной маски;
- тонкие срастания - мелкие/раздробленные connected components;
- нерудная матрица - остаток изображения.

Цвета в UI:

```text
blue  - talc
green - ordinary/coarse sulfide intergrowth
red   - thin/disseminated sulfide intergrowth
```

Считаемые статистики:

```text
source_width
source_height
analysis_width
analysis_height
analysis_scale
total_pixels
talc_pixels
sulfide_pixels
gangue_pixels
ordinary_intergrowth_pixels
thin_intergrowth_pixels
talc_percent
sulfide_percent
gangue_percent
ordinary_intergrowth_area_percent
thin_intergrowth_area_percent
min_component_pixels
coarse_component_pixels
sulfide_component_count
fine_component_count
coarse_component_count
largest_sulfide_component_pixels
```

Формулы:

```text
talc_percent = talc_pixels / total_pixels * 100
sulfide_percent = sulfide_pixels / total_pixels * 100
gangue_percent = gangue_pixels / total_pixels * 100
ordinary_intergrowth_area_percent = ordinary_intergrowth_pixels / total_pixels * 100
thin_intergrowth_area_percent = thin_intergrowth_pixels / total_pixels * 100
```

Для больших изображений фазовый анализ делается на downsample до `MAX_ANALYSIS_SIDE = 4096`, чтобы не блокировать сервер.

## Статистика маски талька

`panorama_app/backend/stats.py` считает статистику по `mask_final.png`:

```text
total_pixels
mask_pixels
fill_percent
component_count
largest_component_pixels
largest_component_bbox
```

Маленькие компоненты фильтруются:

```text
PANORAMA_MIN_COMPONENT_PIXELS = 500
```

`mask_pixels` и `fill_percent` считаются по всей маске, а `component_count` и `largest_component_*` считают только компоненты >= 500 px.

## Web UI

Frontend:

```text
panorama_app/frontend/src/App.tsx
```

Основные функции:

- upload jpg/png/tif/bmp;
- run inference;
- cancel inference;
- tile viewer через OpenSeadragon;
- переключение `Talc` / `Phases`;
- threshold slider;
- opacity slider;
- brush add/erase для талька;
- reset all;
- export ZIP;
- панель `Ore class`:
  - final class;
  - model class;
  - model probabilities;
  - talc/sulfide/gangue percentages;
  - phase component counts;
  - decision reason.

## Backend API

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
GET  /api/projects/{id}/source
GET  /api/projects/{id}/mask
GET  /api/projects/{id}/overlay
GET  /api/projects/{id}/phase_overlay
GET  /api/projects/{id}/tiles/image/{z}/{x}/{y}.jpg
GET  /api/projects/{id}/tiles/mask/{z}/{x}/{y}.png
GET  /api/projects/{id}/tiles/phases/{z}/{x}/{y}.png
```

Export ZIP включает:

```text
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
source image
```

## Запуск

На сервере:

```bash
cd /home/team117/nornik
bash panorama_app/start_server.sh
```

Туннель с Windows:

```powershell
ssh -i "$env:USERPROFILE\.ssh\team117_id_rsa" -L 5173:127.0.0.1:5173 -L 7860:127.0.0.1:7860 team117@111.88.151.8
```

UI:

```text
http://127.0.0.1:5173
```

Логи:

```bash
tail -f /home/team117/nornik/logs/panorama_backend.log
tail -f /home/team117/nornik/logs/panorama_frontend.log
```

## Важные текущие проблемы

1. Чёрный фон на панорамах может ошибочно попадать в маску талька, потому что U-Net++ училась на тёмных областях талькования.
2. Фазовые маски сульфидов (`green/red`) эвристические и могут быть неточными.
3. Поэтому вывод ONNX-классификатора всегда показывается отдельно и не скрывается экспертной логикой.

Следующий быстрый фикс для талька:

- добавить `sample/background mask`;
- занулять probability талька вне области образца;
- использовать connected component от краёв для чёрного фона;
- затем при необходимости fine-tune U-Net++ на hard negatives с пустыми масками.

## Git / сохранение состояния

Локальный репозиторий находится здесь:

```text
C:\Users\Vladimir\PycharmProjects\nornik
```

На сервере `/home/team117/nornik` сейчас нет `.git`, поэтому сохранение версии делается локальным commit после синхронизации нужных серверных файлов.

Файл `add_disk.sh` добавлен локально от организаторов и не включён в commit приложения.
