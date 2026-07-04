# Nornik Ore Analysis Submission

## Состав

В Docker-образ входит только runtime-минимум:

- `panorama_app` - FastAPI backend и собранный React frontend;
- `models/talc_unetpp_effb3_768.onnx` - сегментация талька;
- `models/ore_classifier_3class_effb3.onnx` - классификация руды;
- `models/ore_classifier_3class_effb3.json` - metadata классов;
- `demo_images/` - несколько изображений для быстрой демонстрации;
- `PROJECT_CONTEXT.md` - подробное описание логики.

Большие обучающие датасеты, ноутбуки обучения U-Net и wandb/runs в Docker не входят.

## Запуск

```bash
docker compose up --build
```

UI:

```text
http://localhost:7860
```

Для GPU нужен NVIDIA Container Toolkit. В `docker-compose.yml` включено:

```yaml
runtime: nvidia
NVIDIA_VISIBLE_DEVICES: all
```

Если проверяющий запускает без Compose, эквивалентно:

```bash
docker run --gpus all -p 7860:7860 -v ./data:/app/data nornik-ore-analysis:latest
```

Рабочие проекты сохраняются в volume:

```text
./data:/app/data
```

## Демо

Файлы для быстрой загрузки лежат в контейнере и в репозитории:

```text
demo_images/
  demo_talc_00001.jpg
  demo_ordinary_00001.jpg
  demo_difficult_thin_00001.jpg
  demo_difficult_refractory_00001.jpg
  demo_panorama_00001.jpg
```

## Что показывает UI

- `Final class` - итоговый класс после экспертной логики.
- `Model class` и вероятности - сырой вывод ONNX-классификатора.
- `Talc / Phases` - переключение маски талька и вспомогательных фазовых эвристик.
- `View controls` справа - brush size, threshold, opacity.
- `Export` - ZIP с масками, статистиками, классификацией и overlay.
