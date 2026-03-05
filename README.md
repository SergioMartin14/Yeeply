# Person Detection with YOLOv8

## Dataset

Para el entrenamiento del modelo de detección de personas se ha utilizado el dataset **Persons**, disponible en Roboflow Universe:

https://universe.roboflow.com/enova/persons-gfzae

Este dataset fue seleccionado porque contiene imágenes anotadas específicamente para la detección de la clase **person** en entornos exteriores (outdoor). Las imágenes presentan variabilidad en condiciones de iluminación, ángulos de cámara, distancias y densidad de personas, lo que permite entrenar un modelo más robusto frente a diferentes escenarios del mundo real.

La plataforma Roboflow proporciona además métricas relevantes del dataset, como:

- Número total de imágenes
- Número de anotaciones de objetos
- División del dataset en conjuntos de **train**, **validation** y **test**
- Distribución de clases
- Formato de anotación compatible con modelos YOLO

Estas características facilitan la integración directa del dataset con el pipeline de entrenamiento de **YOLOv8**.





## Demo

[▶️ Watch the detection video](people-detection_out.mp4)
