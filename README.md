# Person Detection with YOLOv8

## Dataset

Para el entrenamiento del modelo de detección de personas se ha utilizado el dataset **persons Computer Vision Dataset**, disponible en Roboflow Universe:

https://universe.roboflow.com/enova/persons-gfzae

El dataset contiene aproximadamente **3875 imágenes anotadas** destinadas a tareas de **detección de objetos**. Cada instancia de la clase *person* está anotada mediante **bounding boxes**, lo que permite entrenar modelos de detección como YOLOv8.

El dataset incluye **una única clase (person)** y está compuesto principalmente por escenas **outdoor**, donde aparecen personas en diferentes condiciones de iluminación, distancias y perspectivas. 

Roboflow permite además exportar el dataset en **formato YOLO**, lo que facilita su integración directa en el pipeline de entrenamiento del modelo.





## Demo

[▶️ Watch the detection video](people-detection_out.mp4)
