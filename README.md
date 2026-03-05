# Person Detection with YOLOv8

## Dataset

Para entrenar el detector de personas se ha utilizado el dataset **persons Computer Vision Dataset**, disponible en Roboflow Universe:

https://universe.roboflow.com/enova/persons-gfzae

El dataset contiene **3875 imágenes anotadas**, **una única clase (person)** y está compuesto por escenas **indoor** y **outdoor**, donde aparecen personas en diferentes condiciones de iluminación, distancias y perspectivas. Cada instancia de la clase *person* está anotada mediante **bounding boxes**. El dataset se divide en **3175 imágenes para entrenamiento**, **400 imágenes para validación** y **300 imágenes para test**. Roboflow permite exportar el dataset en **formato YOLO**.




## Demo

[▶️ Watch the detection video](people-detection_out.mp4)
