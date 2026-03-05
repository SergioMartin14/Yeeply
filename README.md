# Person Detection with YOLOv8

## Dataset

Para entrenar el detector de personas se ha utilizado el dataset **persons Computer Vision Dataset**, disponible en Roboflow Universe:

https://universe.roboflow.com/enova/persons-gfzae

El dataset contiene **3875 imágenes anotadas**, **una única clase (person)** y está compuesto por escenas **indoor** y **outdoor**, donde aparecen personas en diferentes condiciones de iluminación, distancias y perspectivas. 

![Example Dataset](Ejemplo_dataset_1.PNG)

Cada instancia de la clase *person* está anotada mediante **bounding boxes**. 

Una característica destacable de este dataset es que cada imagen suele contener **múltiples personas**, generando un gran número de **bounding boxes por imagen**. Además, muchas de estas detecciones corresponden a personas parcialmente visibles, pequeñas o en segundo plano, lo que introduce anotaciones **más sutiles y difíciles de detectar**. Esto hace que el dataset sea **completo y desafiante**, ya que el modelo debe detectar personas en **escenarios densos** y con **oclusiones**, mejorando así la robustez del detector.

El dataset se divide en:
- **3175 imágenes para entrenamiento**
- **400 imágenes para validación**
- **300 imágenes para test**

Roboflow permite exportar el dataset en **formato YOLO**, lo que facilita su integración directa con modelos como YOLOv8.

### Dataset Summary

| Feature | Value |
|-------|-------|
| Total images | 3875 |
| Classes | 1 (person) |
| Training images | 3175 |
| Validation images | 400 |
| Test images | 300 |
| Annotation type | Bounding boxes |
| Format | YOLO |

## Demo Person Detector Outdoor-Indoor

![Detection Demo](demo-videos/demo-detector-outdoor.gif) 

![Detection Demo](demo-videos/demo-detector-indoor.gif) 

La **evaluación del modelo** sobre vídeos de prueba muestra que el detector identifica las personas presentes con **alta precisión**. Esto sugiere que las métricas de validación pueden estar penalizadas por la dificultad intrínseca del dataset, caracterizado por escenas densas, personas de pequeño tamaño y frecuentes oclusiones, más que por una falta de capacidad del modelo. Al aplicarlo a vídeos de escenas más habituales, el detector demuestra **una buena capacidad de generalización**, lo que indica que ha aprendido representaciones robustas para la detección de personas.
