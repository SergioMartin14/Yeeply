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

### Data Leakage Experiment

Para analizar el impacto del **data leakage**, se creó una segunda versión del dataset en la que se añadieron las **400 imágenes del conjunto de validación al conjunto de entrenamiento**. De esta forma, el modelo tiene acceso durante el entrenamiento a ejemplos que también aparecen en validación, simulando una situación de **fuga de información entre splits**.

Tras esta modificación, el conjunto de entrenamiento pasa a tener **3575 imágenes**, mientras que los conjuntos de validación y test se mantienen sin cambios. Este experimento permite observar cómo el *data leakage* puede producir **métricas de validación artificialmente optimistas**, ya que el modelo se evalúa sobre datos que ya ha visto durante el entrenamiento.

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

## Dataset Comparison

| Feature | Original dataset | Dataset with leakage |
|-------|------------------|----------------------|
| Total images | 3875 | 4275 |
| Classes | 1 (person) | 1 (person) |
| Training images | 3175 | 3575 |
| Validation images | 400 | 400 |
| Test images | 300 | 300 |
| Annotation type | Bounding boxes | Bounding boxes |
| Format | YOLO | YOLO |

### Dataset Comparison

| Split | Original dataset | Dataset with leakage |
|------|------------------|----------------------|
| Train | 3175 | 3575 |
| Validation | 400 | 400 |
| Test | 300 | 300 |
| **Total images** | **3875** | **4275** |

## Demo Person Detector Outdoor-Indoor

La **evaluación del modelo** sobre vídeos de prueba muestra que el detector identifica las personas presentes con **alta precisión**. Esto sugiere que las métricas de validación pueden estar penalizadas por la **dificultad intrínseca del dataset**, caracterizado por **escenas densas**, **personas de pequeño tamaño** y **frecuentes oclusiones**, más que por una falta de capacidad del modelo. Al aplicarlo a vídeos de escenas más habituales, el detector demuestra una **buena capacidad de generalización**, habiendo aprendido **representaciones robustas para la detección de personas**.

![Detection Demo](demo-videos/demo-detector-outdoor.gif) 

![Detection Demo](demo-videos/demo-detector-indoor.gif) 
