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

Roboflow permite exportar el dataset en **formato YOLO** facilitando su integración directa con modelos como YOLOv8.

### Data Leakage Experiment

Para analizar el impacto del **data leakage**, se creó una segunda versión del dataset en la que se añadieron las **400 imágenes del conjunto de validación al conjunto de entrenamiento**. De esta forma, el modelo tiene acceso durante el entrenamiento a ejemplos que también aparecen en validación, simulando una situación de **fuga de información entre splits**.

Tras esta modificación, el conjunto de entrenamiento pasa a tener **3575 imágenes**, mientras que los conjuntos de validación y test se mantienen sin cambios. Este experimento permite observar cómo el *data leakage* puede producir **métricas de validación artificialmente optimistas**, ya que el modelo se evalúa sobre datos que ya ha visto durante el entrenamiento.

### Dataset Summary
| Feature | Original dataset | Dataset with leakage |
|-------|------------------|----------------------|
| Total images | 3875 | 4275 |
| Classes | 1 (person) | 1 (person) |
| Training images | 3175 | 3575 |
| Validation images | 400 | 400 |
| Test images | 300 | 300 |
| Annotation type | Bounding boxes | Bounding boxes |
| Format | YOLO | YOLO |

## Training YOLO

Debido a la falta de disponibilidad de hardware local adecuado, los entrenamientos se realizaron en **Google Colab**, lo que permitió utilizar **aceleración por GPU**. No obstante, el entorno presenta limitaciones de tiempo de uso y las sesiones se interrumpían al alcanzar el límite de ejecución. Esto obligó a priorizar entrenamientos relativamente cortos para evitar perder el progreso del entrenamiento.

Como consecuencia de estas restricciones, se utilizó el modelo **YOLOv8n** y se limitaron el número de **épocas de entrenamiento**. Aun así, cada entrenamiento tuvo una duración aproximada de **más de dos horas**.

Se realizaron **tres entrenamientos diferentes**:

- **1. Entrenamiento base (óptimo)**: realizado con el **dataset original** y las **técnicas de data augmentation por defecto** de YOLOv8. 

- **2. Entrenamiento sin data augmentation**: realizado con el **dataset original** pero con **data augmentation completamente desactivado**.

*La idea inicial era **entrenar el modelo durante muchas más épocas que el baseline** para provocar un **overfitting claro**, permitiendo que el modelo memorizara los datos de entrenamiento. Sin embargo, debido a **las limitaciones de tiempo de ejecución de Google Colab**, no fue posible alargar el número de épocas, resultando principalmente en **menor variabilidad de datos durante el entrenamiento**.*

- **3. Entrenamiento con data leakage**: realizado con el **dataset modificado** y **data augmentation desactivado**, donde las imágenes de validación se añadieron al conjunto de entrenamiento.


### Parámetros de Data Augmentation (todos desactivados)

```python
# -- Alteración de Color (Fotometría) --
hsv_h=0.0       # Sin variación de tono
hsv_s=0.0       # Sin variación de saturación
hsv_v=0.0       # Sin variación de brillo
bgr=0.0

# -- Alteración Espacial (Geometría) --
degrees=0.0     # Sin rotación
translate=0.0   # Sin traslación
scale=0.0       # Sin escalado (zoom in/out)
shear=0.0       # Sin cizallado
perspective=0.0 # Sin distorsión

# -- Volteos --
flipud=0.0      # Sin volteo vertical
fliplr=0.0      # Sin volteo horizontal

# -- Técnicas Avanzadas de Composición --
mosaic=0.0      # Sin mosaic
mixup=0.0
copy_paste=0.0
erasing=0.0     # Sin borrado aleatorio
crop_fraction=1.0
```

A continuación se muestran las curvas de entrenamiento generadas por YOLOv8 para los tres entrenamientos realizados. Cada gráfico resume la evolución de las **losses de entrenamiento y validación**, así como las principales **métricas de evaluación** (precision, recall y mAP). Además, se incluyen las **curvas Precision–Recall (PR)** y las **matrices de confusión**.

### 1. Baseline Training (dataset original + augmentations por defecto)

![Baseline Training Results](training-images/results_base.png)
<p align="center">
  <img src="training-images/BoxPR_curve_base.png" width="44%">
  <img src="training-images/confusion_matrix_base.png" width="44%">
</p>

### 2. No Data Augmentation Training (dataset original + no augmentation)

![Baseline Training Results](training-images/results_overfit.png)
<p align="center">
  <img src="training-images/BoxPR_curve_overfit.png" width="44%">
  <img src="training-images/confusion_matrix_overfit.png" width="44%">
</p>

### 3. Data Leakage Training (dataset modificado + no augmentation)

![Baseline Training Results](training-images/results_leakage.png)
<p align="center">
  <img src="training-images/BoxPR_curve_leakage.png" width="44%">
  <img src="training-images/confusion_matrix_leakage.png" width="44%">
</p>

### 4. Interpretation

#### mAP@0.5 y mAP@0.5:0.95

Las métricas **mAP@0.5** y **mAP@0.5:0.95** miden la calidad global de las detecciones considerando el grado de solapamiento entre las *bounding boxes* predichas y las reales (IoU).

- **mAP@0.5** utiliza un umbral de IoU de 0.5, por lo que es más permisivo: basta con que la detección se aproxime razonablemente al objeto.
- **mAP@0.5:0.95** promedia resultados en múltiples umbrales de IoU (de 0.5 a 0.95), siendo una métrica **más estricta y representativa** del rendimiento real del detector.

En el **entrenamiento baseline**, ambas métricas crecen de forma progresiva, indicando que el modelo aprende a localizar correctamente a las personas.  

En el caso **sin data augmentation**, el modelo tiende a ajustarse más a los datos de entrenamiento y limita su capacidad de generalización. 

En el experimento con **data leakage**, las métricas pueden aparecer **artificialmente elevadas**, ya que el modelo está siendo evaluado con imágenes que también han sido utilizadas durante el entrenamiento. Esto genera una estimación demasiado optimista del rendimiento real.

#### Precision–Recall Curve (PR Curve)

La **curva Precision–Recall** muestra la relación entre **precision** y **recall** al variar el **umbral de confianza** de las detecciones.

Mover el umbral implica cambiar el criterio con el que el modelo decide si una detección se considera válida:

- **Umbral alto** → menos detecciones, mayor *precision*, pero menor *recall* (porque al hacer menos detecciones lo lógico es que las pocas que hace las acierte pero que se deje muchas personas sin detectar).
- **Umbral bajo** → más detecciones, mayor *recall*, pero mayor riesgo de *false positives* (porque al hacer más detecciones lo lógico es que falle más pero que no se deje personas sin detectar). 

Una curva PR más cercana a la esquina superior derecha indica un mejor equilibrio entre precisión y cobertura.

Comparando los tres entrenamientos:
- El **baseline** muestra una curva equilibrada.
- El **sin data augmentation** muestra un comportamiento menos estable, indicando menor capacidad de generalización.
- El **data leakage** muestra una curva aparentemente mejor, pero esta métrica no refleja el rendimiento real del modelo en datos no vistos.

#### Matriz de Confusión

La **matriz de confusión** permite analizar de forma directa los tipos de aciertos y errores del modelo. 

- **True Positives (TP)**: personas detectadas correctamente.
- **False Positives (FP)**: detecciones incorrectas (el modelo detecta una persona donde no la hay).
- **False Negatives (FN)**: personas presentes en la imagen que el modelo no detecta.

Al comparar los tres entrenamientos, en el caso de **sin data augmentation** aparecen más errores en validación debido a la menor capacidad de generalización. En el **experimento con data leakage**, la matriz muestra menos errores de los que realmente existirían en un escenario real, ya que parte de los datos de validación ya fueron vistos durante el entrenamiento.

## Test-set Evaluation (Comparing the 3 trained models)

Para comparar de forma justa los **tres modelos** (baseline, sin data augmentation y leakage) se evaluan todos sobre el **subset de test**, que no se usó durante el entrenamiento ni la validación. Esta evaluación es clave para **desenmascarar el modelo con data leakage**, ya que un leakage suele inflar métricas en validación, pero **no debería traducirse en una mejora real en test**.

### Val Metrics Comparison

| Model | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 |
|------|--------------|---------|---------|
| Baseline | 0.4053 | 0.6812 | 0.4138 |
| No augm | 0.3017 | 0.5662 | 0.2850 |
| Leakage | 0.5273 | 0.7664 | 0.5869 |

### Test Metrics Comparison

| Model | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 |
|------|--------------|---------|---------|
| Baseline | 0.3022 | 0.5687 | 0.2804 |
| No augm | 0.2157 | 0.4521 | 0.1792 |
| Leakage | 0.2191 | 0.4635 | 0.1842 |


El **modelo baseline** obtiene el mejor rendimiento en el conjunto de **test** en todas las métricas. En **mAP@0.5**, el baseline alcanza **0.5687**, mientras que el modelo con **sin data augmentation** obtiene **0.4521** y el modelo con **data leakage** **0.4635**. 

Al comparar las métricas entre **validación (val)** y **test**, se observan diferencias claras en los tres modelos:

- El **baseline** pasa de **0.6812 en validación** a **0.5687 en test**, con una diferencia de **-0.1125**, lo que indica una ligera pérdida de rendimiento al generalizar a datos no vistos.
- El **modelo sin data augmentation** pasa de **0.5662 en validación** a **0.4521 en test**, una caída de **-0.1141**, mostrando un comportamiento similar al baseline pero con peores métricas generales lo que confirma que la ausencia de *data augmentation* reduce la capacidad de generalización.
- El **modelo con data leakage** muestra la mayor diferencia: **0.7664 en validación** frente a **0.4635 en test**, una caída de **-0.3029**, lo que indica que las métricas de validación estaban infladas artificialmente debido al leakage.

En conjunto, el **baseline mantiene el mejor rendimiento real en test**, mientras que el **modelo con leakage aparenta ser mejor en validación pero no generaliza correctamente**, evidenciando el efecto negativo del data leakage.


## Demo Person Detector Outdoor-Indoor (Baseline Model)

La **evaluación del modelo** sobre vídeos de prueba muestra que el detector identifica las personas presentes con **alta precisión**. Esto sugiere que las métricas anteriores pueden estar penalizadas por la **dificultad intrínseca del dataset**, caracterizado por **escenas densas**, **personas de pequeño tamaño** y **frecuentes oclusiones**, más que por una falta de capacidad del modelo. Al aplicarlo a vídeos de escenas más habituales, el detector demuestra una **buena capacidad de generalización**, habiendo aprendido **representaciones robustas para la detección de personas**.

![Detection Demo](demo-videos/demo-detector-outdoor.gif) 

![Detection Demo](demo-videos/demo-detector-indoor.gif) 

## Limitaciones y trabajo futuro

Debido a las **limitaciones de tiempo y recursos de hardware disponibles**, el alcance de los experimentos ha sido necesariamente limitado. Todos los entrenamientos se han ejecutado en **Google Colab**, lo que impone restricciones tanto en **tiempo máximo de ejecución** como en **capacidad de GPU**. Aun así, los experimentos realizados permiten **entender el comportamiento del modelo ante diferentes configuraciones de entrenamiento** (baseline, sin data augmentation y con data leakage), así como identificar **qué factores afectan más a la generalización del modelo**. 

### TO DO 

- Utilizar **modelos de YOLO más grandes** que mejoren el desempeño global de la solución.
- Incorporar **ByteTrack** que permita mantener un **tracking continuo de los objetos detectados (personas)**.
- Evaluar **YOLO26** siendo el **estado del arte (SOTA)** de este tipo de arquitectura.
- Probar **diferentes configuraciones de data augmentation** distintas a la utilizada en el baseline (*mixup*, *erasing*, *copy_paste*...).
- Integrar **herramientas MLOps centradas en el seguimiento de experimentos**, como **ClearML** o **W&B**.


