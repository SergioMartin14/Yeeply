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

- **Entrenamiento base (óptimo)**: realizado con el **dataset original** y las **técnicas de data augmentation por defecto** de YOLOv8. 

- **Entrenamiento con overfitting**: realizado con el **dataset original** y **data augmentation pobre**. Debido a las limitaciones de tiempo de Google Colab no fue posible realizar un entrenamiento más largo.

- **Entrenamiento con data leakage**: realizado con el **dataset modificado** y **data augmentation pobre**, donde las imágenes de validación se añadieron al conjunto de entrenamiento.

A continuación se muestran las curvas de entrenamiento generadas por YOLOv8 para los tres entrenamientos realizados. Cada gráfico resume la evolución de las **losses de entrenamiento y validación**, así como las principales **métricas de evaluación** (precision, recall y mAP). Además, se incluyen las **curvas Precision–Recall (PR)** y las **matrices de confusión**.

### 1. Baseline Training (dataset original + augmentations por defecto)

![Baseline Training Results](training-images/results_base.png)
<p align="center">
  <img src="training-images/BoxPR_curve_base.png" width="44%">
  <img src="training-images/confusion_matrix_base.png" width="44%">
</p>

### 2. Overfitting Training (dataset original + augmentation reducido)

![Baseline Training Results](training-images/results_overfit.png)
<p align="center">
  <img src="training-images/BoxPR_curve_overfit.png" width="44%">
  <img src="training-images/confusion_matrix_overfit.png" width="44%">
</p>

### 3. Data Leakage Training (dataset modificado)

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

En el caso de **overfitting**, el modelo tiende a ajustarse más a los datos de entrenamiento (al no haber data augmentation) y limita su capacidad de generalización. 

En el experimento con **data leakage**, las métricas pueden aparecer **artificialmente elevadas**, ya que el modelo está siendo evaluado con imágenes que también han sido utilizadas durante el entrenamiento. Esto genera una estimación demasiado optimista del rendimiento real.

#### Precision–Recall Curve (PR Curve)

La **curva Precision–Recall** muestra la relación entre **precision** y **recall** al variar el **umbral de confianza** de las detecciones.

Mover el umbral implica cambiar el criterio con el que el modelo decide si una detección se considera válida:

- **Umbral alto** → menos detecciones, mayor *precision*, pero menor *recall* (porque al hacer menos detecciones lo lógico es que las pocas que hace las acierte).
- **Umbral bajo** → más detecciones, mayor *recall*, pero mayor riesgo de *false positives* (porque al hacer más detecciones lo lógico es que falle más). 

Una curva PR más cercana a la esquina superior derecha indica un mejor equilibrio entre precisión y cobertura.

Comparando los tres entrenamientos:
- El **baseline** muestra una curva equilibrada.
- El **overfitting** muestra un comportamiento menos estable, indicando menor capacidad de generalización.
- El **data leakage** muestra una curva aparentemente mejor, pero esta métrica no refleja el rendimiento real del modelo en datos no vistos.

#### Matriz de Confusión

La **matriz de confusión** permite analizar de forma directa los tipos de aciertos y errores del modelo. 

- **True Positives (TP)**: personas detectadas correctamente.
- **False Positives (FP)**: detecciones incorrectas (el modelo detecta una persona donde no la hay).
- **False Negatives (FN)**: personas presentes en la imagen que el modelo no detecta.

Al comparar los tres entrenamientos, en el caso de **overfitting** aparecen más errores en validación debido a la menor capacidad de generalización. En el **experimento con data leakage**, la matriz muestra menos errores de los que realmente existirían en un escenario real, ya que parte de los datos de validación ya fueron vistos durante el entrenamiento.

## Test-set Evaluation (Comparing the 3 trained models)

Para comparar de forma justa los **tres modelos** (baseline, overfitting y leakage) se evaluan todos sobre el **subset de test**, que no se usó durante el entrenamiento ni la validación. Esta evaluación es clave para **desenmascarar el modelo con data leakage**, ya que un leakage suele inflar métricas en validación, pero **no debería traducirse en una mejora real en test**.

### Val Metrics Comparison

| Model | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 |
|------|--------------|---------|---------|
| Baseline | 0.4053 | 0.6812 | 0.4138 |
| Overfitting | 0.3017 | 0.5662 | 0.2850 |
| Leakage | 0.5273 | 0.7664 | 0.5869 |

### Test Metrics Comparison

| Model | mAP@0.5:0.95 | mAP@0.5 | mAP@0.75 |
|------|--------------|---------|---------|
| Baseline | 0.3022 | 0.5687 | 0.2804 |
| Overfitting | 0.2157 | 0.4521 | 0.1792 |
| Leakage | 0.2191 | 0.4635 | 0.1842 |





El **modelo baseline** obtiene los mejores resultados en todas las métricas, lo que indica una mejor capacidad de **generalización** al utilizar el dataset original junto con las augmentations por defecto.

El modelo **overfitting** presenta una caída clara en mAP, lo que sugiere que el modelo se ajustó demasiado a los datos de entrenamiento al no utilizar data augmentation.

El modelo con **data leakage** muestra métricas ligeramente mejores que el overfitting, pero sigue siendo inferior al baseline. Esto confirma que el *data leakage* puede inflar artificialmente las métricas durante validación, pero **no mejora el rendimiento real en el conjunto de test**.

### Interpretación de Resultados

El **modelo baseline** obtiene el mejor rendimiento en el conjunto de **test** en todas las métricas. En **mAP@0.5**, el baseline obtiene **0.5687**, mientras que el modelo overfitting alcanza **0.4521** y el de leakage **0.4635**, lo que representa caídas de **0.1166** y **0.1052** respectivamente.  

Al comparar las métricas entre **validación (val)** y **test**, también se observan diferencias importantes:

- El **modelo baseline** obtiene **0.4053 mAP@0.5:0.95 en validación** frente a **0.3022 en test**, una diferencia de **-0.1031**, lo que indica un pequeño gap de generalización pero un comportamiento relativamente consistente.
- El **modelo con overfitting** pasa de **0.3017 en validación** a **0.2157 en test**, una caída de **-0.0860**, lo que confirma que la ausencia de *data augmentation* reduce la capacidad de generalización.
- El **modelo con data leakage** muestra la mayor discrepancia: **0.5273 en validación** frente a **0.2191 en test**, con una diferencia de **-0.3082**, lo que indica que las métricas de validación estaban fuertemente infladas debido al leakage.

En conjunto, los resultados muestran que **el modelo baseline ofrece el rendimiento más robusto y generalizable**, mientras que **el overfitting reduce el rendimiento en test** y **el data leakage puede inflar artificialmente las métricas de validación sin mejorar el rendimiento real en el conjunto de test**.



## Demo Person Detector Outdoor-Indoor (Baseline Model)

La **evaluación del modelo** sobre vídeos de prueba muestra que el detector identifica las personas presentes con **alta precisión**. Esto sugiere que las métricas anteriores pueden estar penalizadas por la **dificultad intrínseca del dataset**, caracterizado por **escenas densas**, **personas de pequeño tamaño** y **frecuentes oclusiones**, más que por una falta de capacidad del modelo. Al aplicarlo a vídeos de escenas más habituales, el detector demuestra una **buena capacidad de generalización**, habiendo aprendido **representaciones robustas para la detección de personas**.

![Detection Demo](demo-videos/demo-detector-outdoor.gif) 

![Detection Demo](demo-videos/demo-detector-indoor.gif) 
