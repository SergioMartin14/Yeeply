from ultralytics import YOLO
from clearml import Task
import os

os.environ["CLEARML_API_ACCESS_KEY"] = "PLDTER5YZYBL2ZPJ84XZ0SG0ET5ALT"
os.environ["CLEARML_API_SECRET_KEY"] = "F3f7JjrgP17s3ApknasqKtsp13LS-zTSLGZfvebQ4KbLjZFRPK-wfVOUaA6VErZtAHk"
os.environ["CLEARML_API_HOST"] = "https://api.clear.ml"
os.environ["CLEARML_WEB_HOST"] = "https://app.clear.ml"
os.environ["CLEARML_FILES_HOST"] = "https://files.clear.ml"

def main():
    # ==========================================
    # 1. INTEGRACIÓN CON CLEARML
    # ==========================================
    # Inicializamos la tarea en ClearML. Ultralytics detectará esto automáticamente
    # y enviará las métricas, gráficas y pesos al dashboard.
    task = Task.init(
        project_name="YOLO_Vitaminado_Project",
        task_name="Entrenamiento_Avanzado",
        tags=["YOLO", "DataAugmentation", "Experiment_1"]
    )

    # ==========================================
    # 2. CARGA DEL MODELO
    # ==========================================
    # Puedes cargar un modelo pre-entrenado (recomendado) o uno desde cero (.yaml)
    model = YOLO("yolo26m.pt") # Cambia a yolov8s.pt, yolov11m.pt, etc. según necesites

    # ==========================================
    # 3. PARÁMETROS DE ENTRENAMIENTO "VITAMINADOS"
    # ==========================================
    resultados = model.train(
        # --- Parámetros Básicos ---
        data="/home/training/mariohernandez/ultralytics/data.yaml", # TU DATASET AQUÍ
        epochs=300,                    # Épocas máximas
        patience=10,                   # Early stopping: para si no mejora en 50 épocas
        batch=-1,                      # Tamaño del batch (ajústalo a tu VRAM, o usa -1 para AutoBatch)
        imgsz=640,                     # Tamaño de imagen
        device=0,                      # '0' para GPU, 'cpu' para CPU, o '0,1,2' para Multi-GPU
        workers=8,                     # Hilos para cargar datos (bájalo si satura tu RAM/CPU)

        # --- Optimizadores y Regularización Avanzada ---
        optimizer="auto",              # Opciones: 'SGD', 'Adam', 'AdamW', 'RMSProp'
        lr0=0.01,                      # Learning rate inicial
        lrf=0.01,                      # Learning rate final (como fracción de lr0)
        momentum=0.937,                # Momentum del optimizador
        weight_decay=0.0005,           # Penalización L2 para evitar sobreajuste
        warmup_epochs=3.0,             # Épocas de calentamiento del LR
        cos_lr=True,                   # Usar scheduler de learning rate por coseno (muy recomendado)
        close_mosaic=10,               # Desactiva mosaic en las últimas N épocas (mejora la precisión final)
        amp=True,                      # Automatic Mixed Precision (acelera el entreno en GPUs modernas)
        dropout=0.1,                   # Dropout para clasificación/redes densas (si aplica)

        # ==========================================
        # 4. DATA AUGMENTATION (EL NÚCLEO VITAMINADO)
        # ==========================================
        # Valores de 0.0 a 1.0 son probabilidades. Otros son rangos o grados.

        # -- Alteración de Color (Fotometría) --
        hsv_h=0.015,     # Variación del tono (Hue)
        hsv_s=0.7,       # Variación de la saturación
        hsv_v=0.4,       # Variación del brillo (Value)
        bgr=0.0,         # Probabilidad de invertir canales a BGR (0.0 = desactivado)

        # -- Alteración Espacial (Geometría) --
        degrees=10.0,    # Rotación aleatoria en grados (+/-)
        translate=0.1,   # Traslación de la imagen (fracción)
        scale=0.5,       # Escalado (+/- ganancia)
        shear=2.0,       # Cizallado en grados (Shear)
        perspective=0.0, # Distorsión de perspectiva (0.0 a 0.001)

        # -- Volteos --
        flipud=0.0,      # Probabilidad de volteo vertical (arriba a abajo)
        fliplr=0.5,      # Probabilidad de volteo horizontal (izquierda a derecha)

        # -- Técnicas Avanzadas de Composición --
        mosaic=0.0,      # Probabilidad de combinar 4 imágenes en 1 (Excelente para objetos pequeños)
        mixup=0.0,       # Probabilidad de superponer 2 imágenes con transparencia
        copy_paste=0.1,  # Probabilidad de copiar objetos segmentados y pegarlos en otra imagen
        erasing=0.4,     # Probabilidad de borrar partes aleatorias (Random Erasing)
        crop_fraction=1.0, # Fracción de recorte de la imagen (para clasificación)

        # ==========================================
        # 5. GUARDADO Y LOGGING (MÁS ALLÁ DE CLEARML)
        # ==========================================
        save=True,       # Guardar pesos y gráficas
        save_period=-1,  # Guardar cada N épocas (-1 = solo el mejor y el último)
        cache=False,     # Poner en True ('ram' o 'disk') si tienes RAM de sobra, acelera el entreno
        deterministic=True, # Hace que el entreno sea reproducible
        seed=42          # Semilla para la reproducibilidad
    )

    print("¡Entrenamiento finalizado con éxito! Revisa tu dashboard de ClearML.")

if __name__ == "__main__":
    main()