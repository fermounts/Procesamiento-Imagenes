# TP5 — YOLO: Detección de objetos en tiempo real
**Instituto de Formación Técnica Superior N° 33**  
**Materia:** Técnicas de Procesamiento de Imágenes  

---

## ¿Qué es YOLO?

YOLO (You Only Look Once) es un algoritmo de detección de objetos en tiempo real. A diferencia de otros métodos que analizan la imagen en múltiples pasadas, YOLO la procesa una sola vez — de ahí su nombre — lo que lo hace extremadamente rápido.

Dado una imagen, YOLO detecta todos los objetos presentes, los clasifica y devuelve un recuadro (bounding box) alrededor de cada uno junto con un porcentaje de confianza. Por ejemplo: `person 0.91`, `bottle 0.83`.

---

## Versiones

| Versión | Estado | Características |
|---|---|---|
| YOLOv5 | Estable | Muy usado, excelente documentación |
| YOLOv8 | Producción | El más usado actualmente en empresas |
| YOLOv11 | Beta | El más nuevo y preciso |

> Dato real: la empresa UiFlou (Rosario) usa distintas versiones según el caso. Como indicó uno de sus desarrolladores: *"depende de lo que necesitemos"* — para casos simples o hardware limitado se usan versiones más livianas, para máxima precisión se usa YOLOv8 o YOLOv11.

---

## ¿Cómo funciona?

1. La imagen se divide en una grilla de celdas
2. Cada celda predice si hay un objeto en su zona y qué tipo es
3. Se calcula un score de confianza para cada predicción
4. Se eliminan las predicciones redundantes (Non-Maximum Suppression)
5. Se devuelven solo las detecciones más confiables

Todo esto ocurre en una sola pasada por la red neuronal, lo que permite procesar video en tiempo real.

---

## Caso real: UiFlou — Rosario, Argentina

UiFlou es una empresa de Rosario que desarrolla soluciones de análisis de video con IA usando YOLO. Sus servicios se dividen en 4 áreas:

### 1. Productividad
Optimiza el movimiento humano y de materiales identificando cuellos de botella e ineficiencias en procesos industriales.

### 2. Seguridad
Reduce accidentes laborales detectando posturas de riesgo, falta de equipamiento de protección o situaciones peligrosas en tiempo real.

### 3. Bienestar
Aumenta la participación en pausas activas y guía ejercicios de rehabilitación mediante IA.

### 4. Cumplimiento
Reduce costos de auditoría y reclamaciones incorrectas usando los datos capturados como evidencia.

### Industrias donde se aplica
- **Agricultura** — 10x mayor escala de información y prevención
- **Logística** — detección de personas y objetos en depósitos
- **Manufactura** — control de líneas de producción

### Flujo típico en logística

```
Cámara filma el depósito 24/7
        ↓
YOLO detecta personas y objetos
(person 0.91 — bottle 0.83)
        ↓
Analiza movimientos y posturas
(esqueleto corporal + trayectorias)
        ↓
Genera alerta o reporte
(ineficiencia, riesgo o incumplimiento)
```

---

## Entrenamiento de modelos con imágenes propias

UiFlou no usa YOLO con su configuración genérica sino que **entrena modelos propios** para cada cliente. El proceso es:

1. **Recolección** — se capturan imágenes del entorno real del cliente (depósito, fábrica, sucursal)
2. **Etiquetado** — se marca manualmente en cada imagen qué es cada objeto: persona, casco, caja, postura de riesgo
3. **Entrenamiento** — el modelo aprende a reconocer esos objetos específicos
4. **Despliegue** — se instala en las cámaras del cliente y corre en tiempo real

Esto permite detectar objetos muy específicos: un casco de un color particular, una postura de riesgo, una caja en el lugar equivocado.

---

## Otro caso de uso: reducción de fraude bancario

YOLO también se aplica en sucursales bancarias y cajeros automáticos:

| Aplicación | Descripción |
|---|---|
| Detección de skimmers | Identifica dispositivos instalados en cajeros para robar datos |
| Comportamiento sospechoso | Detecta si alguien mira el PIN de otro usuario |
| Control de acceso | Identifica personas con el rostro cubierto al ingresar |
| Objetos abandonados | Detecta bolsos u objetos sospechosos |
| Gestión de colas | Mide tiempos de espera y cantidad de cajas abiertas |

---

## Prueba de código

### Instalación

```python
!pip install ultralytics
```

### Detección de objetos con YOLOv8

```python
from ultralytics import YOLO
from PIL import Image
import matplotlib.pyplot as plt

# Cargar modelo preentrenado
model = YOLO('yolov8n.pt')

# Correr detección sobre una imagen
results = model('imagen.jpg')

# Mostrar resultados
results[0].show()

# Ver objetos detectados
for box in results[0].boxes:
    clase = results[0].names[int(box.cls)]
    confianza = float(box.conf)
    print(f"{clase}: {confianza:.2f}")
```

### Resultado esperado

```
person: 0.91
bottle: 0.83
chair: 0.76
```

---

## Conclusión

YOLO representa una de las aplicaciones más concretas del procesamiento de imágenes en la industria actual. Lo que aprendimos en los TPs anteriores — umbralización, filtros, OCR — son la base teórica sobre la que se construyen sistemas como los de UiFlou. La diferencia es que YOLO agrega una red neuronal entrenada que aprende a reconocer patrones visuales complejos de forma autónoma.

---

*IFTS N° 33 — Técnicas de Procesamiento de Imágenes*
