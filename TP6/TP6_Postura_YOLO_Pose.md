# TP6 — Detección de Postura Corporal con YOLO Pose
**Instituto de Formación Técnica Superior N° 33**  
**Materia:** Técnicas de Procesamiento de Imágenes  

---

## ¿Qué es YOLO Pose?

YOLO Pose es una variante de YOLOv8 especializada en la detección del cuerpo humano. A diferencia de YOLO estándar que detecta objetos en general, YOLO Pose detecta y rastrea **17 puntos clave del cuerpo** (keypoints) en tiempo real a partir de imágenes o video.

> **Nota:** Se evaluó el uso de MediaPipe (librería de Google) para esta tarea, pero presenta conflictos de compatibilidad con las versiones actuales de Colab y TensorFlow. YOLO Pose ofrece la misma funcionalidad con mayor estabilidad.

---

## Puntos clave detectados (17 keypoints)

| Nº | Punto | Nº | Punto |
|---|---|---|---|
| 0 | Nariz | 9 | Muñeca izquierda |
| 1 | Ojo izquierdo | 10 | Muñeca derecha |
| 2 | Ojo derecho | 11 | Cadera izquierda |
| 3 | Oreja izquierda | 12 | Cadera derecha |
| 4 | Oreja derecha | 13 | Rodilla izquierda |
| 5 | Hombro izquierdo | 14 | Rodilla derecha |
| 6 | Hombro derecho | 15 | Tobillo izquierdo |
| 7 | Codo izquierdo | 16 | Tobillo derecho |
| 8 | Codo derecho | | |

---

## Objetivo de la aplicación

El sistema detecta a una persona desde una **vista lateral** para medir la inclinación del cuello y del torso respecto a un eje de referencia vertical. Clasifica la postura como correcta o incorrecta según umbrales de ángulo.

---

## Flujo de trabajo

```
Captura RGB
Adquisición de la imagen o fotograma
        ↓
Detección
YOLO Pose detecta personas y extrae 17 keypoints
        ↓
Cálculo de ángulos
Se mide la inclinación del cuello y el torso
        ↓
Evaluación
Clasificación de postura correcta o incorrecta
```

---

## Cálculo de ángulos de inclinación

Se calcula el ángulo que forman la línea del cuello y la línea del torso respecto al eje vertical, usando tres puntos clave: **oreja, hombro y cadera**.

### Fórmula

$$\theta = \arctan\left(\frac{|x_2 - x_1|}{|y_2 - y_1|}\right)$$

---

## Lógica de evaluación de postura

| Estado | Condición | Resultado |
|---|---|---|
| ✅ Postura correcta | Cuello < 40° y torso < 10° | Postura saludable |
| ⚠️ Postura incorrecta | Se superan los umbrales | Se recomienda corregir |

---

## Código principal

```python
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n-pose.pt')

def calcular_angulo_vertical(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dy == 0:
        return 90
    return np.degrees(np.arctan2(dx, dy))

results = model('postura.jpg')

for i, persona in enumerate(results[0].keypoints):
    kps = persona.data[0].numpy()
    oreja  = (kps[3][0], kps[3][1])
    hombro = (kps[5][0], kps[5][1])
    cadera = (kps[11][0], kps[11][1])

    ang_cuello = calcular_angulo_vertical(hombro, oreja)
    ang_torso  = calcular_angulo_vertical(cadera, hombro)

    print(f'Persona {i+1}:')
    print(f'  Cuello: {ang_cuello:.1f}°')
    print(f'  Torso:  {ang_torso:.1f}°')

    if ang_cuello < 40 and ang_torso < 10:
        print('  ✅ POSTURA CORRECTA')
    else:
        print('  ⚠️ POSTURA INCORRECTA')
```

---

## Comparativa: OpenCV vs YOLO vs YOLO Pose

| Característica | OpenCV Haar | YOLO | YOLO Pose |
|---|---|---|---|
| Detecta | Rostros | Objetos en general | 17 puntos del cuerpo |
| Precisión | Media | Alta | Muy alta |
| Velocidad | Muy rápida | Rápida | Rápida |
| Requiere GPU | No | Recomendado | Recomendado |
| Caso de uso | Vigilancia básica | Logística, seguridad | Postura, fitness, bienestar |

---

## Aplicaciones futuras

- **Filtros de realidad aumentada** — estilo Instagram o Snapchat
- **Control por gestos** — manejar videollamadas con movimientos de mano
- **Detección de somnolencia** — seguridad vial
- **Rehabilitación física** — guiar ejercicios y corregir movimientos

---

*IFTS N° 33 — Técnicas de Procesamiento de Imágenes*
