# TP6 — Detección de Postura Corporal con MediaPipe
**Instituto de Formación Técnica Superior N° 33**  
**Materia:** Técnicas de Procesamiento de Imágenes  

---

## ¿Qué es MediaPipe?

MediaPipe es una librería de Google para el procesamiento de video e imágenes en tiempo real. A diferencia de YOLO que detecta objetos en general, MediaPipe se especializa en el cuerpo humano: detecta y rastrea **33 puntos de referencia 3D** del cuerpo (landmarks) a partir de fotogramas RGB.

---

## MediaPipe Pose — BlazePose

El módulo de detección de postura usa la topología **BlazePose**, un superconjunto del estándar COCO que optimiza la detección facial, de manos y corporal simultáneamente.

Mapea con precisión 33 articulaciones críticas desde la cabeza hasta los pies:

| Nº | Punto | Nº | Punto |
|---|---|---|---|
| 0 | Nariz | 17 | Nudillo meñique derecho |
| 1 | Ojo interno derecho | 18 | Nudillo meñique izquierdo |
| 7 | Oreja derecha | 11 | Hombro derecho |
| 8 | Oreja izquierda | 12 | Hombro izquierdo |
| 13 | Codo derecho | 14 | Codo izquierdo |
| 15 | Muñeca derecha | 16 | Muñeca izquierda |
| 23 | Cadera derecha | 24 | Cadera izquierda |
| 25 | Rodilla derecha | 26 | Rodilla izquierda |
| 27 | Tobillo derecho | 28 | Tobillo izquierdo |

---

## Objetivo de la aplicación

El sistema detecta a una persona desde una **vista lateral** para medir la inclinación del cuello y del torso respecto a un eje de referencia vertical. Alerta al usuario cuando la inclinación supera un umbral saludable.

---

## Flujo de trabajo

```
Captura RGB
Adquisición y conversión de fotogramas en RGB
        ↓
Procesamiento
Detección y extracción de landmarks clave
        ↓
Verificación
Alineación de cámara y cálculo de ángulos posturales
        ↓
Evaluación
Clasificación de postura y envío de alertas
```

---

## Alineación de la cámara

Para un análisis preciso, el usuario debe estar **de perfil**. El sistema mide la distancia horizontal (offset) entre el hombro izquierdo y el derecho:

- **Vista lateral correcta:** los puntos de los hombros casi coinciden, el offset está por debajo del umbral → análisis válido
- **Vista incorrecta:** offset alto indica que la cámara no está perpendicular al usuario → se muestra advertencia

---

## Cálculo de ángulos de inclinación

Se calcula el ángulo que forman la línea del cuello y la línea del torso respecto al eje vertical, usando tres puntos clave: **oreja, hombro y cadera**.

### Fórmula vectorial

$$\theta = \arccos\left(\frac{y_1^2 - y_1 \cdot y_2}{y_1 \sqrt{(x_2-x_1)^2 + (y_2-y_1)^2}}\right)$$

Donde:
- $P_1$ = hombro (punto central)
- $P_2$ = oreja (para el cuello) o cadera (para el torso)
- $P_3$ = punto de referencia vertical

---

## Lógica de evaluación de postura

| Estado | Condición | Indicador visual |
|---|---|---|
| ✅ Postura correcta | Cuello < 40° y torso < 10° | Líneas verdes |
| ⚠️ Postura incorrecta | Se superan los umbrales | Líneas rojas + contador |
| 🚨 Alerta activa | Mala postura > 3 minutos (180s) | Se dispara la advertencia |

---

## Configuración de la API

```python
import mediapipe as mp

mp_pose = mp.solutions.pose

pose = mp_pose.Pose(
    static_image_mode=False,        # False para video (rastrea continuamente)
    model_complexity=1,             # 0=rápido, 1=balanceado, 2=preciso
    enable_segmentation=False,      # True para máscara de fondo
    min_detection_confidence=0.5,   # Umbral mínimo de detección inicial
    min_tracking_confidence=0.5     # Umbral mínimo de seguimiento
)
```

---

## Código completo

```python
import cv2
import mediapipe as mp
import numpy as np
import time

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def calcular_angulo(p1, p2, p3):
    """Calcula el ángulo entre tres puntos."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    angulo = np.degrees(np.arctan2(y3 - y2, x3 - x2) -
                        np.arctan2(y1 - y2, x1 - x2))
    if angulo < 0:
        angulo += 360
    return angulo

# Captura de video (0 = cámara, o ruta de video)
cap = cv2.VideoCapture(0)

tiempo_mala_postura = 0
inicio_mala_postura = None

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        imagen_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = pose.process(imagen_rgb)

        if resultados.pose_landmarks:
            landmarks = resultados.pose_landmarks.landmark
            h, w, _ = frame.shape

            # Obtener puntos clave
            oreja   = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].x * w),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_EAR].y * h))
            hombro  = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
            cadera  = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * w),
                       int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * h))

            # Calcular ángulos
            ang_cuello = calcular_angulo(oreja, hombro, (hombro[0], 0))
            ang_torso  = calcular_angulo(hombro, cadera, (cadera[0], 0))

            # Evaluar postura
            if ang_cuello < 40 and ang_torso < 10:
                color = (0, 255, 0)  # verde
                if inicio_mala_postura:
                    inicio_mala_postura = None
            else:
                color = (0, 0, 255)  # rojo
                if inicio_mala_postura is None:
                    inicio_mala_postura = time.time()
                else:
                    tiempo_mala_postura = time.time() - inicio_mala_postura
                    if tiempo_mala_postura > 180:
                        cv2.putText(frame, '¡ALERTA! Corregí tu postura',
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

            # Dibujar líneas
            cv2.line(frame, oreja, hombro, color, 2)
            cv2.line(frame, hombro, cadera, color, 2)

            # Mostrar ángulos
            cv2.putText(frame, f'Cuello: {int(ang_cuello)}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f'Torso: {int(ang_torso)}',
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow('Detector de Postura', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
```

---

## Comparativa: OpenCV vs YOLO vs MediaPipe

| Característica | OpenCV Haar | YOLO | MediaPipe |
|---|---|---|---|
| Detecta | Rostros | Objetos en general | Cuerpo humano (33 puntos) |
| Precisión | Media | Alta | Muy alta |
| Velocidad | Muy rápida | Rápida | Rápida |
| Requiere GPU | No | Recomendado | No |
| Caso de uso | Vigilancia básica | Logística, seguridad | Bienestar, fitness, postura |

---

## Aplicaciones futuras

- **Filtros de realidad aumentada** — estilo Instagram o Snapchat
- **Control por gestos** — manejar videollamadas con movimientos de mano
- **Detección de somnolencia** — seguridad vial, detectar si un conductor se queda dormido
- **Rehabilitación física** — guiar ejercicios y corregir movimientos

---

*IFTS N° 33 — Técnicas de Procesamiento de Imágenes*
