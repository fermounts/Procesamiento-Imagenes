# TP4 — OCR (Reconocimiento Óptico de Caracteres)
**Instituto de Formación Técnica Superior N° 33**  
**Materia:** Técnicas de Procesamiento de Imágenes  

---

## Punto 1 — Historia del OCR

El OCR (Optical Character Recognition) tiene sus orígenes en los años 20, cuando Emanuel Goldberg creó una máquina capaz de leer caracteres y convertirlos en código telegráfico. En los años 50, David Shepard desarrolló el primer sistema comercial de reconocimiento de caracteres.

Un salto importante llegó en los años 70 cuando Ray Kurzweil creó el primer OCR capaz de leer cualquier tipografía, utilizado para convertir libros impresos en audio para personas con discapacidad visual. En los años 90 aparecieron los primeros softwares de escritorio como OmniPage, que llevaron el OCR al usuario común.

Con la llegada del deep learning en los 2000-2010, la precisión mejoró enormemente, dando lugar a los servicios en la nube actuales como Google Vision API y Amazon Textract, capaces de leer escritura a mano, tablas y documentos complejos.

---

## Punto 2 — Funciones principales de OCR con Python

### `image_to_string` — Extrae el texto completo como texto plano

```python
texto = pytesseract.image_to_string(imagen, config='--psm 6 -l spa')
print(texto)
# Salida: "Hola mundo"
```

### `image_to_data` — Devuelve texto + posición + nivel de confianza

```python
datos = pytesseract.image_to_data(imagen, output_type=pytesseract.Output.DICT)
print(datos['text'])    # ['', 'Hola', 'mundo']
print(datos['conf'])    # ['-1', '95', '87']  (% de confianza; -1 indica separador interno)
```

### `image_to_boxes` — Coordenadas exactas de cada carácter

```python
boxes = pytesseract.image_to_boxes(imagen)
print(boxes)
# H 31 92 45 107 0
# o 45 92 58 107 0
# Formato: carácter x1 y1 x2 y2 página
# x1/x2: distancia en píxeles desde el borde izquierdo
# y1/y2: distancia en píxeles desde el borde inferior
```

### `image_to_osd` — Detecta orientación e idioma

```python
osd = pytesseract.image_to_osd(imagen)
print(osd)
# Orientation: 0
# Script: Latin
```

### `get_languages` — Lista los idiomas instalados

```python
idiomas = pytesseract.get_languages()
print(idiomas)
# ['eng', 'spa', 'fra']
```

---

## Punto 3 — Prueba del código con imagen real

Se utilizó como imagen de prueba una factura comercial con texto de distintos tamaños y formatos.

### Código

```python
import cv2
import pytesseract
from PIL import Image

# Cargar imagen
img = cv2.imread("factura.png")

# 1. Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Reducir ruido
gray = cv2.medianBlur(gray, 3)

# 3. Binarización (blanco y negro)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# 4. Configuración OCR
config = r'--oem 3 --psm 6 -l spa'

# 5. OCR
texto = pytesseract.image_to_string(thresh, config=config)

print("Texto detectado:")
print(texto)
```

### Resultado obtenido

El código procesó la imagen correctamente extrayendo el texto visible, incluyendo encabezados, datos del emisor, montos y número de operación.

> *(Ver captura de pantalla adjunta)*

---

## Punto 4 — Mejora del código

Se incorporó un paso de **aumento de contraste** mediante `equalizeHist()` antes de la binarización. Esto redistribuye los tonos de gris para maximizar la diferencia entre el texto y el fondo, siendo especialmente útil en facturas o documentos escaneados que presentan bajo contraste.

### Código mejorado

```python
import cv2
import pytesseract

# Cargar imagen
img = cv2.imread("factura.png")

# 1. Convertir a escala de grises
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 2. Mejora: aumentar contraste
gray = cv2.equalizeHist(gray)

# 3. Reducir ruido
gray = cv2.medianBlur(gray, 3)

# 4. Binarización
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

# 5. Configuración OCR
config = r'--oem 3 --psm 6 -l spa'

# 6. OCR
texto = pytesseract.image_to_string(thresh, config=config)

print("Texto detectado (con mejora de contraste):")
print(texto)
```

### ¿Por qué mejora?

El **contraste** es la diferencia entre las zonas claras y oscuras de una imagen. Una imagen con poco contraste tiene el texto y el fondo con tonos muy similares, lo que confunde al OCR. Al aumentarlo, el texto queda bien negro sobre fondo blanco y la detección mejora notablemente.

---

## Comparativa de herramientas OCR

| Característica | Google Vision API | Amazon Textract | Python (Tesseract) |
|---|---|---|---|
| Precisión | Muy alta | Muy alta | Media |
| Tipos de documentos | Imágenes generales | Formularios, facturas | Imágenes simples |
| Detección de tablas | Básica | Avanzada | Muy limitada |
| Costo | Pago por uso | Pago por uso | Gratis |
| Requiere internet | Sí | Sí | No |

---

*IFTS N° 33 — Técnicas de Procesamiento de Imágenes*
