# TP3 — Umbralización y Eliminación de Ruido
**Instituto de Formación Técnica Superior N° 33**  
**Materia:** Técnicas de Procesamiento de Imágenes  

---

## Introducción

La **umbralización** es una técnica básica de segmentación que convierte una imagen en escala de grises en una imagen binaria (blanco y negro), separando objetos del fondo según un valor límite (threshold).

---

## Punto 1 — Prueba de Umbralización

Se ejecutó el código provisto utilizando como imagen de prueba una fotografía en escala de grises. Se aplicaron las tres técnicas de umbralización:

- **Umbral Fijo (Manual):** se utilizó un valor de `threshold = 127`. El resultado muestra una separación clara entre zonas claras y oscuras, pero pierde detalle en las zonas de sombra.
- **Umbral de Otsu (Automático):** el algoritmo calculó automáticamente el umbral óptimo analizando el histograma de la imagen, obteniendo un resultado muy similar al manual ya que la imagen tiene una distribución de grises balanceada.
- **Umbral Adaptativo:** calcula un umbral diferente para cada región de la imagen. Es el que mejor preserva los detalles finos como texturas y bordes, siendo ideal para imágenes con iluminación irregular.

### Código

```python
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

def procesar_imagen(ruta_imagen):
    if not os.path.exists(ruta_imagen):
        print(f"Error: No se encontró el archivo '{ruta_imagen}'")
        return

    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    desenfocada = cv2.GaussianBlur(imagen, (5, 5), 0)

    # A. Umbral Fijo (Manual)
    _, th_fijo = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY)

    # B. Umbral de Otsu (Automático)
    _, th_otsu = cv2.threshold(desenfocada, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # C. Umbral Adaptativo
    th_adapt = cv2.adaptiveThreshold(desenfocada, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 11, 2)

    titulos = ['Original', 'Fijo (Manual)', 'Otsu (Auto)', 'Adaptativo']
    imagenes = [imagen, th_fijo, th_otsu, th_adapt]

    plt.figure(figsize=(15, 5))
    for i in range(4):
        plt.subplot(1, 4, i + 1)
        plt.imshow(imagenes[i], cmap='gray')
        plt.title(titulos[i], fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('resultado.png', dpi=150, bbox_inches='tight')
    print("Imagen guardada como resultado.png")

procesar_imagen('documento.jpg')
```

---

## Punto 2 — Fórmulas Matemáticas para Eliminar Ruido

Se utilizan principalmente cuatro filtros:

### 1. Filtro de Media
Reemplaza cada píxel con el promedio de su vecindad:

$$g(x,y) = \frac{1}{n^2} \sum_{(i,j) \in \Omega} f(i,j)$$

### 2. Filtro Gaussiano
Promedio ponderado donde los píxeles más cercanos al centro tienen más peso:

$$G(x,y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}}$$

> Es el que usa el código del TP con `cv2.GaussianBlur()`.

### 3. Filtro de Mediana
Ordena los píxeles de la vecindad y toma el valor del centro. Es el más efectivo contra el ruido sal y pimienta.

### 4. Filtro Bilateral
Suaviza preservando los bordes, combinando cercanía espacial y similitud de intensidad:

$$g(x,y) = \frac{\sum f(i,j) \cdot w_s \cdot w_r}{\sum w_s \cdot w_r}$$

### Código — Comparación de filtros

```python
def comparar_filtros(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    con_ruido = agregar_sal_pimienta(imagen, cantidad=0.05)

    filtro_media     = cv2.blur(con_ruido, (5, 5))
    filtro_gaussiano = cv2.GaussianBlur(con_ruido, (5, 5), 0)
    filtro_mediana   = cv2.medianBlur(con_ruido, 5)
    filtro_bilateral = cv2.bilateralFilter(con_ruido, 9, 75, 75)

    titulos = ['Con ruido', 'Filtro Media', 'Filtro Gaussiano', 'Filtro Mediana', 'Filtro Bilateral']
    imagenes = [con_ruido, filtro_media, filtro_gaussiano, filtro_mediana, filtro_bilateral]

    plt.figure(figsize=(18, 5))
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(imagenes[i], cmap='gray')
        plt.title(titulos[i], fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('resultado_filtros.png', dpi=150, bbox_inches='tight')
    print("Imagen guardada como resultado_filtros.png")

comparar_filtros('documento.jpg')
```

---

## Punto 3 — Ruido Sal y Pimienta

El **ruido sal y pimienta** es un tipo de ruido impulsivo que aparece como píxeles completamente blancos (`255` = "sal") o completamente negros (`0` = "pimienta") distribuidos aleatoriamente sobre la imagen.

| Característica | Descripción |
|---|---|
| **Origen** | Fallas en sensores, errores de transmisión o píxeles defectuosos del hardware |
| **Apariencia** | Manchas blancas y negras dispersas aleatoriamente |
| **Efecto en umbralización** | Muy problemático: un píxel blanco en zona oscura puede detectarse erróneamente como un objeto |
| **Mejor filtro** | Filtro de mediana, porque los valores extremos (0 ó 255) quedan en los extremos del ordenamiento y no son elegidos como mediana |

### Código — Demostración de ruido sal y pimienta

```python
def agregar_sal_pimienta(imagen, cantidad=0.05):
    resultado = imagen.copy()
    total_pixeles = imagen.size

    # Agregar sal (píxeles blancos)
    num_sal = int(total_pixeles * cantidad)
    coords_sal = [np.random.randint(0, i, num_sal) for i in imagen.shape]
    resultado[coords_sal[0], coords_sal[1]] = 255

    # Agregar pimienta (píxeles negros)
    num_pimienta = int(total_pixeles * cantidad)
    coords_pimienta = [np.random.randint(0, i, num_pimienta) for i in imagen.shape]
    resultado[coords_pimienta[0], coords_pimienta[1]] = 0

    return resultado

def demostrar_sal_pimienta(ruta_imagen):
    imagen = cv2.imread(ruta_imagen, cv2.IMREAD_GRAYSCALE)
    con_ruido = agregar_sal_pimienta(imagen, cantidad=0.05)
    sin_ruido = cv2.medianBlur(con_ruido, 3)

    titulos = ['Original', 'Con ruido sal y pimienta', 'Filtro de mediana aplicado']
    imagenes = [imagen, con_ruido, sin_ruido]

    plt.figure(figsize=(15, 5))
    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.imshow(imagenes[i], cmap='gray')
        plt.title(titulos[i], fontsize=10)
        plt.axis('off')

    plt.tight_layout()
    plt.savefig('resultado_sal_pimienta.png', dpi=150, bbox_inches='tight')
    print("Imagen guardada como resultado_sal_pimienta.png")

demostrar_sal_pimienta('documento.jpg')
```

---

*IFTS N° 33 — Técnicas de Procesamiento de Imágenes*# Procesamiento-Imagenes
