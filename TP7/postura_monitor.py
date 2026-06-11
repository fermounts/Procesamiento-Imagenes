"""
Monitor de postura en tiempo real
----------------------------------
- Cámara lateral (perfil derecho)
- Guía de encuadre para posicionarse
- Tablero: cuello, torso, hombros, piernas/pies
- Cartel de aviso a los 30 segundos de mala postura
- Alarma sonora al minuto
----------------------------------
Requisitos:
    pip install mediapipe opencv-python numpy
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import urllib.request
import threading

try:
    from win10toast import ToastNotifier
    toaster = ToastNotifier()
    NOTIFICACIONES_OK = True
except ImportError:
    NOTIFICACIONES_OK = False

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ══════════════════════════════════════════════════════
# CONFIGURACIÓN
# ══════════════════════════════════════════════════════

CAMARA_ID          = 1       # 1 = DroidCam

ANCHO_VENTANA      = 1100
ALTO_VENTANA       = 680

UMBRAL_CUELLO      = 20      # > 20° = cuello adelantado
UMBRAL_TORSO       = 10      # > 10° = torso inclinado
UMBRAL_HOMBRO_X    = 30      # px: hombro más adelante que cadera
UMBRAL_RODILLA     = 80      # < 80° = piernas cruzadas o levantadas (90° = correcto)

SEGUNDOS_AVISO     = 30      # cartel grande a los 30 seg
SEGUNDOS_ALARMA    = 60      # alarma sonora al minuto
INTERVALO_RESUMEN  = 60


# ══════════════════════════════════════════════════════
# COLORES (BGR)
# ══════════════════════════════════════════════════════
VERDE       = (60,  200, 60)
ROJO        = (50,  50,  220)
AMARILLO    = (0,   200, 220)
NARANJA     = (0,   140, 255)
BLANCO      = (255, 255, 255)
GRIS_OSCURO = (40,  40,  40)
GRIS_MEDIO  = (90,  90,  90)
AZUL_CLARO  = (220, 180, 80)
FONDO_PANEL = (28,  28,  28)


# ══════════════════════════════════════════════════════
# MODELO
# ══════════════════════════════════════════════════════
MODEL_PATH = 'pose_landmarker.task'

def descargar_modelo():
    if not os.path.exists(MODEL_PATH):
        print('Descargando modelo MediaPipe...')
        urllib.request.urlretrieve(
            'https://storage.googleapis.com/mediapipe-models/'
            'pose_landmarker/pose_landmarker_heavy/float16/1/'
            'pose_landmarker_heavy.task',
            MODEL_PATH)
        print('Modelo descargado.')

descargar_modelo()


# ══════════════════════════════════════════════════════
# ALARMA
# ══════════════════════════════════════════════════════
_ultima_notif = 0
_INTERVALO_NOTIF = 35

def enviar_notificacion(titulo, mensaje):
    global _ultima_notif
    ahora = time.time()
    if ahora - _ultima_notif < _INTERVALO_NOTIF:
        return
    _ultima_notif = ahora
    print(f"[NOTIF] {titulo}: {mensaje}")
    if NOTIFICACIONES_OK:
        def _notif():
            notification.notify(
                title=titulo,
                message=mensaje,
                timeout=6
            )
        threading.Thread(target=_notif, daemon=True).start()

def sonar_alarma():
    import sys
    if sys.platform == 'win32':
        import winsound
        for _ in range(6):
            winsound.Beep(880, 400)
            time.sleep(0.15)
    else:
        for _ in range(6):
            print('', end='', flush=True)
            time.sleep(0.3)


# ══════════════════════════════════════════════════════
# CÁLCULO DE ÁNGULOS Y POSTURA
# ══════════════════════════════════════════════════════
def angulo_vertical(p1, p2):
    dx = abs(p2[0] - p1[0])
    dy = abs(p2[1] - p1[1])
    if dy == 0:
        return 90.0
    return float(np.degrees(np.arctan2(dx, dy)))


def angulo_tres_puntos(a, b, c):
    """Ángulo en el punto b formado por a-b-c."""
    v1 = np.array([a[0]-b[0], a[1]-b[1]], dtype=float)
    v2 = np.array([c[0]-b[0], c[1]-b[1]], dtype=float)
    cos_a = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cos_a, -1, 1))))


def punto_px(lm, w, h):
    return (int(lm.x * w), int(lm.y * h))


def calcular_posturas(landmarks, w, h):
    lm = landmarks

    # Perfil derecho: landmarks del lado derecho
    oreja   = punto_px(lm[8],  w, h)
    hombro  = punto_px(lm[12], w, h)
    cadera  = punto_px(lm[24], w, h)
    rodilla = punto_px(lm[26], w, h)
    tobillo = punto_px(lm[28], w, h)

    espalda_alta = (
        (oreja[0] + hombro[0]) // 2,
        (oreja[1] + hombro[1]) // 2
    )

    ang_cuello  = angulo_vertical(hombro, oreja)
    ang_torso   = angulo_vertical(cadera, hombro)
    ang_rodilla = angulo_tres_puntos(cadera, rodilla, tobillo)

    # Hombros adelantados: perfil derecho → hombro con X mayor que cadera
    desp_hombro = hombro[0] - cadera[0]

    # Pies en el piso: rodilla cerca de 90°
    # Si ang_rodilla < UMBRAL_RODILLA → piernas cruzadas o levantadas
    pies_ok = ang_rodilla >= UMBRAL_RODILLA

    cuello_ok = ang_cuello  < UMBRAL_CUELLO
    torso_ok  = ang_torso   < UMBRAL_TORSO
    hombro_ok = desp_hombro < UMBRAL_HOMBRO_X

    postura_ok = cuello_ok and torso_ok and hombro_ok and pies_ok

    return {
        'oreja':        oreja,
        'hombro':       hombro,
        'cadera':       cadera,
        'rodilla':      rodilla,
        'tobillo':      tobillo,
        'espalda_alta': espalda_alta,
        'ang_cuello':   ang_cuello,
        'ang_torso':    ang_torso,
        'ang_rodilla':  ang_rodilla,
        'desp_hombro':  desp_hombro,
        'cuello_ok':    cuello_ok,
        'torso_ok':     torso_ok,
        'hombro_ok':    hombro_ok,
        'pies_ok':      pies_ok,
        'postura_ok':   postura_ok,
    }


# ══════════════════════════════════════════════════════
# ESQUELETO
# ══════════════════════════════════════════════════════
def dibujar_esqueleto(frame, p):
    color = VERDE if p['postura_ok'] else ROJO
    color_pies = VERDE if p['pies_ok'] else ROJO

    cv2.line(frame, p['oreja'],        p['hombro'],  color,      2)
    cv2.line(frame, p['hombro'],       p['cadera'],  color,      2)
    cv2.line(frame, p['cadera'],       p['rodilla'], color_pies, 2)
    cv2.line(frame, p['rodilla'],      p['tobillo'], color_pies, 2)
    cv2.line(frame, p['espalda_alta'], p['hombro'],  NARANJA,    2)

    for pt in [p['oreja'], p['hombro'], p['cadera'],
               p['rodilla'], p['tobillo'], p['espalda_alta']]:
        cv2.circle(frame, pt, 7, color, -1)
        cv2.circle(frame, pt, 7, BLANCO, 1)

    ox, oy = p['oreja']
    hx, hy = p['hombro']
    rx, ry = p['rodilla']
    cv2.putText(frame, f"C:{p['ang_cuello']:.0f}",
                (ox+8, oy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"T:{p['ang_torso']:.0f}",
                (hx+8, hy-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    cv2.putText(frame, f"R:{p['ang_rodilla']:.0f}",
                (rx+8, ry-6), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_pies, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════
# GUÍA DE ENCUADRE
# ══════════════════════════════════════════════════════
def dibujar_guia_encuadre(frame, w, h, persona_detectada):
    overlay = frame.copy()
    x1, y1 = w//4, h//8
    x2, y2 = 3*w//4, 7*h//8
    cv2.rectangle(overlay, (x1,y1), (x2,y2), AMARILLO, 2)
    cv2.line(overlay, (w//2, y1), (w//2, y2), (80,80,80), 1)

    if not persona_detectada:
        msgs = [
            "Posicionate de perfil frente a la camara",
            "Tienen que verse: oreja, hombro, cadera, rodilla y tobillo",
            "Centrate dentro del recuadro amarillo"
        ]
        color_msg = AMARILLO
    else:
        msgs = [
            "Perfil detectado correctamente",
            "Presiona  ESPACIO  para comenzar el monitoreo",
            ""
        ]
        color_msg = VERDE

    for i, msg in enumerate(msgs):
        cv2.putText(overlay, msg, (w//2-280, h-90+i*28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, color_msg, 1, cv2.LINE_AA)

    cv2.putText(overlay, "MODO CALIBRACION  —  presiona ESPACIO para empezar",
                (w//2-290, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.52, AMARILLO, 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)


# ══════════════════════════════════════════════════════
# PANEL LATERAL
# ══════════════════════════════════════════════════════
def dibujar_panel(panel, w, h, p, t_seguida, t_total_mala, resumen, pausado):
    panel[:] = FONDO_PANEL

    cv2.putText(panel, "MONITOR DE POSTURA", (16, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLANCO, 1, cv2.LINE_AA)
    cv2.line(panel, (16,42), (w-16,42), GRIS_MEDIO, 1)

    checks = [
        ('Cuello',  p['cuello_ok'], f"{p['ang_cuello']:.0f}  /  <{UMBRAL_CUELLO}"),
        ('Torso',   p['torso_ok'],  f"{p['ang_torso']:.0f}  /  <{UMBRAL_TORSO}"),
        ('Hombros', p['hombro_ok'], 'OK' if p['hombro_ok'] else 'ADELANTADOS'),
        ('Pies',    p['pies_ok'],   f"{p['ang_rodilla']:.0f}  /  >{UMBRAL_RODILLA}"),
    ]

    y = 68
    for nombre, ok, valor in checks:
        ic = VERDE if ok else ROJO
        sim = 'OK' if ok else 'MAL'
        cv2.rectangle(panel, (12, y-18), (w-12, y+28), GRIS_OSCURO, -1)
        cv2.rectangle(panel, (12, y-18), (w-12, y+28), GRIS_MEDIO,   1)
        cv2.rectangle(panel, (12, y-18), (18,   y+28), ic,           -1)
        cv2.putText(panel, nombre, (28, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50, BLANCO, 1, cv2.LINE_AA)
        cv2.putText(panel, sim, (28, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, ic, 1, cv2.LINE_AA)
        cv2.putText(panel, valor, (w-120, y+18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS_MEDIO, 1, cv2.LINE_AA)
        y += 56

    # Estado general
    cv2.line(panel, (16,y), (w-16,y), GRIS_MEDIO, 1)
    y += 20
    etxt  = 'POSTURA CORRECTA' if p['postura_ok'] else 'POSTURA INCORRECTA'
    ecolor= VERDE if p['postura_ok'] else ROJO
    cv2.putText(panel, etxt, (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, ecolor, 1, cv2.LINE_AA)

    # Tiempo mal
    y += 36
    mins = int(t_total_mala // 60)
    segs = int(t_total_mala % 60)
    cv2.putText(panel, "Tiempo mal en sesion:", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS_MEDIO, 1, cv2.LINE_AA)
    y += 20
    cv2.putText(panel, f"{mins:02d}:{segs:02d}", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, AMARILLO, 1, cv2.LINE_AA)

    # Aviso de alarma próxima
    if t_seguida > 0 and not p['postura_ok']:
        y += 42
        cv2.line(panel, (16,y), (w-16,y), GRIS_MEDIO, 1)
        y += 18
        if t_seguida < SEGUNDOS_AVISO:
            seg_aviso = SEGUNDOS_AVISO - t_seguida
            cv2.putText(panel, f"Aviso en: {seg_aviso:.0f}s", (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, AMARILLO, 1, cv2.LINE_AA)
        elif t_seguida < SEGUNDOS_ALARMA:
            seg_alarm = SEGUNDOS_ALARMA - t_seguida
            cv2.putText(panel, f"ALARMA en: {seg_alarm:.0f}s", (16, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, ROJO, 1, cv2.LINE_AA)

    # Resumen último minuto
    y = h - 120
    cv2.line(panel, (16,y), (w-16,y), GRIS_MEDIO, 1)
    y += 18
    cv2.putText(panel, "Ultimo minuto:", (16, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS_MEDIO, 1, cv2.LINE_AA)
    y += 18
    if resumen:
        pct = resumen['pct_ok']
        bw  = w - 32
        fill = int(bw * pct / 100)
        cv2.rectangle(panel, (16,y), (16+bw, y+12), GRIS_OSCURO, -1)
        bc = VERDE if pct >= 70 else (AMARILLO if pct >= 40 else ROJO)
        cv2.rectangle(panel, (16,y), (16+fill, y+12), bc, -1)
        y += 20
        cv2.putText(panel, f"{pct:.0f}% correcto", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, bc, 1, cv2.LINE_AA)
    else:
        cv2.putText(panel, "Esperando datos...", (16, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, GRIS_MEDIO, 1, cv2.LINE_AA)

    if pausado:
        cv2.putText(panel, "PAUSADO", (16, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, AMARILLO, 2, cv2.LINE_AA)

    cv2.putText(panel, "ESPACIO=pausar  R=reset  Q=salir",
                (12, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.32, GRIS_MEDIO, 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════
# CARTEL DE ALERTA SOBRE EL FRAME
# ══════════════════════════════════════════════════════
def dibujar_alerta(frame, p, t_seguida):
    if p['postura_ok'] or t_seguida == 0:
        return

    fh, fw = frame.shape[:2]

    # Cartel chico desde el primer segundo de mala postura (esquina)
    mensajes_chicos = []
    if not p['cuello_ok']:
        mensajes_chicos.append(f"Cuello adelantado ({p['ang_cuello']:.0f})")
    if not p['torso_ok']:
        mensajes_chicos.append(f"Torso inclinado ({p['ang_torso']:.0f})")
    if not p['hombro_ok']:
        mensajes_chicos.append("Hombros adelantados")
    if not p['pies_ok']:
        mensajes_chicos.append(f"Pies no apoyados ({p['ang_rodilla']:.0f})")

    # Cartel GRANDE a los 30 segundos
    if t_seguida >= SEGUNDOS_AVISO:
        overlay = frame.copy()
        # Fondo semitransparente central
        cv2.rectangle(overlay, (fw//6, fh//3), (5*fw//6, 2*fh//3), (20,20,60), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        # Parpadeo
        if int(time.time() * 2) % 2 == 0:
            cv2.putText(frame, "ESTAS MAL SENTADO", (fw//6+20, fh//2-20),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, ROJO, 3, cv2.LINE_AA)
            for i, msg in enumerate(mensajes_chicos[:3]):
                cv2.putText(frame, msg, (fw//6+20, fh//2+20+i*28),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, AMARILLO, 1, cv2.LINE_AA)
    else:
        # Cartel chico en esquina inferior
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, fh-80), (fw, fh), (20,20,60), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        for i, msg in enumerate(mensajes_chicos[:2]):
            cv2.putText(frame, msg, (16, fh-55+i*30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, ROJO, 2, cv2.LINE_AA)


# ══════════════════════════════════════════════════════
# LOOP PRINCIPAL
# ══════════════════════════════════════════════════════
def main():
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )
    detector = vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMARA_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print(f"No se pudo abrir la cámara {CAMARA_ID}.")
        return

    panel_w = 260
    cam_w   = ANCHO_VENTANA - panel_w

    modo_calibracion = True
    monitoreo_activo = False
    pausado          = False

    t_inicio_mala   = None
    t_total_mala    = 0.0
    alarma_sonando  = False

    t_inicio_minuto  = time.time()
    conteo_ok        = 0
    conteo_total     = 0
    conteo_problemas = {'Cuello':0,'Torso':0,'Hombros':0,'Pies':0}
    ultimo_resumen   = None

    postura_vacia = {
        'oreja':(0,0),'hombro':(0,0),'cadera':(0,0),
        'rodilla':(0,0),'tobillo':(0,0),'espalda_alta':(0,0),
        'ang_cuello':0.0,'ang_torso':0.0,'ang_rodilla':90.0,'desp_hombro':0,
        'cuello_ok':True,'torso_ok':True,'hombro_ok':True,'pies_ok':True,
        'postura_ok':True,
    }
    p = postura_vacia.copy()

    print("="*52)
    print("  Monitor de postura iniciado")
    print("  Posicionáte de perfil y presioná ESPACIO")
    print("  Q=salir | ESPACIO=pausar | R=resetear")
    print("="*52)

    frame_ts = 0

    while True:
        ret, frame_orig = cap.read()
        if not ret:
            print("No se pudo leer el frame.")
            break

        frame_ts += 33
        frame_orig = cv2.flip(frame_orig, 1)
        frame = cv2.resize(frame_orig, (cam_w, ALTO_VENTANA))
        fh, fw = frame.shape[:2]

        # Detección
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img    = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        resultado = detector.detect_for_video(mp_img, frame_ts)

        persona_detectada = bool(resultado.pose_landmarks)

        if persona_detectada:
            lms = resultado.pose_landmarks[0]
            p   = calcular_posturas(lms, fw, fh)

            if monitoreo_activo and not pausado:
                dibujar_esqueleto(frame, p)

                conteo_total += 1
                if p['postura_ok']:
                    conteo_ok += 1
                    if t_inicio_mala is not None:
                        t_total_mala += time.time() - t_inicio_mala
                        t_inicio_mala = None
                    alarma_sonando = False
                else:
                    if not p['cuello_ok']:  conteo_problemas['Cuello']  += 1
                    if not p['torso_ok']:   conteo_problemas['Torso']   += 1
                    if not p['hombro_ok']:  conteo_problemas['Hombros'] += 1
                    if not p['pies_ok']:    conteo_problemas['Pies']    += 1
                    if t_inicio_mala is None:
                        t_inicio_mala = time.time()

                # Tiempo seguido en mala postura
                t_seguida = 0.0
                if t_inicio_mala is not None:
                    t_seguida = time.time() - t_inicio_mala

                t_total_ahora = t_total_mala
                if t_inicio_mala is not None:
                    t_total_ahora += t_seguida

                # Cartel de alerta
                dibujar_alerta(frame, p, t_seguida)

                # Notificacion de Windows a los 30 segundos
                if t_seguida >= SEGUNDOS_AVISO and not p['postura_ok']:
                    problemas = []
                    if not p['cuello_ok']: problemas.append(f"Cuello {p['ang_cuello']:.0f}")
                    if not p['torso_ok']:  problemas.append(f"Torso {p['ang_torso']:.0f}")
                    if not p['pies_ok']:   problemas.append(f"Pies {p['ang_rodilla']:.0f}")
                    enviar_notificacion(
                        "Postura incorrecta",
                        " | ".join(problemas) if problemas else "Corregí tu postura"
                    )

                # Alarma sonora al minuto
                if t_seguida >= SEGUNDOS_ALARMA and not alarma_sonando:
                    alarma_sonando = True
                    threading.Thread(target=sonar_alarma, daemon=True).start()
                    print(f"\n ALARMA: {t_seguida:.0f}s con mala postura\n")

                # Resumen por minuto
                if time.time() - t_inicio_minuto >= INTERVALO_RESUMEN:
                    pct = (conteo_ok / conteo_total * 100) if conteo_total else 0
                    probs = sorted([(k,v) for k,v in conteo_problemas.items() if v>0],
                                   key=lambda x: -x[1])
                    ultimo_resumen = {'pct_ok': pct, 'problemas': probs}
                    print(f"\n--- Resumen del último minuto ---")
                    print(f"   Postura correcta: {pct:.0f}%")
                    for k, v in probs:
                        print(f"   {k}: {v}s incorrectos")
                    t_inicio_minuto  = time.time()
                    conteo_ok        = 0
                    conteo_total     = 0
                    conteo_problemas = {'Cuello':0,'Torso':0,'Hombros':0,'Pies':0}
        else:
            p = postura_vacia.copy()
            t_seguida     = 0.0
            t_total_ahora = t_total_mala

        if modo_calibracion:
            dibujar_guia_encuadre(frame, fw, fh, persona_detectada)

        # Panel
        panel = np.zeros((ALTO_VENTANA, panel_w, 3), dtype=np.uint8)

        t_seg_panel   = 0.0
        t_total_panel = t_total_mala
        if monitoreo_activo and t_inicio_mala is not None:
            t_seg_panel    = time.time() - t_inicio_mala
            t_total_panel += t_seg_panel

        if not monitoreo_activo:
            # Panel en calibración
            panel[:] = FONDO_PANEL
            cv2.putText(panel, "MONITOR DE POSTURA", (16,32),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, BLANCO, 1, cv2.LINE_AA)
            cv2.putText(panel, "EN CALIBRACION", (16, ALTO_VENTANA//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, AMARILLO, 1, cv2.LINE_AA)
        else:
            dibujar_panel(panel, panel_w, ALTO_VENTANA,
                          p, t_seg_panel, t_total_panel,
                          ultimo_resumen, pausado)

        ventana = np.hstack([frame, panel])
        cv2.imshow('Monitor de Postura', ventana)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            if modo_calibracion:
                if persona_detectada:
                    modo_calibracion = False
                    monitoreo_activo = True
                    t_inicio_minuto  = time.time()
                    print("Monitoreo iniciado.")
                else:
                    print("No se detectó persona. Posicionáte mejor.")
            else:
                pausado = not pausado
                if pausado:
                    if t_inicio_mala is not None:
                        t_total_mala += time.time() - t_inicio_mala
                        t_inicio_mala = None
                    print("Pausado.")
                else:
                    print("Reanudado.")
        elif key == ord('r'):
            t_total_mala     = 0.0
            t_inicio_mala    = None
            conteo_ok        = 0
            conteo_total     = 0
            conteo_problemas = {'Cuello':0,'Torso':0,'Hombros':0,'Pies':0}
            t_inicio_minuto  = time.time()
            ultimo_resumen   = None
            alarma_sonando   = False
            print("Contadores reseteados.")
        elif key == ord('c'):
            modo_calibracion = True
            monitoreo_activo = False
            print("Volviendo a calibración.")

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Monitor cerrado.")


if __name__ == '__main__':
    main()
