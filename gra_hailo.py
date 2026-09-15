import math
import random
import cv2
from hailo_platform import VDevice
import numpy as np
from picamera2 import Picamera2

# ==========================================
# 1. KONFIGURACJA MODELU I PUNKTÓW COCO
# ==========================================
HEF_PATH = "/usr/share/hailo-models/yolov8s_pose_h8l_pi.hef"
INPUT_SIZE = (640, 640)

LEFT_WRIST = 9
RIGHT_WRIST = 10
CLAP_THRESHOLD = 70  # Próg odległości dłoni w pikselach

# ==========================================
# 2. PARAMETRY FIZYKI FLAPPY BIRD
# ==========================================
GRAVITY = 0.9
JUMP_STRENGTH = -11.0

bird_x = 120
bird_y = 240.0
bird_velocity = 0.0

pipe_x = 640.0
pipe_speed = 6.0
GAP_SIZE = 160
gap_y = random.randint(150, 330)

score = 0
game_over = False
was_clapping = False


def reset_game():
    global bird_y, bird_velocity, pipe_x, gap_y, score, game_over
    bird_y = 240.0
    bird_velocity = 0.0
    pipe_x = 640.0
    gap_y = random.randint(150, 330)
    score = 0
    game_over = False


# ==========================================
# 3. KAMERA PICAMERA2 (CSI / IMX477)
# ==========================================
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()

# ==========================================
# 4. INICJALIZACJA HAILO (BUFORY PRZED PĘTLĄ)
# ==========================================
target = VDevice()
infer_model = target.create_infer_model(HEF_PATH)
infer_model.set_batch_size(1)
configured_infer_model = infer_model.configure()

# Tworzymy powiązania i bufory pamięci raz przed startem gry (kluczowe dla płynności)
bindings = configured_infer_model.create_bindings()
output_buffers = {
    name: np.empty(
        infer_model.output(name).shape, dtype=infer_model.output(name).dtype
    )
    for name in infer_model.output_names
}
for name, buf in output_buffers.items():
    bindings.output(name).set_buffer(buf)

print("Akcelerator Hailo gotowy do gry!")
print("Klaśnij dłońmi, aby ptak skoczył. 'R' - reset gry, 'Q' - wyjście.")

# ==========================================
# 5. PĘTLA GRY
# ==========================================
try:
    while True:
        # Pobranie klatki z sensora i odbicie lustrzane
        frame = picam2.capture_array()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # Przygotowanie tensora wejściowego dla modelu (640x640, RGB)
        input_frame = cv2.resize(frame, INPUT_SIZE)
        input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        input_data = np.expand_dims(input_frame, axis=0)

        # Wstrzyknięcie klatki do bufora i obliczenia na Hailo
        bindings.input().set_buffer(input_data)
        configured_infer_model.run([bindings], timeout_ms=1000)

        # --- ODCZYT I PARSOWANIE POZYCJI DŁONI ---
        is_clapping = False
        dist = None

        for out_name, buf in output_buffers.items():
            # Szukamy tensora zawierającego punkty szkieletu
            if "keypoints" in out_name or len(buf.shape) == 3:
                raw_kpts = buf[0]  # Kształt [17, 3]

                if len(raw_kpts) >= 17:
                    scale_y = h / INPUT_SIZE[1]
                    scale_x = w / INPUT_SIZE[0]

                    p_left = raw_kpts[LEFT_WRIST]
                    p_right = raw_kpts[RIGHT_WRIST]

                    conf_left = p_left[2] if len(p_left) > 2 else 1.0
                    conf_right = p_right[2] if len(p_right) > 2 else 1.0

                    if conf_left > 0.4 and conf_right > 0.4:
                        x_l = int(p_left[0] * scale_x)
                        y_l = int(p_left[1] * scale_y)
                        x_p = int(p_right[0] * scale_x)
                        y_p = int(p_right[1] * scale_y)

                        dist = math.dist((x_l, y_l), (x_p, y_p))
                        is_clapping = dist < CLAP_THRESHOLD

                        # Wizualizacja dłoni i odległości
                        kolor = (0, 0, 255) if is_clapping else (0, 255, 0)
                        cv2.circle(frame, (x_l, y_l), 8, kolor, -1)
                        cv2.circle(frame, (x_p, y_p), 8, kolor, -1)
                        cv2.line(frame, (x_l, y_l), (x_p, y_p), kolor, 2)
                break

        # --- LOGIKA STEROWANIA (DETEKCJA ZBOCZA) ---
        if is_clapping and not was_clapping:
            if game_over:
                reset_game()
            else:
                bird_velocity = JUMP_STRENGTH

        was_clapping = is_clapping

        # --- MECHANIKA I KOLIZJE FLAPPY BIRD ---
        if not game_over:
            bird_velocity += GRAVITY
            bird_y += bird_velocity

            pipe_x -= pipe_speed
            if pipe_x < -70:
                pipe_x = w
                gap_y = random.randint(140, 340)
                score += 1

            # Kolizja z sufitem i podłogą
            if bird_y < 15 or bird_y > h - 15:
                game_over = True

            # Kolizja z rurami (AABB)
            w_pasmie = (pipe_x < bird_x + 15) and (bird_x - 15 < pipe_x + 60)
            uderzenie_gora = bird_y - 15 < (gap_y - GAP_SIZE // 2)
            uderzenie_dol = bird_y + 15 > (gap_y + GAP_SIZE // 2)

            if w_pasmie and (uderzenie_gora or uderzenie_dol):
                game_over = True

        # --- RYSOWANIE GRAFIKI ---
        # Przeszkody
        cv2.rectangle(
            frame,
            (int(pipe_x), 0),
            (int(pipe_x + 60), gap_y - GAP_SIZE // 2),
            (0, 180, 0),
            -1,
        )
        cv2.rectangle(
            frame,
            (int(pipe_x), gap_y + GAP_SIZE // 2),
            (int(pipe_x + 60), h),
            (0, 180, 0),
            -1,
        )

        # Ptak
        cv2.circle(frame, (int(bird_x), int(bird_y)), 16, (0, 220, 255), -1)
        cv2.circle(frame, (int(bird_x + 6), int(bird_y - 4)), 3, (0, 0, 0), -1)

        # Informacje na ekranie
        cv2.putText(
            frame,
            f"Wynik: {score}",
            (25, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (255, 255, 255),
            2,
        )

        if dist is not None:
            cv2.putText(
                frame,
                f"Dlonie: {int(dist)}px",
                (25, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        if game_over:
            cv2.putText(
                frame,
                "GAME OVER",
                (w // 2 - 170, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.4,
                (0, 0, 255),
                4,
            )
            cv2.putText(
                frame,
                "Klasnij lub nacisnij 'R', aby zagrac",
                (w // 2 - 210, h // 2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        cv2.imshow("Flappy Bird AI - Hailo-8 NPU", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            reset_game()

finally:
    picam2.stop()
    cv2.destroyAllWindows()
