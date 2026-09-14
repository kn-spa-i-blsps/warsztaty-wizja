import math
import random
import cv2
from ultralytics import YOLO

# ==========================================
# 1. KONFIGURACJA MODELU I PUNKTÓW COCO
# ==========================================
model = YOLO("yolo26n-pose.pt")  # lub yolov8n-pose.pt

LEFT_WRIST = 9
RIGHT_WRIST = 10
CLAP_THRESHOLD = 70  # Dystans w pikselach uznawany za klaśnięcie

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
was_clapping = False  # Flaga do wykrywania pojedynczego impulsu (zbocza)


def reset_game():
    global bird_y, bird_velocity, pipe_x, gap_y, score, game_over
    bird_y = 240.0
    bird_velocity = 0.0
    pipe_x = 640.0
    gap_y = random.randint(150, 330)
    score = 0
    game_over = False


# ==========================================
# 3. GŁÓWNA PĘTLA GRY
# ==========================================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Gra uruchomiona! Klaśnij, aby skoczyć. Klawisz 'R' resetuje, 'Q' wyłącza.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Lustrzane odbicie kamery (naturalniejszy ruch przed ekranem)
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # --- INFERENCJA MODELU POSE ---
    # verbose=False wyłącza spamowanie konsoli logami przy każdej klatce
    results = model(frame, verbose=False)

    is_clapping = False
    dist = None

    # Odczytujemy dłonie, jeśli model wykrył co najmniej jedną osobę
    if len(results[0].keypoints) > 0:
        points = results[0].keypoints.xy[0].tolist()
        confs = results[0].keypoints.conf[0].tolist()

        if confs[LEFT_WRIST] > 0.4 and confs[RIGHT_WRIST] > 0.4:
            x_l, y_l = int(points[LEFT_WRIST][0]), int(points[LEFT_WRIST][1])
            x_p, y_p = int(points[RIGHT_WRIST][0]), int(points[RIGHT_WRIST][1])

            # Odległość euklidesowa między nadgarstkami
            dist = math.dist((x_l, y_l), (x_p, y_p))
            is_clapping = dist < CLAP_THRESHOLD

            # Wizualizacja dłoni i linii dystansu
            kolor = (0, 0, 255) if is_clapping else (0, 255, 0)
            cv2.circle(frame, (x_l, y_l), 8, kolor, -1)
            cv2.circle(frame, (x_p, y_p), 8, kolor, -1)
            cv2.line(frame, (x_l, y_l), (x_p, y_p), kolor, 2)

    # --- LOGIKA STEROWANIA (WYKRYWANIE ZBOCZA) ---
    # Skok następuje tylko w momencie zetknięcia dłoni (zbocze narastające).
    # Zapobiega to ciągłemu lotowi w sufit przy trzymaniu złożonych rąk.
    if is_clapping and not was_clapping:
        if game_over:
            reset_game()
        else:
            bird_velocity = JUMP_STRENGTH

    was_clapping = is_clapping

    # --- MECHANIKA GRY ---
    if not game_over:
        # Całkowanie prędkości i grawitacji (fizyka)
        bird_velocity += GRAVITY
        bird_y += bird_velocity

        # Ruch przeszkody w lewo
        pipe_x -= pipe_speed
        if pipe_x < -70:
            pipe_x = w
            gap_y = random.randint(140, 340)
            score += 1

        # Sprawdzenie kolizji z krawędziami ekranu
        if bird_y < 15 or bird_y > h - 15:
            game_over = True

        # Sprawdzenie kolizji z rurami (AABB)
        w_pasmie_rury = (pipe_x < bird_x + 15) and (bird_x - 15 < pipe_x + 60)
        uderzenie_gora = bird_y - 15 < (gap_y - GAP_SIZE // 2)
        uderzenie_dol = bird_y + 15 > (gap_y + GAP_SIZE // 2)

        if w_pasmie_rury and (uderzenie_gora or uderzenie_dol):
            game_over = True

    # --- RYSOWANIE GRAFIKI GRY ---
    # Rury (Górna i Dolna)
    cv2.rectangle(
        frame, (int(pipe_x), 0), (int(pipe_x + 60), gap_y - GAP_SIZE // 2), (0, 180, 0), -1
    )
    cv2.rectangle(
        frame, (int(pipe_x), gap_y + GAP_SIZE // 2), (int(pipe_x + 60), h), (0, 180, 0), -1
    )

    # Ptak (Żółte kółko)
    cv2.circle(frame, (int(bird_x), int(bird_y)), 16, (0, 220, 255), -1)
    cv2.circle(frame, (int(bird_x + 6), int(bird_y - 4)), 3, (0, 0, 0), -1)  # Oko ptaka

    # Interfejs gracza (UI)
    cv2.putText(
        frame, f"Wynik: {score}", (25, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2
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

    # Wyświetlenie okna
    cv2.imshow("Flappy Bird AI - Sterowanie Klasnieciem", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
    elif key == ord("r"):
        reset_game()

cap.release()
cv2.destroyAllWindows()
