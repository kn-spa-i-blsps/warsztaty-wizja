import math
import random
import cv2
from hailo_platform import (
    HEF,
    ConfigureParams,
    FormatType,
    HailoStreamInterface,
    InferVStreams,
    InputVStreamParams,
    OutputVStreamParams,
    VDevice,
)
import numpy as np
from picamera2 import Picamera2

# ==========================================
# 1. KONFIGURACJA MODELU I PUNKTÓW COCO
# ==========================================
HEF_PATH = "/usr/share/hailo-models/yolov8s_pose_h8.hef"
INPUT_SIZE = (640, 640)

LEFT_WRIST = 9
RIGHT_WRIST = 10
CLAP_THRESHOLD = 75

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
# 4. INICJALIZACJA HAILO-8 (VStreams API)
# ==========================================
target = VDevice()
hef = HEF(HEF_PATH)

# Pobranie metadanych strumieni z HEF
input_vstream_infos = hef.get_input_vstream_infos()
output_vstream_infos = hef.get_output_vstream_infos()
input_name = input_vstream_infos[0].name

# Konfiguracja PCIe i parametrów VStreams
configure_params = ConfigureParams.create_from_hef(
    hef, interface=HailoStreamInterface.PCIe
)
network_group = target.configure(hef, configure_params)[0]
network_group_params = network_group.create_params()

input_vstreams_params = InputVStreamParams.make(
    network_group, format_type=FormatType.UINT8
)
output_vstreams_params = OutputVStreamParams.make(
    network_group, format_type=FormatType.FLOAT32
)

print("Akcelerator Hailo-8 skonfigurowany pomyślnie!")
print(f"Strumień wejściowy: {input_name}")
print("Strumienie wyjściowe:")
for info in output_vstream_infos:
    print(f"  • {info.name}: shape = {info.shape}")

print("Sterowanie: Klaśnięcie = skok. Klawisz 'R' = reset, 'Q' = wyjście.")

# ==========================================
# 5. GŁÓWNA PĘTLA GRY
# ==========================================
try:
    with network_group.activate(network_group_params):
        with InferVStreams(
            network_group, input_vstreams_params, output_vstreams_params
        ) as infer_pipeline:
            while True:
                # 1. Pobranie i przygotowanie klatki
                frame = picam2.capture_array()
                frame = cv2.flip(frame, 1)
                h, w, _ = frame.shape

                input_frame = cv2.resize(frame, INPUT_SIZE)
                input_frame = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
                input_data = {
                    input_name: np.expand_dims(input_frame, axis=0).astype(
                        np.uint8
                    )
                }

                # 2. Inferencja NPU
                outputs = infer_pipeline.infer(input_data)

                # 3. Odczyt współrzędnych dłoni
                is_clapping = False
                dist = None

                for out_name, buf in outputs.items():
                    # Format siatkowy [1, H, W, 51]
                    if buf.shape[-1] == 51:
                        grid_h, grid_w = buf.shape[1], buf.shape[2]
                        stride = INPUT_SIZE[0] // grid_w

                        flat_kpts = buf[0].reshape(grid_h * grid_w, 17, 3)
                        mean_confs = flat_kpts[:, :, 2].mean(axis=1)
                        best_cell_idx = int(np.argmax(mean_confs))

                        best_kpts = flat_kpts[best_cell_idx]
                        grid_y = best_cell_idx // grid_w
                        grid_x = best_cell_idx % grid_w

                        p_left = best_kpts[LEFT_WRIST]
                        p_right = best_kpts[RIGHT_WRIST]

                        scale_y = h / INPUT_SIZE[1]
                        scale_x = w / INPUT_SIZE[0]

                        x_l = int(
                            (p_left[0] * 2.0 - 0.5 + grid_x) * stride * scale_x
                        )
                        y_l = int(
                            (p_left[1] * 2.0 - 0.5 + grid_y) * stride * scale_y
                        )
                        x_p = int(
                            (p_right[0] * 2.0 - 0.5 + grid_x) * stride * scale_x
                        )
                        y_p = int(
                            (p_right[1] * 2.0 - 0.5 + grid_y) * stride * scale_y
                        )

                        dist = math.dist((x_l, y_l), (x_p, y_p))
                        is_clapping = dist < CLAP_THRESHOLD

                        kolor = (0, 0, 255) if is_clapping else (0, 255, 0)
                        cv2.circle(frame, (x_l, y_l), 8, kolor, -1)
                        cv2.circle(frame, (x_p, y_p), 8, kolor, -1)
                        cv2.line(frame, (x_l, y_l), (x_p, y_p), kolor, 2)
                        break

                    # Format punktów zdekodowanych [1, 17, 3]
                    elif "keypoints" in out_name or (
                        len(buf.shape) >= 2 and buf.shape[-2:] == (17, 3)
                    ):
                        raw_kpts = buf[0]
                        if len(raw_kpts.shape) == 3:
                            raw_kpts = raw_kpts[0]

                        scale_y = h / INPUT_SIZE[1]
                        scale_x = w / INPUT_SIZE[0]

                        p_left = raw_kpts[LEFT_WRIST]
                        p_right = raw_kpts[RIGHT_WRIST]

                        if p_left[2] > 0.3 and p_right[2] > 0.3:
                            x_l = int(p_left[0] * scale_x)
                            y_l = int(p_left[1] * scale_y)
                            x_p = int(p_right[0] * scale_x)
                            y_p = int(p_right[1] * scale_y)

                            dist = math.dist((x_l, y_l), (x_p, y_p))
                            is_clapping = dist < CLAP_THRESHOLD

                            kolor = (0, 0, 255) if is_clapping else (0, 255, 0)
                            cv2.circle(frame, (x_l, y_l), 8, kolor, -1)
                            cv2.circle(frame, (x_p, y_p), 8, kolor, -1)
                            cv2.line(frame, (x_l, y_l), (x_p, y_p), kolor, 2)
                        break

                # 4. Sterowanie ptakiem
                if is_clapping and not was_clapping:
                    if game_over:
                        reset_game()
                    else:
                        bird_velocity = JUMP_STRENGTH

                was_clapping = is_clapping

                # 5. Fizyka i kolizje
                if not game_over:
                    bird_velocity += GRAVITY
                    bird_y += bird_velocity

                    pipe_x -= pipe_speed
                    if pipe_x < -70:
                        pipe_x = w
                        gap_y = random.randint(140, 340)
                        score += 1

                    if bird_y < 15 or bird_y > h - 15:
                        game_over = True

                    w_pasmie = (pipe_x < bird_x + 15) and (
                        bird_x - 15 < pipe_x + 60
                    )
                    uderzenie_gora = bird_y - 15 < (gap_y - GAP_SIZE // 2)
                    uderzenie_dol = bird_y + 15 > (gap_y + GAP_SIZE // 2)

                    if w_pasmie and (uderzenie_gora or uderzenie_dol):
                        game_over = True

                # 6. Rysowanie grafiki
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

                cv2.circle(
                    frame, (int(bird_x), int(bird_y)), 16, (0, 220, 255), -1
                )
                cv2.circle(
                    frame, (int(bird_x + 6), int(bird_y - 4)), 3, (0, 0, 0), -1
                )

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
