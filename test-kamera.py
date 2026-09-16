import cv2
from picamera2 import Picamera2

# Inicjalizacja kamery CSI (np. HQ Camera / IMX477)
picam2 = Picamera2()
config = picam2.create_video_configuration(
    main={"size": (640, 480), "format": "BGR888"}
)
picam2.configure(config)
picam2.start()

print("Podgląd uruchomiony. Wciśnij klawisz 'q' w oknie podglądu, aby wyjść.")

try:
    while True:
        # Pobranie klatki jako tablica NumPy (format BGR gotowy dla OpenCV)
        frame = picam2.capture_array()

        # Wyświetlenie obrazu w oknie (działa przez X11 Forwarding na laptopie)
        cv2.imshow("Podglad - Dostroj ostrosc i zamknij klawiszem Q", frame)

        # Zamknij pętlę po naciśnięciu 'q'
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    picam2.stop()
    cv2.destroyAllWindows()
