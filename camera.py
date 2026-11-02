from pathlib import Path
import cv2
from ultralytics import YOLO

model_path = Path(__file__).parent / "best.pt"

if not model_path.exists():
    raise FileNotFoundError("best.pt ko camera.py wale folder mein rakho.")

model = YOLO(str(model_path))
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    raise RuntimeError("Camera nahi khul raha. Camera use karne wali apps band karo.")

try:
    while True:
        success, frame = camera.read()
        if not success:
            print("Camera se frame nahi mila.")
            break

        results = model.predict(frame, conf=0.5, verbose=False)
        annotated_frame = results[0].plot()

        cv2.imshow("Waste Detection - Q to exit", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()