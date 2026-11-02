from pathlib import Path
from threading import Lock

import cv2
from flask import Flask, Response
from ultralytics import YOLO

app = Flask(__name__)

model_path = Path(__file__).parent / "best.pt"
if not model_path.is_file():
    raise FileNotFoundError("best.pt ko app.py wale folder mein rakho.")

model = YOLO(str(model_path))
camera_lock = Lock()

PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Waste Detection</title>
    <style>
        body {
            margin: 0; background: #101827; color: #edf2f7;
            font-family: Arial, sans-serif; text-align: center;
        }
        main { max-width: 900px; margin: 45px auto; padding: 20px; }
        h1 { color: #47dfb0; }
        p { color: #b5c4d8; line-height: 1.6; }
        img {
            display: none; width: 100%; margin-top: 25px;
            border-radius: 16px; background: #000;
        }
        button {
            padding: 13px 25px; margin: 8px; border: 0;
            border-radius: 8px; font-size: 16px; cursor: pointer;
        }
        .start { background: #47dfb0; }
        .stop { background: #ff7676; }
        #status { min-height: 25px; }
    </style>
</head>
<body>
<main>
    <h1>Smart Waste Detection</h1>
    <p>Show a waste item to your laptop camera.</p>

    <button class="start" onclick="startCamera()">Start Camera</button>
    <button class="stop" onclick="stopCamera()">Stop Camera</button>

    <p id="status">Camera is stopped.</p>
    <img id="feed" alt="Live waste detection">

    <script>
        const feed = document.getElementById("feed");
        const status = document.getElementById("status");

        function startCamera() {
            if (feed.hasAttribute("src")) return;
            status.textContent = "Starting camera...";
            feed.style.display = "block";
            feed.src = "/video_feed?t=" + Date.now();
        }

        function stopCamera() {
            feed.removeAttribute("src");
            feed.style.display = "none";
            status.textContent = "Camera is stopped.";
        }

        feed.onload = () => {
            status.textContent = "Live detection is running.";
        };

        feed.onerror = () => {
            stopCamera();
            status.textContent =
                "Camera unavailable. Close other camera apps and try again.";
        };
    </script>
</main>
</body>
</html>
"""


@app.get("/")
def home():
    return PAGE


def generate_frames(camera):
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            result = model.predict(
                frame, conf=0.5, imgsz=640, verbose=False
            )[0]
            annotated = result.plot()
            success, buffer = cv2.imencode(".jpg", annotated)
            if not success:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + buffer.tobytes()
                + b"\r\n"
            )
    finally:
        camera.release()
        camera_lock.release()


@app.get("/video_feed")
def video_feed():
    if not camera_lock.acquire(blocking=False):
        return "Camera busy. Close the other detection tab.", 409

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        camera.release()
        camera_lock.release()
        return "Camera unavailable.", 503

    return Response(
        generate_frames(camera),
        mimetype="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-store"},
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)