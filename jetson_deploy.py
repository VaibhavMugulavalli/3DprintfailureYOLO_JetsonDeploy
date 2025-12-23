import os, time, cv2, requests
from datetime import datetime
from pathlib import Path
from ultralytics import YOLO

MODEL_ENGINE = "best_fp16.engine"  # TensorRT engine

# -------- HP W200 USB webcam config --------
CAM_INDEX  = 0
CAM_WIDTH  = 1280
CAM_HEIGHT = 720
CAM_FPS    = 30   # some W200 modes are 30, others 15; camera decides

IMG_SIZE = 640
CONF_THRESH = 0.35
TARGET_NAMES = {"warping", "Spaghetti - v11 2025-04-15 12:28am"}
CONSEC_FRAMES = 8
COOLDOWN_SEC = 180

SNAP_DIR = Path("snapshots"); SNAP_DIR.mkdir(exist_ok=True)
VIDEO_DIR = Path("videos"); VIDEO_DIR.mkdir(exist_ok=True)

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8195204038:AAFGy5bu0FyX-5TBns_Qeg0-P_kwefeazLM")
CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID", "1442216765")

def send_telegram(text, photo_path=None):
    if not BOT_TOKEN or not CHAT_ID:
        print("[WARN] Telegram not configured.")
        return
    try:
        requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
                      data={"chat_id": CHAT_ID, "text": text}, timeout=10)
        if photo_path:
            with open(photo_path, "rb") as f:
                requests.post(f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto",
                              data={"chat_id": CHAT_ID}, files={"photo": f}, timeout=20)
    except Exception as e:
        print("[WARN] Telegram send failed:", e)

def norm(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())

TARGET_NORM = {norm(x) for x in TARGET_NAMES}

def open_hp_w200(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)

    # Request MJPEG (often smoother on Jetson for 720p webcams)
    # If MJPG is not supported, the camera will ignore it and fall back.
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, CAM_FPS)

    # Reduce latency
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap

def main():
    print("[INFO] Loading TensorRT engine...")
    model = YOLO(MODEL_ENGINE)
    print("[INFO] Model names:", model.names)

    print(f"[INFO] Opening HP W200 on /dev/video{CAM_INDEX} ...")
    cap = open_hp_w200(CAM_INDEX)
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try CAM_INDEX=1 or check /dev/video*.")

    # Read first frame (also ensures writer uses correct size)
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("Camera opened but couldn't read frames.")

    actual_h, actual_w = frame.shape[:2]
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)])

    print(f"[INFO] Camera actual: {actual_w}x{actual_h} @ {actual_fps:.2f} FPS, FOURCC={fourcc_str}")

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = VIDEO_DIR / f"monitor_{ts}.mp4"

    # Save at 20 fps (stable file even if capture FPS fluctuates)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        20.0,
        (actual_w, actual_h)
    )
    if not writer.isOpened():
        raise RuntimeError("VideoWriter failed. Install ffmpeg: sudo apt install ffmpeg")

    print("[INFO] Recording:", out_path)

    consec, last_alert = 0, 0
    win = "Jetson Nano 3D Print Monitor (HP W200 USB Cam)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.01)
            continue

        res = model(frame, imgsz=IMG_SIZE, conf=CONF_THRESH, verbose=False)[0]

        has_fail = False
        if res.boxes is not None and len(res.boxes) > 0:
            for c, cf in zip(res.boxes.cls.tolist(), res.boxes.conf.tolist()):
                name = res.names.get(int(c), str(int(c)))
                if norm(name) in TARGET_NORM and cf >= CONF_THRESH:
                    has_fail = True
                    break

        vis = res.plot()
        consec = consec + 1 if has_fail else 0

        if consec >= CONSEC_FRAMES and (time.time() - last_alert) > COOLDOWN_SEC:
            ts2 = datetime.now().strftime("%Y%m%d_%H%M%S")
            snap = SNAP_DIR / f"failure_{ts2}.jpg"
            cv2.imwrite(str(snap), vis)
            msg = f"⚠️ 3D print failure detected at {ts2} ({consec}/{CONSEC_FRAMES})"
            print("[ALERT]", msg)
            send_telegram(msg, str(snap))
            last_alert = time.time()

        cv2.putText(vis, f"hits: {consec}/{CONSEC_FRAMES}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

        writer.write(vis)
        cv2.imshow(win, vis)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print("[INFO] Saved video:", out_path)

if __name__ == "__main__":
    main()
