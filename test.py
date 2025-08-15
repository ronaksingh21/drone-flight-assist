#!/usr/bin/env python3
"""
Guide-Drone Follow & Guard
- ArUco: metric distance to user (follow spacing).
- YOLOv5: continuous obstacle detection + spoken warnings.
- Area-based distance model for obstacles with 2-point calibration.
- Per-class cooldown + hysteresis to avoid chatter.

Keys:
  q / Esc   quit
  1         capture NEAR calibration sample (set --near_ft)
  2         capture FAR  calibration sample (set --far_ft)

Tip: Install OpenCV with ArUco via:
  pip uninstall -y opencv-python
  pip install opencv-contrib-python
"""

import os, sys, time, math, json, argparse
from pathlib import Path
import numpy as np
import cv2
import torch

# ---------- Speech (pyttsx3 optional) ----------
try:
    import pyttsx3
    _tts = pyttsx3.init()
    def speak(text: str):
        _tts.say(text)
        _tts.runAndWait()
except Exception:
    def speak(text: str):
        print("[SAY]", text)

# ---------- CLI ----------
ap = argparse.ArgumentParser()
ap.add_argument("--cam", type=int, default=0, help="webcam index")
ap.add_argument("--yolo_root", type=str, default=".", help="path to YOLOv5 repo root")
ap.add_argument("--weights", type=str, default="yolov5n.pt", help="YOLOv5 weights (n/s/m...)")
ap.add_argument("--img", type=int, default=416, help="inference size")
ap.add_argument("--conf", type=float, default=0.35, help="confidence threshold")
ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
ap.add_argument("--corridor", type=float, default=0.40, help="forward corridor width fraction (0..1)")
ap.add_argument("--alert_ft", type=float, default=10.0, help="default obstacle alert distance (ft)")
ap.add_argument("--cooldown_s", type=float, default=3.0, help="per-class speech cooldown")
ap.add_argument("--near_ft", type=float, default=3.0, help="ground-truth near distance (ft) for 2-pt calib")
ap.add_argument("--far_ft", type=float, default=10.0, help="ground-truth far distance (ft) for 2-pt calib")
ap.add_argument("--show", action="store_true", help="show UI window")
ap.add_argument("--no-aruco", action="store_true", help="disable ArUco even if available")
ap.add_argument("--calibrate-aruco", action="store_true", help="run ArUco focal-length calibration mode")
ap.add_argument("--marker_mm", type=float, default=70.0, help="ArUco black square width (mm)")
ap.add_argument("--dist_m", type=float, default=0.60, help="calibration distance to marker (meters)")
ap.add_argument("--aruco_dict", type=int, default=5, help="5 means DICT_5X5_50; change if you printed a different one")
args = ap.parse_args()


ap.add_argument('--calibrate-aruco', action='store_true',
                    help='Run ArUco marker calibration')
ap.add_argument('--marker_mm', type=float, default=70,
                    help='Physical marker size in millimeters')


# ---------- YOLOv5 imports ----------
YOLO_ROOT = Path(args.yolo_root).resolve()
if str(YOLO_ROOT) not in sys.path:
    sys.path.append(str(YOLO_ROOT))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes, check_img_size
from utils.torch_utils import select_device
from utils.augmentations import letterbox

# ---------- Globals / Config ----------
IMG_SIZE = args.img
CONF_THRES = args.conf
IOU_THRES  = args.iou
FORWARD_FOV_FRAC = max(0.1, min(0.95, args.corridor))
ALERT_DIST_FEET = args.alert_ft
ALERT_COOLDOWN_S = args.cooldown_s

# Per-class default thresholds (you can tweak)
CLASS_THRESH_FT = {
    "person": 8.0,
    "bicycle": 8.0,
    "chair": 6.0,
    "bench": 6.0,
    "tv": 8.0,
    "dog": 6.0,
}

# Interest set (COCO indices)
INTERESTING = None  # None = all; or e.g., {0,56,57,63}
ONLY_PERSON_FOR_CAL = True  # use 'person' boxes for 2-pt calibration

# --------- Obstacle distance model: D ≈ a / sqrt(area) + b ----------
A_B_CAL = {"a": 900.0, "b": 0.0}  # overwritten after 2-pt cal
ROLL_S, ROLL_S_MAXLEN = [], 5
EMA_ALPHA = 0.65
ema_dist = None

def est_dist_from_area(w_px, h_px):
    """Distance in feet from bbox area proxy with smoothing & median."""
    global ema_dist
    w_px, h_px = max(4, int(w_px)), max(4, int(h_px))
    s_now = math.sqrt(float(w_px * h_px))
    ROLL_S.append(s_now)
    if len(ROLL_S) > ROLL_S_MAXLEN:
        del ROLL_S[0]
    s_med = float(np.median(ROLL_S))
    a, b = A_B_CAL["a"], A_B_CAL["b"]
    d_raw = (a / max(1.0, s_med)) + b
    d_raw = float(np.clip(d_raw, 1.0, 100.0))
    if ema_dist is None or abs(d_raw - ema_dist) > 4.0:
        ema_dist = d_raw
    else:
        ema_dist = EMA_ALPHA * d_raw + (1.0 - EMA_ALPHA) * ema_dist
    return ema_dist

def solve_ab_area(s1, d1, s2, d2):
    """Solve a,b from D=a/s + b with two points."""
    if s1 == s2:
        return None
    a = (d1 - d2) / ((1.0/s1) - (1.0/s2))
    b = d1 - a * (1.0 / s1)
    return a, b

# ---------- ArUco helpers ----------
CAL_FILE = Path("focal.json")

def have_aruco():
    if args.no-aruco:
        return False
    try:
        _ = cv2.aruco
        return True
    except Exception:
        return False

def get_aruco_detector():
    """Return (detect_fn, dict_obj) or (None, None) if unavailable."""
    if not have_aruco():
        return None, None
    aruco = cv2.aruco
    # choose dictionary
    if args.aruco_dict == 5:
        DICT = aruco.getPredefinedDictionary(aruco.DICT_5X5_50)
    else:
        DICT = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # detector across API versions
    if hasattr(aruco, "ArucoDetector"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(DICT, params)
        def _detect(frame):
            corners, ids, _ = detector.detectMarkers(frame)
            return corners, ids
    else:
        # old API
        params = aruco.DetectorParameters_create() if hasattr(aruco, "DetectorParameters_create") else aruco.DetectorParameters()
        def _detect(frame):
            corners, ids, _ = aruco.detectMarkers(frame, DICT, parameters=params)
            return corners, ids
    return _detect, DICT

def marker_px_width(corners):
    pts = corners.reshape(-1, 2)
    w1 = np.linalg.norm(pts[1] - pts[0])
    w2 = np.linalg.norm(pts[2] - pts[3])
    return float((w1 + w2) / 2.0)

def aruco_calibrate():
    detect, _ = get_aruco_detector()
    if detect is None:
        raise SystemExit("ArUco not available. Install opencv-contrib-python or use --no-aruco.")
    W_METERS = args.marker_mm / 1000.0
    D_m = args.dist_m
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened(): raise SystemExit("Camera not found")
    print(f"Show marker (width {args.marker_mm:.0f}mm) at exactly {D_m:.2f} m. Press 'c' to capture, 'q' to finish.")
    F_vals = []
    while True:
        ok, frame = cap.read()
        if not ok: continue
        corners, ids = detect(frame)
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        cv2.putText(frame, "Calibrate: press 'c' to capture, 'q' to save", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
        cv2.imshow("ArUco Calibrate", frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('c') and ids is not None:
            P = np.mean([marker_px_width(c) for c in corners])
            F = (P * D_m) / W_METERS
            F_vals.append(F)
            print(f"[CAL] P={P:.1f}px  F={F:.1f}")
        if k == ord('q') or k == 27:
            break
    cap.release(); cv2.destroyAllWindows()
    if not F_vals:
        raise SystemExit("No captures taken. Try again.")
    F_mean = float(np.median(F_vals))
    CAL_FILE.write_text(json.dumps({"F": F_mean, "W_m": W_METERS}, indent=2))
    print("Saved", CAL_FILE.name, "=>", {"F": F_mean, "W_m": W_METERS})

def aruco_distance_ft(frame, detect, F, W_m):
    """Return (feet, box_xyxy) or (None, None)."""
    corners, ids = detect(frame)
    if ids is None:
        return None, None
    P = np.mean([marker_px_width(c) for c in corners])
    D_m = (W_m * F) / max(P, 1.0)
    # make a single box around all detected markers (or just first)
    c = corners[0].reshape(-1,2).astype(int)
    x1 = int(np.min(c[:,0])); x2 = int(np.max(c[:,0]))
    y1 = int(np.min(c[:,1])); y2 = int(np.max(c[:,1]))
    return D_m * 3.28084, (x1,y1,x2,y2)

# ---------- Main run ----------
def main():
    if args.calibrate_aruco:
        aruco_calibrate()
        return

    # ArUco runtime (optional)
    aruco_detect, _ = get_aruco_detector()
    F, W_m = None, None
    if aruco_detect and CAL_FILE.exists():
        try:
            d = json.loads(CAL_FILE.read_text())
            F, W_m = d["F"], d["W_m"]
            print(f"[ArUco] Using focal F={F:.1f}, width={W_m*1000:.0f}mm")
        except Exception:
            print("[ArUco] Bad focal.json; ArUco disabled")
            aruco_detect = None

    # YOLOv5
    device = select_device("")  # auto
    weights_path = (YOLO_ROOT / args.weights).as_posix()
    model = DetectMultiBackend(weights_path, device=device, dnn=False,
                               data=(YOLO_ROOT / "data/coco128.yaml").as_posix(), fp16=False)
    names = model.names if isinstance(model.names, list) else [str(i) for i in range(80)]
    stride, pt = model.stride, model.pt
    imgsz = check_img_size((IMG_SIZE, IMG_SIZE), s=stride)
    model.warmup(imgsz=(1, 3, *imgsz))

    # Video
    cap = cv2.VideoCapture(args.cam, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
    if not cap.isOpened(): raise SystemExit("Camera not found")

    print("Running. 'q' to quit. '1'/'2' to capture near/far calibration.")
    last_alert = {}  # per-class cooldown
    warning_state = False  # hysteresis
    ARM_FT, DISARM_FT = ALERT_DIST_FEET, ALERT_DIST_FEET + 2.0

    global ema_dist
    ema_dist = None
    CAL_NEAR = None
    CAL_FAR  = None

    while True:
        ok, frame0 = cap.read()
        if not ok:
            time.sleep(0.005)
            continue

        h0, w0 = frame0.shape[:2]
        cx1 = int(w0 * (0.5 - FORWARD_FOV_FRAC / 2))
        cx2 = int(w0 * (0.5 + FORWARD_FOV_FRAC / 2))

        # ArUco (follow distance to user)
        user_ft, aruco_box = (None, None)
        if aruco_detect and F is not None and W_m is not None:
            try:
                user_ft, aruco_box = aruco_distance_ft(frame0, aruco_detect, F, W_m)
            except Exception:
                user_ft, aruco_box = (None, None)

        # --- YOLO preprocess ---
        img = letterbox(frame0, imgsz, stride=stride, auto=pt)[0]
        img = img.transpose((2, 0, 1))  # HWC->CHW
        img = np.ascontiguousarray(img)
        im = torch.from_numpy(img).to(model.device).float() / 255.0
        if im.ndimension() == 3:
            im = im.unsqueeze(0)

        # --- inference & NMS ---
        with torch.no_grad():
            pred = model(im, augment=False, visualize=False)
        pred = non_max_suppression(pred, CONF_THRES, IOU_THRES, classes=None, agnostic=False, max_det=300)
        det = pred[0]

        nearest_txt, nearest_box, nearest_ft = None, None, 1e9

        if len(det):
            det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], frame0.shape).round()

            # pick the tallest interesting box inside corridor (best stability)
            best, best_h = None, -1
            for *xyxy, conf, cls in det:
                c = int(cls.item())
                if INTERESTING is not None and c not in INTERESTING:
                    continue

                x1, y1, x2, y2 = [int(v.item()) for v in xyxy]
                x1 = max(0, min(x1, w0 - 1)); x2 = max(0, min(x2, w0 - 1))
                y1 = max(0, min(y1, h0 - 1)); y2 = max(0, min(y2, h0 - 1))
                if x2 <= x1 or y2 <= y1:
                    continue

                # skip region overlapping ArUco (avoid “detecting the user” as obstacle)
                if aruco_box is not None:
                    ax1, ay1, ax2, ay2 = aruco_box
                    # IoU test
                    ix1, iy1 = max(x1, ax1), max(y1, ay1)
                    ix2, iy2 = min(x2, ax2), min(y2, ay2)
                    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
                    a1 = (x2-x1)*(y2-y1); a2 = (ax2-ax1)*(ay2-ay1)
                    if a1 > 0 and a2 > 0:
                        iou = inter / float(a1 + a2 - inter + 1e-6)
                        if iou > 0.3:
                            continue

                # forward corridor
                cx = (x1 + x2) // 2
                if cx < cx1 or cx > cx2:
                    continue

                h_box = (y2 - y1)
                if h_box < 24:  # ignore tiny/partial
                    continue

                if ONLY_PERSON_FOR_CAL and int(cls.item()) != 0:
                    pass  # still can be an obstacle, just won't be used for 2-pt capture prompt
                # choose tallest for stability
                if h_box > best_h:
                    best_h = h_box
                    best = (x1, y1, x2, y2, int(cls.item()))

            if best is not None:
                x1, y1, x2, y2, c = best
                w_box, h_box = (x2-x1), (y2-y1)
                # obstacle distance from area model
                vision_ft = est_dist_from_area(w_box, h_box)
                nearest_box = (x1, y1, x2, y2)
                nearest_ft  = vision_ft
                label = names[c] if isinstance(names, list) else str(c)
                nearest_txt = f"{label} ~{int(round(max(1, nearest_ft)))} ft"

        # --- Alerts with per-class cooldown & hysteresis ---
        now = time.time()
        if nearest_box is not None:
            label = nearest_txt.split()[0] if nearest_txt else "object"
            thresh = CLASS_THRESH_FT.get(label, ALERT_DIST_FEET)
            if nearest_ft < thresh and (last_alert.get(label, 0) + ALERT_COOLDOWN_S <= now):
                speak(f"Caution: {nearest_txt} ahead.")
                last_alert[label] = now

            # hysteresis (global)
            if not warning_state and nearest_ft < ARM_FT:
                warning_state = True
            elif warning_state and nearest_ft > DISARM_FT:
                warning_state = False

        # --- UI ---
        if args.show:
            vis = frame0.copy()
            # corridor
            cv2.rectangle(vis, (cx1, 0), (cx2, h0), (0, 255, 255), 2)
            # ArUco box & user distance
            if aruco_box is not None and user_ft is not None:
                x1, y1, x2, y2 = aruco_box
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,200,0), 2)
                cv2.putText(vis, f"user {user_ft:.1f} ft", (x1, max(18,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,0), 2, cv2.LINE_AA)
            # obstacle box
            if nearest_box is not None and nearest_txt is not None:
                x1, y1, x2, y2 = nearest_box
                cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(vis, nearest_txt, (x1, max(18,y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)

            cv2.putText(vis, "1/2: calibrate near/far | q: quit",
                        (10, h0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
            cv2.imshow("Guide Follow & Guard", vis)

        # --- Keys ---
        key = (cv2.waitKey(1) & 0xFF) if args.show else 255
        if key in (ord('q'), 27):
            break

        # 2-point calibration capture (using the current best box)
        if key in (ord('1'), ord('2')) and nearest_box is not None:
            x1, y1, x2, y2 = nearest_box
            s_now = math.sqrt(float((x2-x1) * (y2-y1)))
            if key == ord('1'):
                CAL_NEAR = (s_now, float(args.near_ft))
                print(f"[CAL] NEAR: s={s_now:.1f} at {args.near_ft:.1f} ft")
            elif key == ord('2'):
                CAL_FAR = (s_now, float(args.far_ft))
                print(f"[CAL] FAR:  s={s_now:.1f} at {args.far_ft:.1f} ft")
            if CAL_NEAR and CAL_FAR:
                (s1, d1), (s2, d2) = CAL_NEAR, CAL_FAR
                sol = solve_ab_area(s1, d1, s2, d2)
                if sol:
                    A_B_CAL["a"], A_B_CAL["b"] = sol
                    ROLL_S.clear()

                    ema_dist = None
                    print(f"[CAL] Fitted a={A_B_CAL['a']:.1f}, b={A_B_CAL['b']:.2f}")
                else:
                    print("[CAL] Failed to solve; try more separated distances")

    cap.release()
    if args.show:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    main()
