import cv2, numpy as np
from collections import deque
import time

# ========= Settings =========
REF_IMAGES = [
    "wafer_ref_1.png",
    "wafer_ref_2.png",
    "wafer_ref_3.png",
    "wafer_ref_4.png",
    "wafer_ref_5.png",
    "wafer_ref_6.png"
]

DETECTOR = "ORB"              # "ORB" | "AKAZE" | "SIFT" (SIFT needs opencv-contrib-python)
CONF_RATIO = 0.70             # Lowe's ratio test
MIN_GOOD_MATCHES = 6         # quick pre-filter before RANSAC
MIN_INLIERS = 6             # accept if any ref has >= this many inliers
SMOOTH_N = 5                  # frames for majority vote smoothing
SHOW_DEBUG = True
CAM_INDEX = 0

# PASS behavior
EXIT_ON_PASS = False           # True = close the app after PASS overlay
PASS_HOLD_MS = 1500           # how long to show PASS before exit (ms)
# ===========================

# ---- choose detector/norm ----
if DETECTOR.upper() == "AKAZE":
    detector = cv2.AKAZE_create()
    norm = cv2.NORM_HAMMING
elif DETECTOR.upper() == "SIFT":
    if not hasattr(cv2, "SIFT_create"):
        raise RuntimeError("SIFT not available. Install opencv-contrib-python.")
    detector = cv2.SIFT_create(nfeatures=3000)
    norm = cv2.NORM_L2
else:
    detector = cv2.ORB_create(nfeatures=3000, fastThreshold=7)
    norm = cv2.NORM_HAMMING

bf = cv2.BFMatcher(norm, crossCheck=False)

# ---- load & preprocess references ----
refs = []
for path in REF_IMAGES:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[warn] Missing ref: {path} (skipping)")
        continue
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kp, des = detector.detectAndCompute(g, None)
    if des is None or len(kp) < 8:
        print(f"[warn] Weak features in ref: {path} (skipping)")
        continue
    h, w = g.shape[:2]
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    refs.append({"name": path, "img": img, "gray": g, "kp": kp, "des": des, "corners": corners})

if not refs:
    raise RuntimeError("No valid reference images loaded. Check REF_IMAGES paths.")

# ---- live video ----
cap = cv2.VideoCapture(CAM_INDEX)
if not cap.isOpened():
    raise RuntimeError("Cannot open camera.")
hits = deque(maxlen=SMOOTH_N)

passed = False
pass_started_at = None

print("Press ESC to quit.")
while True:
    ok, frame = cap.read()
    if not ok:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    kp_f, des_f = detector.detectAndCompute(gray, None)

    best = {"inliers": 0, "name": None, "quad": None}
    if des_f is not None and len(kp_f) >= 8:
        # try all references, keep the one with most inliers
        for R in refs:
            # KNN match (ref -> frame)
            knn = bf.knnMatch(R["des"], des_f, k=2)
            good = [m for m, n in knn if m.distance < CONF_RATIO * n.distance]
            if len(good) < MIN_GOOD_MATCHES:
                continue

            src = np.float32([R["kp"][m.queryIdx].pt for m in good]).reshape(-1,1,2)
            dst = np.float32([kp_f[m.trainIdx].pt     for m in good]).reshape(-1,1,2)
            H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

            if H is None or mask is None:
                continue
            inliers = int(mask.sum())
            if inliers > best["inliers"]:
                best["inliers"] = inliers
                best["name"] = R["name"]
                try:
                    proj = cv2.perspectiveTransform(R["corners"], H)
                except cv2.error:
                    proj = None
                best["quad"] = proj

    detected = best["inliers"] >= MIN_INLIERS

    # majority vote smoothing for stable 1/0
    hits.append(detected)
    wafer_present = sum(hits) >= (len(hits)//2 + 1)

    # If wafer ever seen, trigger PASS state (one-time)
    if wafer_present and not passed:
        passed = True
        pass_started_at = time.time()
        print("\nPASS")  # optional extra line
        # also print final 1 immediately
        print(1)

    # Live 1/0 stream until PASS (or keep printing 1 afterwards if you prefer)
    if not passed:
        print(1 if wafer_present else 0, end="\r")
    else:
        # Draw PASS overlay and optionally exit after hold
        if SHOW_DEBUG:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0,0), (frame.shape[1], frame.shape[0]), (0,255,0), -1)
            frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)
            cv2.putText(frame, "PASS", (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,255,0), 8, cv2.LINE_AA)
            cv2.putText(frame, "Wafer detected", (40, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,255,0), 3, cv2.LINE_AA)

        # auto-exit after PASS_HOLD_MS
        if EXIT_ON_PASS and pass_started_at is not None:
            if (time.time() - pass_started_at) * 1000 >= PASS_HOLD_MS:
                break

    if SHOW_DEBUG:
        # Draw best match quadrilateral (before pass), or show text
        if best["quad"] is not None and not passed:
            cv2.polylines(frame, [np.int32(best["quad"])], True,
                          (0,255,0) if detected else (0,0,255), 3, cv2.LINE_AA)
        status = f"wafer: {'YES' if wafer_present else 'NO'} | best inliers: {best['inliers']}"
        if best["name"] and not passed:
            status += f" | match: {best['name']}"
        cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0,255,0) if wafer_present else (0,0,255), 2)
        cv2.imshow("Wafer Multi-Ref", frame)

    # early manual exit
    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
print("\nDone.")
