// fast-wafer-detector-instant.js
// OpenCV.js wafer detector with INSTANT template-gating + fast ORB fallback.
// - Tiny template match first (very fast) → instant PASS if score ≥ threshold
// - If not passed, run downscaled ORB+RANSAC (early-exit)
// - Preallocated Mats, fewer features, 30Hz compute loop

class WaferDetector {
  constructor() {
    // -------- Tunables --------
    // Template-gating (instant path)
    this.INSTANT_TEMPLATE = true; // turn ON for instant gating
    this.TMPL_SIZE = 96;          // template size (square, px)
    this.TMPL_THR  = 0.55;        // TM_CCOEFF_NORMED threshold for instant PASS (0..1)
    this.EDGE_TMPL = false;        // use Canny edges for matching (more robust to lighting)

    // ORB fallback (runs only if instant gate didn't pass)
    this.CONF_RATIO =0.90;    // Lowe ratio
    this.MIN_GOOD_MATCHES = 3;
    this.MIN_INLIERS = 6;

    // Smoothing/UX
    this.SMOOTH_N = 5;            // instant = 1; increase to 2-3 if you want stability
    this.SHOW_DEBUG = true;
    this.EXIT_ON_PASS = false;
    this.PASS_HOLD_MS = 1000;

    // Processing resolution (downscale for speed; display stays full-res)
    this.PROC_W = 320;
    this.PROC_H = 240;

    // Default reference files (place next to HTML)
    this.REF_IMAGE_FILES = [
      "wafer_ref_4.png",
      "wafer_ref_5.png",
      "wafer_ref_1.jpg",
      "wafer_ref_2.jpg"
     ];

    // -------- State --------
    this.refs = [];           // [{name, kp, des, corners, dataUrl, tmpl, res}]
    this.hits = [];
    this.passed = false;
    this.passStartedAt = null;
    this.isRunning = false;
    this.stream = null;
    this.lastQuad = null;     // [x0,y0,...,x3,y3] in processed coords

    // -------- DOM --------
    this.video = document.getElementById("videoElement");
    this.canvas = document.getElementById("canvas");
    this.ctx = this.canvas.getContext("2d");
    this.output = document.getElementById("output");
    this.status = document.getElementById("status");
    this.startBtn = document.getElementById("startBtn");
    this.stopBtn = document.getElementById("stopBtn");
    this.refContainer = document.getElementById("refImagesContainer");

    // Processing canvas
    this.procCanvas = document.createElement("canvas");
    this.procCanvas.width = this.PROC_W;
    this.procCanvas.height = this.PROC_H;
    this.procCtx = this.procCanvas.getContext("2d", { willReadFrequently: true });

    // OpenCV Mats (alloc after cv ready)
    this.frameRGBA = null;
    this.frameGray = null;
    this.frameEdge = null; // for template edge-match

    this.kpFrame = null;
    this.desFrame = null;

    // Boot OpenCV
    this.initializeOpenCV();
  }

  // -------- OpenCV bootstrap --------
  async initializeOpenCV() {
    const waitReady = (resolve) => {
      if (typeof cv === "undefined") return setTimeout(() => waitReady(resolve), 100);
      if (cv.getBuildInformation) resolve();
      else cv.onRuntimeInitialized = resolve;
    };
    await new Promise(waitReady);

    // Detector + matcher (lightweight ORB)
    if (typeof cv.ORB.create === "function") {
      // nfeatures, scaleFactor, nlevels, edgeThreshold, firstLevel, WTA_K, scoreType, patchSize, fastThreshold
      this.detector = cv.ORB.create(1000, 1.2, 6, 16, 0, 2, 0, 16, 12);
    } else {
      this.detector = new cv.ORB(800);
    }
    this.matcher = new cv.BFMatcher(cv.NORM_HAMMING, false);

    // Prealloc mats
    this.frameRGBA = new cv.Mat(this.PROC_H, this.PROC_W, cv.CV_8UC4);
    this.frameGray = new cv.Mat(this.PROC_H, this.PROC_W, cv.CV_8UC1);
    this.frameEdge = new cv.Mat(this.PROC_H, this.PROC_W, cv.CV_8UC1);

    this.kpFrame = new cv.KeyPointVector();
    this.desFrame = new cv.Mat();

    console.log("OpenCV.js ready (instant template gate enabled).");
  }

  // -------- Settings from UI (optional) --------
  updateSettings() {
    try {
      const confRatio = document.getElementById("confRatio");
      const minGood = document.getElementById("minGoodMatches");
      const minInliers = document.getElementById("minInliers");
      const smoothFrames = document.getElementById("smoothFrames");
      if (confRatio) this.CONF_RATIO = parseFloat(confRatio.value);
      if (minGood) this.MIN_GOOD_MATCHES = parseInt(minGood.value);
      if (minInliers) this.MIN_INLIERS = parseInt(minInliers.value);
      if (smoothFrames) this.SMOOTH_N = parseInt(smoothFrames.value);
    } catch {}
  }

  // -------- Reference images --------
  async loadReferenceImages() {
    // cleanup old
    this.refs.forEach(r => {
      try {
        r.kp?.delete(); r.des?.delete(); r.corners?.delete?.();
        r.tmpl?.delete(); r.res?.delete?.();
      } catch {}
    });
    this.refs = [];
    if (this.refContainer) this.refContainer.innerHTML = "";

    const results = await Promise.allSettled(this.REF_IMAGE_FILES.map(f => this._loadRefFromPath(f, f)));
    let loaded = 0;
    for (const res of results) {
      if (res.status === "fulfilled" && res.value) {
        this.refs.push(res.value);
        loaded++;
        if (this.refContainer && res.value.dataUrl) {
          const img = document.createElement("img");
          img.src = res.value.dataUrl;
          img.className = "ref-image";
          img.title = res.value.name;
          this.refContainer.appendChild(img);
        }
      }
    }
    console.log(`Loaded ${loaded} reference images`);
  }

  async _loadRefFromPath(imagePath, name) {
    return new Promise((resolve) => {
      const img = new Image();
      img.onload = () => {
        try {
          const c = document.createElement("canvas");
          const g = c.getContext("2d");
          c.width = img.width; c.height = img.height;
          g.drawImage(img, 0, 0);

          const id = g.getImageData(0, 0, c.width, c.height);
          const src = cv.matFromImageData(id);
          const gray = new cv.Mat();
          cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

          // ORB features
          const kp = new cv.KeyPointVector();
          const des = new cv.Mat();
          this.detector.detectAndCompute(gray, new cv.Mat(), kp, des);

          if (!des || des.rows < 8) {
            console.warn(`[ref weak] ${name}`);
            src.delete(); gray.delete(); kp?.delete(); des?.delete();
            resolve(null);
            return;
          }

          // Corners (for homography visualization)
          const corners = cv.matFromArray(4, 1, cv.CV_32FC2, [
            0, 0, c.width, 0,
            c.width, c.height, 0, c.height
          ]);

          // --- Template for instant gate ---
          let tmpl = new cv.Mat();
          const size = new cv.Size(this.TMPL_SIZE, this.TMPL_SIZE);
          cv.resize(gray, tmpl, size, 0, 0, cv.INTER_AREA);
          if (this.EDGE_TMPL) {
            const edges = new cv.Mat();
            cv.Canny(tmpl, edges, 50, 150, 3, false);
            tmpl.delete();
            tmpl = edges;
          }
          // Preallocate result mat for matchTemplate (depends on frame size and tmpl size)
          const resW = this.PROC_W - tmpl.cols + 1;
          const resH = this.PROC_H - tmpl.rows + 1;
          const res = new cv.Mat(resH, resW, cv.CV_32FC1);

          const dataUrl = c.toDataURL();

          src.delete(); gray.delete();

          resolve({ name, kp, des, corners, tmpl, res, dataUrl });
        } catch (e) {
          console.error("Ref load err:", e);
          resolve(null);
        }
      };
      img.onerror = () => resolve(null);
      img.crossOrigin = "anonymous";
      img.src = "./" + imagePath;
    });
  }

  async loadCustomReferenceImages(files) {
    // cleanup old
    this.refs.forEach(r => {
      try {
        r.kp?.delete(); r.des?.delete(); r.corners?.delete?.();
        r.tmpl?.delete(); r.res?.delete?.();
      } catch {}
    });
    this.refs = [];
    if (this.refContainer) this.refContainer.innerHTML = "";

    for (const file of files) {
      const ref = await this._processRefFile(file);
      if (ref) {
        this.refs.push(ref);
        if (this.refContainer && ref.dataUrl) {
          const img = document.createElement("img");
          img.src = ref.dataUrl;
          img.className = "ref-image";
          img.title = file.name;
          this.refContainer.appendChild(img);
        }
      }
    }
    console.log(`Loaded ${this.refs.length} custom refs`);
  }

  _processRefFile(file) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        try {
          const c = document.createElement("canvas");
          const g = c.getContext("2d");
          c.width = img.width; c.height = img.height;
          g.drawImage(img, 0, 0);

          const id = g.getImageData(0, 0, c.width, c.height);
          const src = cv.matFromImageData(id);
          const gray = new cv.Mat();
          cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

          const kp = new cv.KeyPointVector();
          const des = new cv.Mat();
          this.detector.detectAndCompute(gray, new cv.Mat(), kp, des);

          if (!des || des.rows < 8) {
            console.warn(`[ref weak] ${file.name}`);
            src.delete(); gray.delete(); kp?.delete(); des?.delete();
            resolve(null);
            return;
          }

          const corners = cv.matFromArray(4, 1, cv.CV_32FC2, [
            0, 0, c.width, 0,
            c.width, c.height, 0, c.height
          ]);

          // Template
          let tmpl = new cv.Mat();
          const size = new cv.Size(this.TMPL_SIZE, this.TMPL_SIZE);
          cv.resize(gray, tmpl, size, 0, 0, cv.INTER_AREA);
          if (this.EDGE_TMPL) {
            const edges = new cv.Mat();
            cv.Canny(tmpl, edges, 50, 150, 3, false);
            tmpl.delete();
            tmpl = edges;
          }
          const resW = this.PROC_W - tmpl.cols + 1;
          const resH = this.PROC_H - tmpl.rows + 1;
          const res = new cv.Mat(resH, resW, cv.CV_32FC1);

          const dataUrl = c.toDataURL();

          src.delete(); gray.delete();

          resolve({ name: file.name, kp, des, corners, tmpl, res, dataUrl });
        } catch (e) { reject(e); }
      };
      img.onerror = reject;
      img.src = URL.createObjectURL(file);
    });
  }

  // -------- Start/Stop --------
  async startDetection() {
    if (!this.refs.length) {
      alert("Load reference images first.");
      return;
    }
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, frameRate: { ideal: 30, max: 30 } }
      });
      this.video.srcObject = this.stream;

      this.video.onloadedmetadata = () => {
        this.isRunning = true;
        this.passed = false;
        this.passStartedAt = null;
        this.hits = [];
        this.lastQuad = null;

        this.updateSettings();
        this.startBtn && (this.startBtn.disabled = true);
        this.stopBtn && (this.stopBtn.disabled = false);

        this._startLiveFeed();
        this._startDetectionLoop();
      };
    } catch (e) {
      console.error("camera err:", e);
      alert("Camera error. Check permissions or device.");
    }
  }

  stopDetection() {
    this.isRunning = false;
    if (this.stream) {
      this.stream.getTracks().forEach(t => t.stop());
      this.stream = null;
    }
    this.startBtn && (this.startBtn.disabled = false);
    this.stopBtn && (this.stopBtn.disabled = true);
    this.updateStatus(false, { inliers: 0 }, false);
    this.updateOutput("0");
    this.lastQuad = null;
  }

  // -------- Loops --------
  _startLiveFeed() {
    const draw = () => {
      if (!this.isRunning) return;

      // draw camera
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      this.ctx.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);

      // draw quad
      if (this.SHOW_DEBUG && this.lastQuad) {
        const sx = this.canvas.width / this.PROC_W;
        const sy = this.canvas.height / this.PROC_H;
        this.ctx.strokeStyle = "rgb(0,255,0)";
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        for (let i = 0; i < 4; i++) {
          const x = this.lastQuad[i * 2] * sx;
          const y = this.lastQuad[i * 2 + 1] * sy;
          if (i === 0) this.ctx.moveTo(x, y);
          else this.ctx.lineTo(x, y);
        }
        this.ctx.closePath();
        this.ctx.stroke();
      }

      // PASS overlay
      if (this.passed) {
        this.ctx.fillStyle = "rgba(0, 255, 0, 0.35)";
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        this.ctx.fillStyle = "rgb(0,255,0)";
        this.ctx.font = "bold 40px Arial";
        this.ctx.fillText("PASS", 20, 60);
        this.ctx.font = "bold 16px Arial";
        this.ctx.fillText("Wafer detected", 20, 90);
      }

      requestAnimationFrame(draw);
    };
    requestAnimationFrame(draw);
  }

  _startDetectionLoop() {
    let lastTS = 0;
    const targetDelta = 1000 / 30; // ~30Hz compute

    const loop = (ts) => {
      if (!this.isRunning) return;
      if (ts - lastTS < targetDelta) return requestAnimationFrame(loop);
      lastTS = ts;

      try {
        // 1) get downscaled frame
        this.procCtx.drawImage(this.video, 0, 0, this.PROC_W, this.PROC_H);
        const id = this.procCtx.getImageData(0, 0, this.PROC_W, this.PROC_H);

        // 2) RGBA -> Gray (+ Edge for template)
        this.frameRGBA.data.set(id.data);
        cv.cvtColor(this.frameRGBA, this.frameGray, cv.COLOR_RGBA2GRAY);
        if (this.EDGE_TMPL) {
          cv.Canny(this.frameGray, this.frameEdge, 50, 150, 3, false);
        }

        let waferPresent = false;
        let bestInliers = 0;
        let bestQuad = null;
        let templateHit = false;

        // -------- INSTANT TEMPLATE GATE --------
        if (this.INSTANT_TEMPLATE) {
          const srcForMatch = this.EDGE_TMPL ? this.frameEdge : this.frameGray;
          for (const ref of this.refs) {
            if (!ref.tmpl || !ref.res) continue;
            cv.matchTemplate(srcForMatch, ref.tmpl, ref.res, cv.TM_CCOEFF_NORMED);
            const mm = cv.minMaxLoc(ref.res);
            console.log(`[TEMPLATE] ${ref.name} maxVal=${mm.maxVal.toFixed(3)} thr=${this.TMPL_THR}`);
            if (mm.maxVal >= this.TMPL_THR) {  
            templateHit = true; // instant PASS
    }

          }
        }

        // -------- ORB FALLBACK (only if not already passed) --------
        if (!waferPresent) {
          // detect/compute features
          this.kpFrame.delete(); this.kpFrame = new cv.KeyPointVector();
          this.desFrame.delete(); this.desFrame = new cv.Mat();
          this.detector.detectAndCompute(this.frameGray, new cv.Mat(), this.kpFrame, this.desFrame);

          if (this.desFrame.rows >= 8) {
            for (const ref of this.refs) {
              const matches = new cv.DMatchVectorVector();
              this.matcher.knnMatch(ref.des, this.desFrame, matches, 2);

              // ratio test
              const good = [];
              for (let i = 0; i < matches.size(); i++) {
                const mv = matches.get(i);
                if (mv.size() === 2) {
                  const m = mv.get(0), n = mv.get(1);
                  if (m.distance < this.CONF_RATIO * n.distance) good.push(m);
                }
              }
              matches.delete();
              if (good.length < this.MIN_GOOD_MATCHES) continue;

              // build point arrays
              const srcPts = new Float32Array(good.length * 2);
              const dstPts = new Float32Array(good.length * 2);
              for (let i = 0; i < good.length; i++) {
                const g = good[i];
                const rk = ref.kp.get(g.queryIdx);
                const fk = this.kpFrame.get(g.trainIdx);
                srcPts[i*2] = rk.pt.x;   srcPts[i*2+1] = rk.pt.y;
                dstPts[i*2] = fk.pt.x;   dstPts[i*2+1] = fk.pt.y;
              }
              const srcMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, srcPts);
              const dstMat = cv.matFromArray(good.length, 1, cv.CV_32FC2, dstPts);

              const mask = new cv.Mat();
              try {
                const H = cv.findHomography(srcMat, dstMat, cv.RANSAC, 5.0, mask);
                if (!H.empty()) {
                  const inliers = cv.countNonZero(mask);
                  console.log(`[ORB] ref=${ref.name} good=${good.length} inliers=${inliers}`);
                  if (inliers > bestInliers) {
                    bestInliers = inliers;
                    if (this.SHOW_DEBUG) {
                      const quad = new cv.Mat();
                      cv.perspectiveTransform(ref.corners, quad, H);
                      bestQuad = Array.from(quad.data32F);
                      quad.delete();
                    }
                  }
                  H.delete();
                }
              } catch {}
              srcMat.delete(); dstMat.delete(); mask.delete();

              if (bestInliers >= this.MIN_INLIERS) break; // early-exit
            }
          }

          waferPresent = (templateHit && bestInliers >= this.MIN_INLIERS);
        }

        // -------- Decision + smoothing --------
        this.hits.push(waferPresent);
        if (this.hits.length > this.SMOOTH_N) this.hits.shift();
        const decided = this.hits.some(Boolean); // N=1 → instant

        // PASS state
        if (decided && !this.passed) {
          this.passed = true;
          this.passStartedAt = performance.now();
          console.log("PASS (instant gate:", this.INSTANT_TEMPLATE, ")");
        }

        // Draw quad only if we used ORB
        this.lastQuad = (this.SHOW_DEBUG && bestQuad && !this.passed) ? bestQuad : null;

        // UI text
        this.updateOutput(decided || this.passed ? "1" : "0");
        this.updateStatus(decided, { inliers: bestInliers }, this.passed);

        // optional auto-exit
        if (this.EXIT_ON_PASS && this.passed && this.passStartedAt) {
          if (performance.now() - this.passStartedAt >= this.PASS_HOLD_MS) {
            this.stopDetection();
            return;
          }
        }
      } catch (e) {
        console.error("detect loop error:", e);
      }

      requestAnimationFrame(loop);
    };

    requestAnimationFrame(loop);
  }

  // -------- UI helpers --------
  updateOutput(v) { this.output && (this.output.textContent = v); }
  updateStatus(waferPresent, best, passed) {
    if (!this.status) return;
    if (passed) {
      this.status.textContent = "PASS - Wafer Detected Instantly!";
      this.status.className = "status pass";
    } else {
      const t = `Wafer: ${waferPresent ? "YES" : "NO"} | Best inliers: ${best?.inliers || 0}`;
      this.status.textContent = t;
      this.status.className = waferPresent ? "status detected" : "status not-detected";
    }
  }
}

// ---- Global helpers for your buttons ----
let detector;
function loadReferenceImages(){ detector?.loadReferenceImages(); }
function loadCustomImages(){
  const fileInput = document.getElementById("refImages");
  if (!fileInput) return;
  fileInput.style.display = "block";
  fileInput.click();
  fileInput.onchange = () => {
    if (detector && fileInput.files.length > 0) {
      detector.loadCustomReferenceImages(fileInput.files);
    }
    fileInput.style.display = "none";
  };
}
function startDetection(){ detector?.startDetection(); }
function stopDetection(){ detector?.stopDetection(); }

window.addEventListener("load", () => { detector = new WaferDetector(); });
