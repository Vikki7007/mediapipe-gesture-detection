// // ---- Lip landmark indices (468-point Face Mesh) ----
// // Using only INNER lips (upper + lower)
// const LIP_INDICES = [
//   // inner upper
//   78, 191, 80, 81, 82, 13, 312, 311, 310,
//   // inner lower
//   178, 88, 95, 402, 318, 324, 308
// ];

// // ---- DOM ----
// const video = document.getElementById('video');
// const canvas = document.getElementById('canvas');
// const ctx = canvas.getContext('2d');

// // ---- State to merge results from both models ----
// let latestFace = null;
// let latestHands = null;

// // ---- Helpers ----
// function getLipPoints(landmarks, w, h) {
//   return LIP_INDICES.map(i => ({
//     x: Math.round(landmarks[i].x * w),
//     y: Math.round(landmarks[i].y * h)
//   }));
// }

// function drawLipPoints(points) {
//   for (const p of points) {
//     ctx.beginPath();
//     ctx.arc(p.x, p.y, 2, 0, Math.PI * 2);
//     ctx.fillStyle = '#00ff88';
//     ctx.fill();
//   }
// }

// function euclideanDistance(p1, p2) {
//   const dx = p2.x - p1.x, dy = p2.y - p1.y;
//   return Math.hypot(dx, dy);
// }

// function getCenterPoint(points) {
//   let sx = 0, sy = 0;
//   for (const p of points) { sx += p.x; sy += p.y; }
//   return { x: sx / points.length, y: sy / points.length };
// }

// // Draw a single hand using drawing_utils’ globals
// function drawHand(lm) {
//   if (!lm) return;
//   drawConnectors(ctx, lm, HAND_CONNECTIONS, { lineWidth: 2 });
//   drawLandmarks(ctx, lm, { radius: 2 });
// }

// // ---- Chew detection config/state ----
// const EAT_WINDOW_MS = 8000; // 8s window
// const CHEW_TARGET   = 2;    // need >=2 chews in window
// const OPEN_THR      = 0.08; // mouth open threshold (ratio)
// const CLOSE_THR     = 0.04; // mouth close threshold (ratio; lower than OPEN_THR)

// let mouthState = "closed";   // "open" | "closed"
// let chewEvents = [];         // timestamps (ms) of each close event
// let eatingDetected = false;
// function pushChewEvent(ts) {
//   chewEvents.push(ts);
//   const cutoff = ts - EAT_WINDOW_MS;
//   while (chewEvents.length && chewEvents[0] < cutoff) chewEvents.shift();
//   eatingDetected = chewEvents.length >= CHEW_TARGET;
// }

// // ---- Wafer-to-mouth gating (both tips in 10–30 px band for hold time) ----
// const CONTACT_REQUIRED_MS = 1000; // hold time (ms)
// const TOUCH_MIN_PX = 10;          // inner radius of acceptable band
// const TOUCH_MAX_PX = 40;          // outer radius of acceptable band
// let waferTaken = false;           // becomes true after hold completes
// let contactStartTs = null;        // when both tips first within band
// let lastHoldMs = 0;               // for on-screen progress

// // ---- FaceMesh setup ----
// const faceMesh = new FaceMesh({
//   locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
// });
// faceMesh.setOptions({
//   maxNumFaces: 1,
//   refineLandmarks: true,
//   minDetectionConfidence: 0.5,
//   minTrackingConfidence: 0.5,
// });
// faceMesh.onResults((results) => { latestFace = results; });

// // ---- Hands setup (SINGLE HAND) ----
// const hands = new Hands({
//   locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
// });
// hands.setOptions({
//   maxNumHands: 1,                // <<< single hand
//   minDetectionConfidence: 0.5,
//   minTrackingConfidence: 0.5,
//   modelComplexity: 1,
// });
// hands.onResults((results) => { latestHands = results; });

// // ---- Camera loop ----
// const cam = new Camera(video, {
//   onFrame: async () => {
//     // Ensure canvas matches current video frame
//     if (video.videoWidth && video.videoHeight) {
//       canvas.width = video.videoWidth;
//       canvas.height = video.videoHeight;
//     }

//     // Send frame to both models
//     await faceMesh.send({ image: video });
//     await hands.send({ image: video });

//     // Clear and draw base video frame
//     ctx.clearRect(0, 0, canvas.width, canvas.height);
//     ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

//     // ---- LIPS: detect and draw lip center ----
//     let lipCenter = null;
//     let openness = 0;
//     if (latestFace?.multiFaceLandmarks?.length) {
//       const lm = latestFace.multiFaceLandmarks[0];
//       const lips = getLipPoints(lm, canvas.width, canvas.height);
//       drawLipPoints(lips);

//       lipCenter = getCenterPoint(lips);

//       // Draw lip center
//       ctx.beginPath();
//       ctx.arc(lipCenter.x, lipCenter.y, 4, 0, Math.PI * 2);
//       ctx.fillStyle = '#ff00ff';
//       ctx.fill();

//       // ---- Chew detection (open/close cycles) ----
//       const pUpper = { x: lm[13].x * canvas.width,  y: lm[13].y * canvas.height };
//       const pLower = { x: lm[14].x * canvas.width,  y: lm[14].y * canvas.height };
//       const pLeft  = { x: lm[61].x * canvas.width,  y: lm[61].y * canvas.height };
//       const pRight = { x: lm[291].x * canvas.width, y: lm[291].y * canvas.height };

//       const mouthWidth = Math.max(1, euclideanDistance(pLeft, pRight));
//       const mouthGap   = euclideanDistance(pUpper, pLower);
//       openness         = mouthGap / mouthWidth;

//       const ts = performance.now();
//       if (mouthState === "closed" && openness > OPEN_THR) {
//         mouthState = "open";
//       } else if (mouthState === "open" && openness < CLOSE_THR) {
//         mouthState = "closed";
//         if (waferTaken) pushChewEvent(ts);
//       }
//     }

//     // ---- HAND (single) : draw and connect thumb + index to lips ----
//     if (latestHands?.multiHandLandmarks?.length && lipCenter) {
//       // Visualize the annulus band around lips
//       ctx.beginPath();
//       ctx.arc(lipCenter.x, lipCenter.y, TOUCH_MIN_PX, 0, Math.PI * 2);
//       ctx.strokeStyle = '#ffaa00';
//       ctx.lineWidth = 1;
//       ctx.stroke();

//       ctx.beginPath();
//       ctx.arc(lipCenter.x, lipCenter.y, TOUCH_MAX_PX, 0, Math.PI * 2);
//       ctx.setLineDash([4, 4]);
//       ctx.strokeStyle = '#ffaa00';
//       ctx.stroke();
//       ctx.setLineDash([]);

//       // Use ONLY the first hand
//       const handLm = latestHands.multiHandLandmarks[0];
//       drawHand(handLm);

//       // Index Tip (#8) and Thumb Tip (#4)
//       const indexTip = {
//         x: handLm[8].x * canvas.width,
//         y: handLm[8].y * canvas.height
//       };
//       const thumbTip = {
//         x: handLm[4].x * canvas.width,
//         y: handLm[4].y * canvas.height
//       };

//       // Draw fingertip points
//       ctx.beginPath();
//       ctx.arc(indexTip.x, indexTip.y, 4, 0, Math.PI * 2);
//       ctx.fillStyle = '#ffff00';
//       ctx.fill();

//       ctx.beginPath();
//       ctx.arc(thumbTip.x, thumbTip.y, 4, 0, Math.PI * 2);
//       ctx.fillStyle = '#00ffff';
//       ctx.fill();

//       // Distances from lip center
//       const indexDistance = euclideanDistance(lipCenter, indexTip);
//       const thumbDistance = euclideanDistance(lipCenter, thumbTip);

//       // Lines to lips
//       ctx.beginPath();
//       ctx.moveTo(lipCenter.x, lipCenter.y);
//       ctx.lineTo(indexTip.x, indexTip.y);
//       ctx.strokeStyle = '#ff0000'; // red line for index
//       ctx.lineWidth = 2;
//       ctx.stroke();

//       ctx.beginPath();
//       ctx.moveTo(lipCenter.x, lipCenter.y);
//       ctx.lineTo(thumbTip.x, thumbTip.y);
//       ctx.strokeStyle = '#00ff00'; // green line for thumb
//       ctx.lineWidth = 2;
//       ctx.stroke();

//       // Labels
//       ctx.fillStyle = '#ffffff';
//       ctx.font = '14px system-ui';
//       ctx.fillText(`Index: ${indexDistance.toFixed(1)} px`, 10, canvas.height - 40);
//       ctx.fillText(`Thumb: ${thumbDistance.toFixed(1)} px`, 10, canvas.height - 20);

//       // BOTH fingertips must be within the fixed band (10–30 px)
//       const indexInRange = (indexDistance >= TOUCH_MIN_PX && indexDistance <= TOUCH_MAX_PX);
//       const thumbInRange = (thumbDistance >= TOUCH_MIN_PX && thumbDistance <= TOUCH_MAX_PX);
//       const holding = indexInRange && thumbInRange;

//       // ----- hold gate for "waffer taken to the mouth" -----
//       const now = performance.now();
//       if (!waferTaken) {
//         if (holding) {
//           if (contactStartTs == null) contactStartTs = now;
//           lastHoldMs = now - contactStartTs;

//           // progress text
//           const secs = Math.min(CONTACT_REQUIRED_MS, lastHoldMs) / 1000;
//           ctx.fillStyle = "#ffd24d";
//           ctx.font = "bold 16px system-ui";
//           ctx.fillText(
//             `Hold near lips: ${secs.toFixed(1)} / ${(CONTACT_REQUIRED_MS/1000).toFixed(1)} s`,
//             10, 24
//           );

//           if (lastHoldMs >= CONTACT_REQUIRED_MS) {
//             waferTaken = true; // enable chew counting
//           }
//         } else {
//           // broke contact -> reset timer
//           contactStartTs = null;
//           lastHoldMs = 0;
//         }
//       }

//       // Status after gating complete
//       if (waferTaken) {
//         ctx.fillStyle = "#7cff8e";
//         ctx.font = "bold 16px system-ui";
//         ctx.fillText("WAFFER TAKEN TO MOUTH ✔  (chew counting active)", 10, 24);
//       }
//     } else {
//       // If no hand or no lips, reset contact timer (but keep waferTaken once true)
//       contactStartTs = null;
//       lastHoldMs = 0;
//     }

//     // ---- Debug readouts (bottom-left) ----
//     if (lipCenter) {
//       ctx.fillStyle = "#ffffff";
//       ctx.font = "14px system-ui";
//       ctx.fillText(`Chews (8s): ${chewEvents.length}`, 10, canvas.height - 42);
//       ctx.fillText(`Mouth openness: ${openness.toFixed(3)}`, 10, canvas.height - 60);
//       if (eatingDetected) {
//         ctx.fillStyle = "#00ffa6";
//         ctx.font = "bold 18px system-ui";
//         ctx.fillText("EATING ✔", 10, canvas.height - 20);
//       }
//     }
//   },
//   width: 640,
//   height: 480,
// });

// cam.start();


// ---- Lip landmark indices (468-point Face Mesh) ----
// Using only INNER lips (upper + lower)
const LIP_INDICES = [
  // inner upper
  78, 191, 80, 81, 82, 13, 312, 311, 310,
  // inner lower
  178, 88, 95, 402, 318, 324, 308
];

// ---- DOM ----
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// ---- State to merge results from both models ----
let latestFace = null;
let latestHands = null;

// ---- Helpers ----
function getLipPoints(landmarks, w, h) {
  return LIP_INDICES.map(i => ({
    x: Math.round(landmarks[i].x * w),
    y: Math.round(landmarks[i].y * h)
  }));
}
function euclideanDistance(p1, p2) {
  const dx = p2.x - p1.x, dy = p2.y - p1.y;
  return Math.hypot(dx, dy);
}
function getCenterPoint(points) {
  let sx = 0, sy = 0;
  for (const p of points) { sx += p.x; sy += p.y; }
  return { x: sx / points.length, y: sy / points.length };
}

// ---- Chew detection config/state ----
const EAT_WINDOW_MS = 8000; // 8s window
const CHEW_TARGET   = 2;    // need >=2 chews in window
const OPEN_THR      = 0.08; // mouth open threshold (ratio)
const CLOSE_THR     = 0.04; // mouth close threshold (ratio; lower than OPEN_THR)

let mouthState = "closed";   // "open" | "closed"
let chewEvents = [];         // timestamps (ms) of each close event
let eatingDetected = false;

function pushChewEvent(ts) {
  chewEvents.push(ts);
  const cutoff = ts - EAT_WINDOW_MS;
  while (chewEvents.length && chewEvents[0] < cutoff) chewEvents.shift();
  const wasEating = eatingDetected;
  eatingDetected = chewEvents.length >= CHEW_TARGET;
  if (!wasEating && eatingDetected) console.log("EATING ✔");
}

// ---- Wafer-to-mouth gating (both tips in 10–40 px band for hold time) ----
const CONTACT_REQUIRED_MS = 500; // hold time (ms)
const TOUCH_MIN_PX = 10;          // inner radius of acceptable band
const TOUCH_MAX_PX = 40;          // outer radius of acceptable band
let waferTaken = false;           // becomes true after hold completes
let contactStartTs = null;        // when both tips first within band
let lastHoldMs = 0;               // progress
let holdingPrev = false;          // for transition logs

// ---- FaceMesh setup ----
const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`,
});
faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});
faceMesh.onResults((results) => { latestFace = results; });

// ---- Hands setup (SINGLE HAND) ----
const hands = new Hands({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
});
hands.setOptions({
  maxNumHands: 1,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
  modelComplexity: 1,
});
hands.onResults((results) => { latestHands = results; });

// ---- Camera loop ----
const cam = new Camera(video, {
  onFrame: async () => {
    // Ensure canvas matches current video frame
    if (video.videoWidth && video.videoHeight) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    // Send frame to both models
    await faceMesh.send({ image: video });
    await hands.send({ image: video });

    // Draw base video frame (labels only on top)
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // ---- Face/Hand presence labels ----
    const faceOk = !!latestFace?.multiFaceLandmarks?.length;
    const handOk = !!latestHands?.multiHandLandmarks?.length;
    ctx.fillStyle = "#ffffff";
    ctx.font = "14px system-ui";
    ctx.fillText(`Face: ${faceOk ? "✓" : "–"}  Hand: ${handOk ? "✓" : "–"}`, 10, 20);

    // ---- LIPS: compute lip center & openness (no drawing of points/lines) ----
    let lipCenter = null;
    let openness = 0;
    if (faceOk) {
      const lm = latestFace.multiFaceLandmarks[0];
      const lips = getLipPoints(lm, canvas.width, canvas.height);
      lipCenter = getCenterPoint(lips);

      // Chew detection (open/close cycles)
      const pUpper = { x: lm[13].x * canvas.width,  y: lm[13].y * canvas.height };
      const pLower = { x: lm[14].x * canvas.width,  y: lm[14].y * canvas.height };
      const pLeft  = { x: lm[61].x * canvas.width,  y: lm[61].y * canvas.height };
      const pRight = { x: lm[291].x * canvas.width, y: lm[291].y * canvas.height };

      const mouthWidth = Math.max(1, euclideanDistance(pLeft, pRight));
      const mouthGap   = euclideanDistance(pUpper, pLower);
      openness         = mouthGap / mouthWidth;

      const ts = performance.now();
      if (mouthState === "closed" && openness > OPEN_THR) {
        mouthState = "open";
      } else if (mouthState === "open" && openness < CLOSE_THR) {
        mouthState = "closed";
        if (waferTaken) pushChewEvent(ts);
      }
    }

    // ---- HAND (single): gating via band check (labels only) ----
    let indexDistance = null, thumbDistance = null;
    if (handOk && lipCenter) {
      const handLm = latestHands.multiHandLandmarks[0];
      const indexTip = { x: handLm[8].x * canvas.width, y: handLm[8].y * canvas.height };
      const thumbTip = { x: handLm[4].x * canvas.width, y: handLm[4].y * canvas.height };

      indexDistance = euclideanDistance(lipCenter, indexTip);
      thumbDistance = euclideanDistance(lipCenter, thumbTip);

      const indexInRange = (indexDistance >= TOUCH_MIN_PX && indexDistance <= TOUCH_MAX_PX);
      const thumbInRange = (thumbDistance >= TOUCH_MIN_PX && thumbDistance <= TOUCH_MAX_PX);
      const holding = indexInRange && thumbInRange;

      const now = performance.now();
      if (!waferTaken) {
        if (holding) {
          if (!holdingPrev) console.log("Hold started (tips near lips)");
          holdingPrev = true;
          if (contactStartTs == null) contactStartTs = now;
          lastHoldMs = now - contactStartTs;
          if (lastHoldMs >= CONTACT_REQUIRED_MS) {
            waferTaken = true;
            console.log("WAFFER TAKEN TO MOUTH ✔  (chew counting active)");
          }
        } else {
          if (holdingPrev) console.log("Hold reset");
          holdingPrev = false;
          contactStartTs = null;
          lastHoldMs = 0;
        }
      }

      // Labels for distances and hold progress
      ctx.fillStyle = "#ffffff";
      ctx.font = "14px system-ui";
      ctx.fillText(`Index distance: ${indexDistance.toFixed(1)} px`, 10, canvas.height - 60);
      ctx.fillText(`Thumb distance: ${thumbDistance.toFixed(1)} px`, 10, canvas.height - 40);

      if (!waferTaken && holding) {
        const secs = Math.min(CONTACT_REQUIRED_MS, lastHoldMs) / 1000;
        ctx.fillStyle = "#ffd24d";
        ctx.font = "bold 16px system-ui";
        ctx.fillText(`Hold near lips: ${secs.toFixed(1)} / ${(CONTACT_REQUIRED_MS/1000).toFixed(1)} s`, 10, 40);
      }
    } else {
      // If no hand or no lips, reset contact timer (but keep waferTaken once true)
      if (contactStartTs !== null) console.log("Hold reset (lost hand/face)");
      contactStartTs = null;
      lastHoldMs = 0;
      holdingPrev = false;
    }

    // ---- Status labels ----
    if (waferTaken) {
      ctx.fillStyle = "#7cff8e";
      ctx.font = "bold 16px system-ui";
      ctx.fillText("WAFFER TAKEN TO MOUTH ✔ (chew counting active)", 10, 40);
    }
    if (lipCenter) {
      ctx.fillStyle = "#ffffff";
      ctx.font = "14px system-ui";
      ctx.fillText(`Chews (last ${EAT_WINDOW_MS/1000}s): ${chewEvents.length}`, 10, canvas.height - 20);
      ctx.fillText(`Mouth openness: ${openness.toFixed(3)}`, 10, canvas.height - 80);
      if (eatingDetected) {
        ctx.fillStyle = "#00ffa6";
        ctx.font = "bold 18px system-ui";
        ctx.fillText("EATING ✔", 10, canvas.height - 100);
      }
    }
  },
  width: 640,
  height: 480,
});

cam.start();
