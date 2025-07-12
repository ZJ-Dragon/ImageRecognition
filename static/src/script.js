

/**
 * Front‑end logic for the Cloud Face‑Recognition demo.
 * Handles:
 *   • Camera preview via getUserMedia
 *   • Capture still frame on button click
 *   • POST image to /infer
 *   • Display annotated result & detection count
 */

const videoEl = document.getElementById('v');
const btnShot = document.getElementById('shot');
const msgEl   = document.getElementById('msg');
const imgRes  = document.getElementById('res');

// ---- 1. Start camera preview (rear if possible) -------------------
navigator.mediaDevices.getUserMedia({ video: { facingMode: 'environment' } })
  .then(stream => {
    videoEl.srcObject = stream;
  })
  .catch(err => {
    msgEl.textContent = `Camera error: ${err}`;
  });

// ---- 2. Capture & upload ------------------------------------------
btnShot.addEventListener('click', () => {
  if (!videoEl.videoWidth) {
    msgEl.textContent = 'Camera not ready';
    return;
  }
  // Draw current video frame to hidden canvas
  const canvas = document.createElement('canvas');
  canvas.width  = videoEl.videoWidth;
  canvas.height = videoEl.videoHeight;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(videoEl, 0, 0);

  canvas.toBlob(async blob => {
    const fd = new FormData();
    fd.append('file', blob, 'frame.jpg');

    msgEl.textContent = 'Uploading…';
    try {
      const resp = await fetch('/infer', { method: 'POST', body: fd });
      const data = await resp.json();
      if (data.success) {
        imgRes.src = `${data.img}?t=${Date.now()}`; // bust cache
        msgEl.textContent = `Detections: ${data.detections.length}`;
      } else {
        msgEl.textContent = `Server error: ${data.error || 'unknown'}`;
      }
    } catch (e) {
      msgEl.textContent = `Network error: ${e}`;
    }
  }, 'image/jpeg', 0.9);
});