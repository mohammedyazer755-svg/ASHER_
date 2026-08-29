import {
  FilesetResolver,
  HandLandmarker,
} from "./vendor/mediapipe/vision_bundle.mjs";

const WASM_ROOT = new URL("./vendor/mediapipe/wasm", import.meta.url)
  .toString()
  .replace(/\/$/, "");
const MODEL_URL = new URL("./assets/hand_landmarker.task", import.meta.url).toString();
const MAX_TRACKING_FPS = 30;

function finiteCoordinate(value) {
  const number = Number(value);
  return Number.isFinite(number) ? Math.max(-2, Math.min(2, number)) : 0;
}

/**
 * Local camera/MediaPipe lifecycle only. Gesture meaning is deliberately
 * interpreted in Python by a presentation-only object with no controller.
 */
export class LocalHandTracker {
  constructor(video, onFrame, onStatus) {
    this.video = video;
    this.onFrame = onFrame;
    this.onStatus = onStatus;
    this.landmarker = null;
    this.stream = null;
    this.running = false;
    this.starting = false;
    this.rafId = 0;
    this.lastVideoTime = -1;
    this.lastInferenceAt = 0;
  }

  async start() {
    if (this.running || this.starting) return;
    this.starting = true;
    this.onStatus({ hands: 0, mode: "idle", state: "starting", error: "" });
    try {
      const vision = await FilesetResolver.forVisionTasks(WASM_ROOT);
      const common = {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "GPU" },
        runningMode: "VIDEO",
        numHands: 2,
        minHandDetectionConfidence: 0.62,
        minHandPresenceConfidence: 0.62,
        minTrackingConfidence: 0.62,
      };
      try {
        this.landmarker = await HandLandmarker.createFromOptions(vision, common);
      } catch (_gpuError) {
        this.landmarker = await HandLandmarker.createFromOptions(vision, {
          ...common,
          baseOptions: { ...common.baseOptions, delegate: "CPU" },
        });
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640 },
          height: { ideal: 480 },
          frameRate: { ideal: 30, max: 30 },
          facingMode: "user",
        },
        audio: false,
      });
      this.video.srcObject = this.stream;
      await this.video.play();
      this.running = true;
      this.starting = false;
      this.lastVideoTime = -1;
      this.lastInferenceAt = 0;
      this.onStatus({ hands: 0, mode: "idle", state: "on", error: "" });
      this._loop();
    } catch (error) {
      const denied = error instanceof DOMException && error.name === "NotAllowedError";
      const message = denied
        ? "Camera access was denied"
        : `Local hand tracking failed: ${String(error?.message || error).slice(0, 180)}`;
      this.stop(false);
      this.onStatus({ hands: 0, mode: "idle", state: "error", error: message });
      throw error;
    }
  }

  _loop = (timestamp = performance.now()) => {
    if (!this.running) return;
    this.rafId = requestAnimationFrame(this._loop);
    if (!this.landmarker || this.video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) {
      return;
    }
    if (this.video.currentTime === this.lastVideoTime) return;
    if (timestamp - this.lastInferenceAt < 1000 / MAX_TRACKING_FPS) return;
    this.lastVideoTime = this.video.currentTime;
    this.lastInferenceAt = timestamp;

    try {
      const result = this.landmarker.detectForVideo(this.video, timestamp);
      const hands = result.landmarks.slice(0, 2).map((landmarks, index) => ({
        label:
          result.handedness[index]?.[0]?.categoryName ||
          result.handedness[index]?.[0]?.displayName ||
          `hand-${index}`,
        landmarks: landmarks.map((point) => ({
          x: finiteCoordinate(point.x),
          y: finiteCoordinate(point.y),
          z: finiteCoordinate(point.z),
        })),
      }));
      this.onFrame({ hands });
      this.onStatus({ hands: hands.length, mode: "idle", state: "on", error: "" });
    } catch (error) {
      this.onStatus({
        hands: 0,
        mode: "idle",
        state: "error",
        error: `Tracking frame failed: ${String(error?.message || error).slice(0, 160)}`,
      });
    }
  };

  stop(emitStatus = true) {
    this.running = false;
    this.starting = false;
    cancelAnimationFrame(this.rafId);
    this.rafId = 0;
    try {
      this.landmarker?.close();
    } catch (_error) {
      // The camera still has to be released if MediaPipe teardown complains.
    }
    this.landmarker = null;
    this.stream?.getTracks().forEach((track) => track.stop());
    this.stream = null;
    try {
      this.video.pause();
    } catch (_error) {
      // A never-started hidden video has nothing to pause.
    }
    this.video.srcObject = null;
    this.lastVideoTime = -1;
    if (emitStatus) {
      this.onStatus({ hands: 0, mode: "idle", state: "off", error: "" });
    }
  }

  dispose() {
    this.stop(false);
    this.onFrame = () => {};
    this.onStatus = () => {};
  }
}
