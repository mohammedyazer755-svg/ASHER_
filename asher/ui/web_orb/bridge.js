import { createAsherOrbRenderer } from "./orb.js";
import { LocalHandTracker } from "./hand_tracker.js";

const container = document.getElementById("orb-root");
const stateNode = document.getElementById("state");
const messageNode = document.getElementById("message");
const errorNode = document.getElementById("renderer-error");
const video = document.getElementById("camera-source");

let renderer = null;
let tracker = null;
let channel = null;
let bridge = null;
let active = true;
let gestureRequested = false;
let lastTrackerStatus = "";

function safeText(value, maximum = 240) {
  return String(value || "").replace(/[\u0000-\u001f]/g, " ").slice(0, maximum);
}

function showError(message) {
  const text = safeText(message, 500);
  errorNode.textContent = text;
  errorNode.style.display = text ? "block" : "none";
}

function reportRendererError(error) {
  const message = safeText(error?.message || error || "WebGL renderer failed", 500);
  showError(message);
  if (bridge) bridge.rendererError(message);
}

function reportTrackerStatus(status) {
  const payload = JSON.stringify({
    hands: Math.max(0, Math.min(2, Number(status.hands) || 0)),
    mode: ["idle", "spin", "unfold"].includes(status.mode) ? status.mode : "idle",
    state: safeText(status.state, 24),
    error: safeText(status.error, 240),
  });
  if (payload !== lastTrackerStatus && bridge) {
    lastTrackerStatus = payload;
    bridge.trackerStatus(payload);
  }
}

function ensureTracker() {
  if (!tracker) {
    tracker = new LocalHandTracker(
      video,
      (frame) => {
        if (bridge && active && gestureRequested) {
          bridge.submitHandFrame(JSON.stringify(frame));
        }
      },
      reportTrackerStatus,
    );
  }
  return tracker;
}

async function syncTracker() {
  if (!active || !gestureRequested) {
    tracker?.stop();
    return;
  }
  try {
    await ensureTracker().start();
  } catch (_error) {
    gestureRequested = false;
  }
}

function connectBridge(exposedBridge) {
  bridge = exposedBridge;
  bridge.stateChanged.connect((value) => renderer.setState(value));
  bridge.audioLevelChanged.connect((value) => renderer.setAudioLevel(value));
  bridge.reducedMotionChanged.connect((value) => renderer.setReducedMotion(value));
  bridge.animationIntensityChanged.connect((value) =>
    renderer.setAnimationIntensity(value),
  );
  bridge.overlayChanged.connect((title, message) => {
    const safeTitle = safeText(title, 80) || "STANDBY";
    const safeMessage = safeText(message, 240);
    stateNode.textContent = safeTitle;
    messageNode.textContent = safeMessage;
    renderer.setOverlay?.(safeTitle, safeMessage);
  });
  bridge.gestureChanged.connect((rotationX, rotationY, expansion, mode) => {
    renderer.applyGesture(rotationX, rotationY, expansion, mode);
  });
  bridge.activeChanged.connect((value) => {
    active = Boolean(value);
    renderer.setActive(active);
    void syncTracker();
  });
  bridge.gestureEnabledChanged.connect((value) => {
    gestureRequested = Boolean(value);
    void syncTracker();
  });
  bridge.rendererReady();
}

function shutdown() {
  gestureRequested = false;
  active = false;
  tracker?.dispose();
  tracker = null;
  renderer?.dispose();
  renderer = null;
  channel = null;
  bridge = null;
}

try {
  renderer = createAsherOrbRenderer(container);
  window.asherOrbDebug = () => renderer?.snapshot?.() || { disposed: true };
  window.asherOrbShutdown = shutdown;
  window.addEventListener("beforeunload", shutdown, { once: true });
  document.addEventListener("visibilitychange", () => {
    renderer?.setDocumentVisible(!document.hidden);
    if (document.hidden) tracker?.stop();
    else void syncTracker();
  });

  if (window.qt?.webChannelTransport && window.QWebChannel) {
    channel = new QWebChannel(window.qt.webChannelTransport, (connected) => {
      connectBridge(connected.objects.asherBridge);
    });
  }
} catch (error) {
  reportRendererError(error);
}
