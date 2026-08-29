import { createOrbScene } from "./ultron_exact/orbScene.js";

export function createAsherOrbRenderer(container) {
  const scene = createOrbScene(container);

  let state = "standby";
  let audioLevel = 0;
  let reducedMotion = false;
  let animationIntensity = 1;
  let active = true;
  let documentVisible = true;
  let expansion = 0;
  let lastExpansion = 0;
  let gestureMode = "idle";
  let overlayTitle = "";
  let overlayMessage = "";

  // Presentation-only speaking breath. ASHER intentionally suppresses the
  // microphone during TTS, so use the truthful canonical SPEAKING state
  // instead of fabricating microphone RMS.
  let speakingRaf = 0;

  // Normal mode sits slightly farther from the exact ULTRON camera.
  // 1.16 means ~14% smaller on screen without changing orb geometry.
  const NORMAL_DISTANCE_FACTOR = 1.16;

  // During SPEAKING the camera moves closer, but never farther than normal.
  // At the peak it approaches the original ULTRON framing.
  const SPEAKING_MAX_EXPANSION = 0.13;

  let speakingZoom = NORMAL_DISTANCE_FACTOR;
  let speakingPhase = 0.0;
  let lastFrameTime = performance.now();

  // Apply the smaller normal framing once.
  scene.zoomBy(NORMAL_DISTANCE_FACTOR);

  function setState(value) {
    state = String(value ?? "standby").trim().toLowerCase() || "standby";
  }

  function animateSpeakingBreath(now) {
    const dt = Math.min(0.05, Math.max(0, (now - lastFrameTime) / 1000));
    lastFrameTime = now;

    const isSpeaking = active && documentVisible && state === "speaking";

    if (isSpeaking) {
      speakingPhase += dt * 4.6;
    }

    // 0..1 pulse: the orb only grows while speaking.
    // It never shrinks below the normal-mode size.
    const wave =
      0.72 * Math.sin(speakingPhase) +
      0.28 * Math.sin(speakingPhase * 1.67 + 0.65);
    const pulse = Math.max(0, Math.min(1, 0.5 + 0.5 * wave));

    const expansion = reducedMotion
      ? SPEAKING_MAX_EXPANSION * 0.35
      : SPEAKING_MAX_EXPANSION;

    const targetZoom = isSpeaking
      ? NORMAL_DISTANCE_FACTOR - expansion * pulse * animationIntensity
      : NORMAL_DISTANCE_FACTOR;

    const response = isSpeaking ? 0.18 : 0.10;
    const nextZoom = speakingZoom + (targetZoom - speakingZoom) * response;

    // zoomBy() multiplies camera distance. Apply only the relative delta so
    // there is no drift and the exact ULTRON camera returns to baseline.
    const relativeFactor = nextZoom / speakingZoom;
    if (Math.abs(relativeFactor - 1.0) > 0.00005) {
      scene.zoomBy(relativeFactor);
    }
    speakingZoom = nextZoom;

    speakingRaf = requestAnimationFrame(animateSpeakingBreath);
  }

  speakingRaf = requestAnimationFrame(animateSpeakingBreath);

  function setAudioLevel(value) {
    const number = Number(value);
    audioLevel = Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
  }

  function setReducedMotion(value) {
    reducedMotion = Boolean(value);
  }

  function setAnimationIntensity(value) {
    const number = Number(value);
    animationIntensity = Number.isFinite(number)
      ? Math.max(0, Math.min(1, number))
      : 1;
  }

  function setOverlay(title, message = "") {
    overlayTitle = String(title ?? "").slice(0, 80);
    overlayMessage = String(message ?? "").slice(0, 240);
  }

  function setActive(value) {
    active = Boolean(value);
    container.style.visibility = active ? "visible" : "hidden";
  }

  function setDocumentVisible(value) {
    documentVisible = Boolean(value);
  }

  function applyGesture(rotationX, rotationY, expansionValue, modeValue = "idle") {
    const rx = Math.max(-0.22, Math.min(0.22, Number(rotationX) || 0));
    const ry = Math.max(-0.22, Math.min(0.22, Number(rotationY) || 0));
    const nextExpansion = Math.max(0, Math.min(1, Number(expansionValue) || 0));
    gestureMode = String(modeValue ?? "idle").toLowerCase();

    if (gestureMode === "spin" || gestureMode === "rotate" || gestureMode === "grab") {
      scene.rotateBy(ry, rx);
    }

    if (gestureMode === "unfold" || gestureMode === "zoom") {
      const delta = nextExpansion - lastExpansion;
      if (Math.abs(delta) > 0.0005) {
        scene.zoomBy(Math.exp(-delta * 1.35));
      }
    }

    expansion = nextExpansion;
    lastExpansion = nextExpansion;
  }

  function resize() {
    // Exact renderer owns its own window resize handler.
  }

  function snapshot() {
    return Object.freeze({
      renderer: "exact-supplied-ultron-orb",
      exactReferenceRenderer: true,
      recolored: false,
      state,
      audioLevel,
      reducedMotion,
      animationIntensity,
      active,
      documentVisible,
      gestureExpansion: expansion,
      gestureMode,
      overlay: Object.freeze({ title: overlayTitle, message: overlayMessage }),
    });
  }

  function dispose() {
    if (speakingRaf) {
      cancelAnimationFrame(speakingRaf);
      speakingRaf = 0;
    }

    if (Math.abs(speakingZoom - 1.0) > 0.00005) {
      scene.zoomBy(1.0 / speakingZoom);
      speakingZoom = 1.0;
    }

    scene.dispose();
  }

  return {
    setState,
    setAudioLevel,
    setReducedMotion,
    setAnimationIntensity,
    setOverlay,
    setActive,
    setDocumentVisible,
    applyGesture,
    resize,
    snapshot,
    dispose,
    rotateBy: scene.rotateBy,
    zoomBy: scene.zoomBy,
    zoomIn: scene.zoomIn,
    zoomOut: scene.zoomOut,
    resetView: scene.resetView,
  };
}
