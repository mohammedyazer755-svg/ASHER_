import * as THREE from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { VRMLoaderPlugin } from '@pixiv/three-vrm';

// ASHER 3D VRM Companion Scene
class CompanionScene {
  constructor() {
    this.isReady = false;
    this.active = true;
    this.reducedMotion = false;
    this.state = 'STANDBY';
    this.audioLevel = 0.0;
    this.activeCharacter = 'male';
    this.targetCharacter = 'male';

    // Three.js Core
    this.container = null;
    this.scene = null;
    this.camera = null;
    this.renderer = null;
    this.clock = null;
    this.lights = {};
    this.platform = null;

    // VRM Models
    this.vrms = {
      male: null,
      female: null
    };
    this.vrmLoading = {
      male: false,
      female: false
    };

    // Pointer Interaction State
    this.pointer = {
      x: 0,
      y: 0,
      targetX: 0,
      targetY: 0,
      isHovered: false,
      isDragging: false,
      dragStartX: 0,
      dragAngle: 0,
      targetDragAngle: 0,
      clickStartTime: 0,
      clickStartPos: { x: 0, y: 0 }
    };

    // Idle & Blinking System
    this.time = 0.0;
    this.nextBlinkTime = 2.8;
    this.blinkProgress = 0.0;
    this.isBlinking = false;
    this.blinkDuration = 0.16;
    this.isDoubleBlink = false;
    this.doubleBlinkStage = 0;

    // Click Reaction State
    this.reactionTime = 0.0;
    this.isReacting = false;
  }

  init() {
    this.container = document.getElementById('companion-canvas-container');
    if (!this.container) return;

    const width = this.container.clientWidth || window.innerWidth || 420;
    const height = this.container.clientHeight || window.innerHeight || 560;

    // 1. Perspective Camera
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(30, width / height, 0.1, 100);
    this.camera.position.set(0, 0.9, 2.8);

    // 2. 100% Transparent WebGL Canvas
    this.renderer = new THREE.WebGLRenderer({
      alpha: true,
      antialias: true,
      powerPreference: 'high-performance'
    });
    this.renderer.setSize(width, height);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.setClearColor(0x000000, 0); // 100% Transparent, NO black wall
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.container.appendChild(this.renderer.domElement);

    this.clock = new THREE.Clock();

    // 3. Cinematic Studio Lighting
    this.setupLighting();

    // 4. Subtle ASHER Holographic Energy Platform
    this.setupPlatform();

    // 5. Initialize VRM Loader with local dependencies
    this.setupVrmLoader();

    // 6. Interactive Pointer Listeners
    this.setupInteractionListeners();

    // 7. Window Resize Listener
    window.addEventListener('resize', this.onResize.bind(this));

    this.isReady = true;
    if (window.CompanionBridge) {
      window.CompanionBridge.notifyReady();
    }

    this.animate();
  }

  setupLighting() {
    // Ambient fill
    this.lights.ambient = new THREE.AmbientLight(0xdce6f8, 1.4);
    this.scene.add(this.lights.ambient);

    // Front-left key light
    this.lights.key = new THREE.DirectionalLight(0xdff0ff, 2.6);
    this.lights.key.position.set(-1.8, 3.2, 2.5);
    this.scene.add(this.lights.key);

    // Violet rim light (back-right, defining silhouette against dark background)
    this.lights.rim = new THREE.DirectionalLight(0x9d65ff, 2.2);
    this.lights.rim.position.set(1.8, 2.6, -2.0);
    this.scene.add(this.lights.rim);

    // Dynamic ASHER core/platform glow
    this.lights.platform = new THREE.PointLight(0x00d4ff, 1.3, 3.5);
    this.lights.platform.position.set(0, 0.05, 0.2);
    this.scene.add(this.lights.platform);

    this.lights.core = new THREE.PointLight(0x00f0ff, 1.2, 1.8);
    this.lights.core.position.set(0, 1.15, 0.35);
    this.scene.add(this.lights.core);
  }

  setupPlatform() {
    this.platform = new THREE.Group();
    this.platform.position.set(0, -0.02, 0);

    // Thin concentric violet and electric-blue holographic rings (transparent)
    this.platform.rings = [];
    const radii = [0.82, 0.68, 0.52];
    for (let i = 0; i < radii.length; i++) {
      const r = radii[i];
      const geo = new THREE.RingGeometry(r - 0.006, r + 0.006, 64);
      const mat = new THREE.MeshBasicMaterial({
        color: (i === 0) ? 0x7852ff : 0x00e5ff,
        transparent: true,
        opacity: 0.8 - i * 0.2,
        side: THREE.DoubleSide
      });
      const ring = new THREE.Mesh(geo, mat);
      ring.rotation.x = -Math.PI / 2;
      ring.position.y = 0.002 * (i + 1);
      this.platform.add(ring);
      this.platform.rings.push(ring);
    }

    this.scene.add(this.platform);
  }

  setupVrmLoader() {
    this.loader = new GLTFLoader();
    this.loader.register((parser) => new VRMLoaderPlugin(parser));

    // Attempt to load male and female VRMs
    this.loadVrm('male', './assets/male_asher.vrm');
    this.loadVrm('female', './assets/female_asher.vrm');
  }

  loadVrm(gender, url) {
    if (this.vrmLoading[gender]) return;
    this.vrmLoading[gender] = true;

    this.loader.load(
      url,
      (gltf) => {
        const vrm = gltf.userData.vrm;
        if (!vrm) {
          console.warn(`No VRM found in ${url}`);
          return;
        }
        this.vrms[gender] = vrm;
        this.scene.add(vrm.scene);

        // Adjust visibility according to active selection
        vrm.scene.visible = (this.activeCharacter === gender);

        // Dynamic camera auto-framing: 75-85% stage height
        if (this.activeCharacter === gender) {
          this.fitCameraToVrm(vrm);
        }
      },
      undefined,
      (error) => {
        // VRM model not found or invalid - inform bridge so host falls back to clean PNG
        console.info(`VRM asset not available at ${url} (${error.message || error})`);
        if (window.CompanionBridge && typeof window.CompanionBridge.notifyVrmMissing === 'function') {
          window.CompanionBridge.notifyVrmMissing(gender);
        }
      }
    );
  }

  // Automatic full-body camera fitting: avatar occupies 75-85% of stage height
  fitCameraToVrm(vrm) {
    if (!this.camera || !vrm || !vrm.scene) return;

    // 1. Compute model bounding box
    const box = new THREE.Box3().setFromObject(vrm.scene);
    const size = new THREE.Vector3();
    box.getSize(size);
    const center = new THREE.Vector3();
    box.getCenter(center);

    const modelHeight = Math.max(1.4, size.y);

    // 2. Camera distance calculation based on vertical FOV:
    // With a 1.10 margin, character height occupies ~78% of the stage height
    const vFov = (this.camera.fov * Math.PI) / 180;
    const distance = (modelHeight / 2) / Math.tan(vFov / 2) * 1.10;

    const aspect = this.camera.aspect || 1.0;
    const hFov = 2 * Math.atan(Math.tan(vFov / 2) * aspect);
    const hDistance = (size.x / 2) / Math.tan(hFov / 2) * 1.15;
    const finalDistance = Math.max(distance, hDistance);

    this.camera.position.set(0, center.y, finalDistance);
    this.camera.lookAt(0, center.y, 0);

    // Position platform at the base of character's feet
    if (this.platform) {
      this.platform.position.y = box.min.y;
    }
  }

  setupInteractionListeners() {
    const container = this.container;

    container.addEventListener('pointerenter', () => {
      this.pointer.isHovered = true;
    });

    container.addEventListener('pointerleave', () => {
      this.pointer.isHovered = false;
      this.pointer.targetX = 0;
      this.pointer.targetY = 0;
      this.pointer.isDragging = false;
      container.classList.remove('dragging');
    });

    container.addEventListener('pointermove', (e) => {
      const rect = container.getBoundingClientRect();
      const clientX = e.clientX - rect.left;
      const clientY = e.clientY - rect.top;

      // Normalized pointer coordinates (-1 to +1)
      this.pointer.targetX = (clientX / rect.width) * 2 - 1;
      this.pointer.targetY = -(clientY / rect.height) * 2 + 1;

      if (this.pointer.isDragging) {
        const deltaX = e.clientX - this.pointer.dragStartX;
        // Constrained horizontal body rotation: max +-35 degrees (+-0.61 rad)
        const angle = (deltaX / rect.width) * 1.8;
        this.pointer.targetDragAngle = Math.max(-0.61, Math.min(0.61, angle));
      }
    });

    container.addEventListener('pointerdown', (e) => {
      if (e.button !== 0) return; // Left mouse button
      this.pointer.isDragging = true;
      this.pointer.dragStartX = e.clientX;
      this.pointer.clickStartTime = performance.now();
      this.pointer.clickStartPos = { x: e.clientX, y: e.clientY };
      container.classList.add('dragging');
    });

    window.addEventListener('pointerup', (e) => {
      if (!this.pointer.isDragging) return;
      this.pointer.isDragging = false;
      container.classList.remove('dragging');

      // Spring return to neutral rotation
      this.pointer.targetDragAngle = 0;

      // Detect click interaction
      const duration = performance.now() - this.pointer.clickStartTime;
      const dist = Math.hypot(e.clientX - this.pointer.clickStartPos.x, e.clientY - this.pointer.clickStartPos.y);
      if (duration < 320 && dist < 12) {
        this.triggerClickReaction();
      }
    });
  }

  // Click Interaction: Small acknowledgement nod or wave
  triggerClickReaction() {
    if (this.isReacting) return;
    this.isReacting = true;
    this.reactionTime = 0.0;
  }

  // Switch between Male and Female VRM models
  switchCharacter(name) {
    const clean = (name === 'female') ? 'female' : 'male';
    if (clean === this.targetCharacter) return;

    this.targetCharacter = clean;
    this.activeCharacter = clean;

    if (this.vrms.male && this.vrms.male.scene) {
      this.vrms.male.scene.visible = (clean === 'male');
    }
    if (this.vrms.female && this.vrms.female.scene) {
      this.vrms.female.scene.visible = (clean === 'female');
    }

    const currentVrm = this.vrms[clean];
    if (currentVrm) {
      this.fitCameraToVrm(currentVrm);
    }
  }

  onStateChanged(newState) {
    this.state = newState || 'STANDBY';
  }

  onAudioLevel(level) {
    this.audioLevel = Math.max(0.0, Math.min(1.0, level || 0.0));
  }

  setActive(isActive) {
    this.active = Boolean(isActive);
    if (this.active) {
      this.clock.start();
      this.animate();
    }
  }

  setReducedMotion(isReduced) {
    this.reducedMotion = Boolean(isReduced);
  }

  onResize() {
    if (!this.container || !this.renderer || !this.camera) return;

    const width = this.container.clientWidth || window.innerWidth;
    const height = this.container.clientHeight || window.innerHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);

    const currentVrm = this.vrms[this.activeCharacter];
    if (currentVrm) {
      this.fitCameraToVrm(currentVrm);
    }
  }

  // Main Render Loop
  animate() {
    if (!this.active) return;
    requestAnimationFrame(this.animate.bind(this));

    let dt = this.clock.getDelta();
    if (dt > 0.1) dt = 0.1;
    this.time += dt;

    const currentVrm = this.vrms[this.activeCharacter];

    this.updatePointerTracking(dt, currentVrm);
    this.updateBlinking(dt, currentVrm);
    this.updateStateReactions(dt);
    this.updateIdleAndBones(dt, currentVrm);

    if (currentVrm) {
      currentVrm.update(dt);
    }

    if (this.platform && this.platform.rings) {
      for (let i = 0; i < this.platform.rings.length; i++) {
        this.platform.rings[i].rotation.z += dt * (i % 2 === 0 ? 0.08 : -0.06);
      }
    }

    this.renderer.render(this.scene, this.camera);
  }

  updatePointerTracking(dt, vrm) {
    // Smooth pointer lerp
    const lerpFactor = Math.min(1.0, dt * 7.0);
    this.pointer.x += (this.pointer.targetX - this.pointer.x) * lerpFactor;
    this.pointer.y += (this.pointer.targetY - this.pointer.y) * lerpFactor;

    // Smooth drag rotation with spring return
    const dragLerp = Math.min(1.0, dt * 8.0);
    this.pointer.dragAngle += (this.pointer.targetDragAngle - this.pointer.dragAngle) * dragLerp;

    if (vrm && vrm.scene) {
      vrm.scene.rotation.y = this.pointer.dragAngle;
    }
  }

  // Natural Randomized Blinking using VRM Expressions
  updateBlinking(dt, vrm) {
    if (!this.isBlinking) {
      if (this.time >= this.nextBlinkTime) {
        this.isBlinking = true;
        this.blinkProgress = 0.0;
        this.isDoubleBlink = (Math.random() < 0.15);
        this.doubleBlinkStage = 0;
      }
    } else {
      this.blinkProgress += dt / this.blinkDuration;
      let blinkWeight = 0.0;

      if (this.blinkProgress < 0.5) {
        blinkWeight = this.blinkProgress / 0.5;
      } else if (this.blinkProgress < 1.0) {
        blinkWeight = 1.0 - ((this.blinkProgress - 0.5) / 0.5);
      } else {
        if (this.isDoubleBlink && this.doubleBlinkStage === 0) {
          this.doubleBlinkStage = 1;
          this.blinkProgress = 0.0;
        } else {
          this.isBlinking = false;
          blinkWeight = 0.0;
          // Randomized interval: 2.5 to 6.0 seconds
          this.nextBlinkTime = this.time + 2.5 + Math.random() * 3.5;
        }
      }

      if (vrm && vrm.expressionManager) {
        vrm.expressionManager.setValue('blink', blinkWeight);
      }
    }
  }

  // Real ASHER State Presentation
  updateStateReactions(dt) {
    const core = this.lights.core;
    const platform = this.lights.platform;
    const baseIntensity = this.pointer.isHovered ? 1.4 : 1.0;

    switch (this.state) {
      case 'LISTENING':
        core.color.setHex(0x00f0ff);
        core.intensity = baseIntensity * 1.5 + Math.sin(this.time * 5.0) * 0.2;
        platform.intensity = 1.6;
        break;
      case 'TRANSCRIBING':
        core.color.setHex(0xa855f7);
        core.intensity = baseIntensity * 1.3 + Math.sin(this.time * 7.0) * 0.25;
        platform.intensity = 1.3;
        break;
      case 'THINKING':
      case 'EXECUTING':
        core.color.setHex(0x7852ff);
        core.intensity = baseIntensity * 1.2 + Math.sin(this.time * 3.5) * 0.3;
        platform.intensity = 1.2;
        break;
      case 'SPEAKING':
        core.color.setHex(0x00f0ff);
        core.intensity = baseIntensity * (1.2 + this.audioLevel * 1.2);
        platform.intensity = 1.4 + this.audioLevel * 0.6;
        break;
      case 'STOPPED':
      case 'ERROR':
        core.color.setHex(0xff453a);
        core.intensity = 0.8;
        platform.intensity = 0.8;
        break;
      default: // STANDBY
        core.color.setHex(0x00e5ff);
        core.intensity = baseIntensity * 1.0 + Math.sin(this.time * 1.8) * 0.15;
        platform.intensity = 1.1;
        break;
    }
  }

  // Subtle breathing, head tracking, speech nods, and click reactions
  updateIdleAndBones(dt, vrm) {
    if (!vrm || !vrm.humanoid) return;

    const head = vrm.humanoid.getNormalizedBoneNode('head');
    const neck = vrm.humanoid.getNormalizedBoneNode('neck');
    const spine = vrm.humanoid.getNormalizedBoneNode('spine');
    const chest = vrm.humanoid.getNormalizedBoneNode('chest');
    const rightArm = vrm.humanoid.getNormalizedBoneNode('rightUpperArm');

    // 1. Natural Sinusoidal Breathing
    if (spine && !this.reducedMotion) {
      const breath = Math.sin(this.time * 1.75);
      spine.rotation.x = breath * 0.012;
      if (chest) chest.rotation.x = breath * 0.015;
    }

    // 2. Smooth Head and Eye Tracking toward Pointer
    // Clamped yaw +-15 deg (+-0.26 rad), pitch +-8 deg (+-0.14 rad)
    const headYaw = this.pointer.x * 0.26;
    const headPitch = this.pointer.y * 0.14;

    let stateLean = 0.0;
    let stateTilt = 0.0;
    if (this.state === 'LISTENING') {
      stateLean = 0.06; // Attentive forward posture
    } else if (this.state === 'THINKING') {
      stateTilt = 0.05; // Thoughtful head tilt
    } else if (this.state === 'SPEAKING') {
      // Natural conversational speech nods
      stateLean = Math.sin(this.time * 6.0) * (0.02 + this.audioLevel * 0.025);
    }

    if (neck) {
      neck.rotation.y = headYaw * 0.4;
      neck.rotation.x = -headPitch * 0.4 + stateLean * 0.5;
    }
    if (head) {
      head.rotation.y = headYaw * 0.6;
      head.rotation.x = -headPitch * 0.6 + stateLean;
      head.rotation.z = stateTilt;
    }

    // 3. Click Reaction (Nod / Acknowledgement)
    if (this.isReacting) {
      this.reactionTime += dt;
      const rT = this.reactionTime;
      if (rT < 1.0) {
        const nodPhase = Math.sin(rT * Math.PI * 2);
        if (head) head.rotation.x += nodPhase * 0.12;
      } else {
        this.isReacting = false;
      }
    }
  }
}

// Instantiate and attach to window
window.CompanionScene = new CompanionScene();

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    window.CompanionScene.init();
  });
} else {
  window.CompanionScene.init();
}
