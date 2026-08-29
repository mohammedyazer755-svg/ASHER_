import * as THREE from "three";

export class OrbitControls {
  constructor(camera, domElement) {
    this.object = camera;
    this.domElement = domElement;
    this.target = new THREE.Vector3();
    this.enableDamping = true;
    this.dampingFactor = 0.04;
    this.minDistance = 0.6;
    this.maxDistance = 40;
    this.zoomSpeed = 1.4;
    this.enablePan = false;

    this._dragging = false;
    this._x = 0;
    this._y = 0;

    this._down = (e) => {
      this._dragging = true;
      this._x = e.clientX;
      this._y = e.clientY;
      this.domElement.setPointerCapture?.(e.pointerId);
    };
    this._move = (e) => {
      if (!this._dragging) return;
      const dx = e.clientX - this._x;
      const dy = e.clientY - this._y;
      this._x = e.clientX;
      this._y = e.clientY;

      const offset = this.object.position.clone().sub(this.target);
      const spherical = new THREE.Spherical().setFromVector3(offset);
      spherical.theta -= dx * 0.005;
      spherical.phi = THREE.MathUtils.clamp(
        spherical.phi + dy * 0.005,
        0.05,
        Math.PI - 0.05,
      );
      spherical.makeSafe();
      offset.setFromSpherical(spherical);
      this.object.position.copy(this.target).add(offset);
      this.object.lookAt(this.target);
    };
    this._up = () => { this._dragging = false; };
    this._wheel = (e) => {
      e.preventDefault();
      const offset = this.object.position.clone().sub(this.target);
      const factor = Math.exp(
        THREE.MathUtils.clamp(e.deltaY * 0.001 * this.zoomSpeed, -0.35, 0.35),
      );
      const distance = THREE.MathUtils.clamp(
        offset.length() * factor,
        this.minDistance,
        this.maxDistance,
      );
      offset.setLength(distance);
      this.object.position.copy(this.target).add(offset);
      this.object.lookAt(this.target);
    };

    this.domElement.style.touchAction = "none";
    this.domElement.addEventListener("pointerdown", this._down);
    window.addEventListener("pointermove", this._move);
    window.addEventListener("pointerup", this._up);
    this.domElement.addEventListener("wheel", this._wheel, { passive: false });
  }

  update() {
    this.object.lookAt(this.target);
  }

  dispose() {
    this.domElement.removeEventListener("pointerdown", this._down);
    window.removeEventListener("pointermove", this._move);
    window.removeEventListener("pointerup", this._up);
    this.domElement.removeEventListener("wheel", this._wheel);
  }
}
