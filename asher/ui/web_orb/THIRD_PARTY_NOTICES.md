# ASHER Web Orb — third-party notices

The ASHER Companion renderer is an ASHER-specific implementation. It adapts
layering and gesture concepts after inspecting the MIT-licensed
`ultron-by-sagar-builds` reference by Sagar Tamang; it does not include ULTRON
branding, its Next.js shell, dashboard, screenshot, or decorative telemetry.

## Reference renderer

Copyright (c) 2026 Sagar Tamang. Licensed under the MIT License. The complete
license is retained in `licenses/SAGAR_ORB_REFERENCE_LICENSE.txt`.

## Three.js 0.185.1

Copyright © 2010-2026 three.js authors. Licensed under the MIT License. The
vendored ESM runtime and selected postprocessing addons are covered by the
complete license in `licenses/THREE_LICENSE.txt`.

## MediaPipe Tasks Vision 0.10.35 and Hand Landmarker

Copyright The MediaPipe Authors. Licensed under the Apache License 2.0. ASHER
vendors the Tasks Vision ESM runtime, WASM runtime files, and the official
float16 Hand Landmarker task bundle for fully local inference. The complete
license is retained in `licenses/MEDIAPIPE_LICENSE.txt`.

Camera frames are processed on-device. ASHER does not record them, display a
production webcam preview, or transmit frames/landmarks over a network.
