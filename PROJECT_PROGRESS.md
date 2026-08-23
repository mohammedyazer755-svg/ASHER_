## Continuation update — PreferenceCore Phase PC-1A: explicit local preference capture (2026-08-23)

### Goal locked
- ASHER is being built as a local-first personal agent, not merely a chatbot with automation. The long-term target is an assistant that can identify the owner, remember useful context, execute authorized digital work, recover from failures, and increasingly behave the way the owner prefers.
- VoiceGuard learns **who is speaking**. PreferenceCore will learn **how the owner prefers ASHER to behave**: brevity, directness, clarification tendency, proactive suggestions, failure-report detail and preferred response wording.
- PreferenceCore must never learn or override permissions, confirmations, strong authentication, emergency-stop behavior, or other deterministic security policy.

### Implemented in PC-1A
- Added `asher/preferences/` with typed feedback records and an owner-only SQLite store.
- Preference capture is opt-in and disabled by default. Normal conversation is never silently turned into training data.
- ASHER keeps only the immediately preceding eligible response in memory as a temporary feedback candidate; it is persisted only when the owner explicitly labels it.
- Local text/voice control commands: `preference learning on/off`, `preference status`, `feedback accept/reject`, `feedback shorter/more detailed/more direct`, `feedback ask less/ask more`, `feedback suggest more/suggest less`, and `feedback preferred: <exact preferred reply>`.
- Safety-policy and confirmation-preview responses are excluded from trainable candidates. Credential-like content is rejected rather than stored.
- Stored context is bounded metadata only: provider/offline flag, allowed tool names and response lengths; tool arguments and secret values are not copied into PreferenceCore context.
- Feedback can be listed/deleted only through a live owner session. Audit records capture feedback type/dimensions only and omit conversation text.
- SQLite schema advances to version 2 with a soft-deletable `preference_events` table.

### Verification in handoff environment
- New PreferenceCore tests: **6/6 passed**.
- Full suite discovers **209 tests** after PC-1A (203 prior + 6 new). In this handoff environment, Qt tests are skipped and the same three previously-known VoiceGuard lazy-import checks fail because optional numerical modules are preloaded; Windows `run_tests.ps1` remains authoritative.
- `python -m compileall -q asher tests`: PASS.

### Exact next action
Apply PC-1A to the owner's current clean checkpoint, run the full Windows suite, and require **209 tests passed**. Then use ASHER normally with PreferenceCore enabled and collect genuine explicit feedback before training any personalization model. PC-1B will add dataset inspection/quality summaries; model training starts only after enough real owner-labelled examples exist.

---

## Continuation note — authoritative exact UI specification received (2026-08-22)

- The owner supplied `ASHER_EXACT_UI_SPEC.md` as the authoritative visual and interaction reference for future UI work. Its contents are UI requirements and constraints, not authorization to rewrite ASHER's security, VoiceGuard, memory, planner, tools, providers, voice pipeline, or Android protocol.
- The specification confirms the existing product split: Workspace remains the practical management surface; Companion is a continuous near-black active-conversation scene dominated by one large state-driven plasma sphere, minimal real state/transcript text, a small emergency stop, and no decorative HUD or fabricated telemetry.
- Companion confirmations must remain minimal while displaying the exact consequential preview and preserving the real approval/strong-auth policy. The exact-preview regression is being closed as a functional safety defect, not a visual redesign.
- Audio reactivity now uses a bounded scalar derived from real microphone PCM16 frames; raw PCM is not emitted or retained for presentation. TTS amplitude is not available from the current providers and is therefore not fabricated. Webcam gesture control must never be implied before a real local tracker exists.
- Four owner-supplied orb screenshots (`230256`, `230310`, `230316`, and `230331`) are recorded as sphere-aesthetic references only: a layered luminous corona, compact hot core, a few braided arcs/wisps, sparse sparks, and blue/amber variants. Their captions, social chrome, monitor HUD, and gesture implications are explicitly excluded; canonical state colors, real RMS reactivity, and the minimal continuous scene remain authoritative.
- After this owner-requested reference pass, UI scope remains frozen pending functional/spec acceptance. A live voice/visual pass on the owner's display is still required for the parts that automated offscreen tests cannot establish.

---

## Continuation update — VoiceGuard VG-1B and acceptance hardening (2026-08-22)

### VoiceGuard software path completed

- Training now loads only immutably sealed sessions registered to active enrollments; partial/failed collection manifests cannot enter training or be edited after finalization. Finalization rejects foreign-manager sessions, duplicate registration, and active identity role drift. Consented desktop partials are surfaced/resumed after restart only after fresh consent, and ambiguous partials fail closed without registration.
- Added dependency-free, aggregate-only readiness reporting. It requires unique real source clips, at least three conservatively metadata-distinct sessions per class, a real non-replay unauthorized speaker identity, complete class coverage in train/validation/test, authorized and unauthorized calibration/test coverage, checksum integrity, and no identical audio crossing session boundaries. Same-environment sessions require a 30-minute timestamp interval; correlated extras are deterministically excluded from both readiness partitions and actual training, while environment/time metadata remains a quality guard rather than proof of physical independence.
- Augmented clips require explicit same-session source linkage and matching classifier/authorization labels, and may enrich only the training partition. They cannot satisfy source-data readiness, enter calibration/test partitions, or create a fabricated noisy-condition metric. Real replay samples are always unauthorized, evaluation-only, and excluded from classifier fitting.
- The student classifier resolves one global identity before authorizing that exact identity; it never authenticates by summing several users' probabilities. Held-out reports expose authorized-identity errors separately from binary FAR/FRR. Speaker and wake-word artifacts use separate registry keys, so experimental wake training cannot replace the active speaker model.
- Model, validation report, and test report are content-versioned over every inference/security-relevant model field plus the stable independent-dataset provenance and written under ASHER's private runtime directory before registry activation. Existing versioned paths are never overwritten; partial persistence/activation failures remove the new bundle without changing prior files or activation.
- `train_voiceguard.py --check` returns privacy-safe structural status without importing NumPy/scikit-learn. Normal training now reports measured held-out accuracy/F1/FAR/FRR and explicitly leaves absent noisy/replay metrics unavailable instead of raising a traceback or guessing.
- Desktop VoiceGuard capture groups six clips into one real session and pauses immediate same-environment repeats. Partial sessions remain unregistered; shared thread/process locks and disk refresh prevent concurrent or stale adapters from losing clips, clearing revocation, or re-enrolling with stale consent. Every partial/finalized revocation routes through the lifecycle manager, which invalidates peer consent before manifest cleanup. Retraining snapshots under the lifecycle lock, fits outside it so revocation stays live, then compare-and-swaps the unchanged registry and full finalized-dataset fingerprint before artifact activation. Runtime verification validates the exact in-memory inference model and rechecks the bound on-disk model plus verified finalized-dataset fingerprint before and after inference.

### Locally testable acceptance gaps closed

- Companion-mode confirmations now display the complete exact preview as read-only plain text while preserving the minimal fullscreen scene, approval binding, strong authentication, and emergency stop.
- The active desktop voice runtime honors `ASHER_MIC_INDEX` as either a non-negative index or exact device name; the setting remains lazy and opens no hardware during import/tests.
- Both Workspace and Companion orbs receive the real microphone's normalized RMS scalar through the existing Qt-thread refresh path. The signal decays/resets safely, exposes no raw PCM, and does not pretend that synthetic TTS has an amplitude feed.
- Workspace Memory now supports owner-only JSON export after fresh device authentication. Audit data records format/count only, never destination names or memory values.
- Expired desktop owner sessions show **SESSION EXPIRED** and require an explicit Workspace-only device-auth reauthentication flow. Pending confirmations from the expired session are not rebound.

### Verification and current real-data status

- `./run_tests.ps1`: **180 tests passed, 0 failed**, including deterministic offscreen PySide6 orb rendering and all new readiness/security/UI/audio-signal/lifecycle regressions.
- Legacy deterministic voice smoke: **16/16 passed**.
- `verify_upgrade.py`: PASS. `pip check`: no broken requirements.
- A metadata-only microphone query reports available Windows input devices. No new audio was recorded or retained during verification.
- The private VoiceGuard inventory contains **1 finalized owner session / 6 clean clips / 1 class**. `train_voiceguard.py --check` exits `2` (not ready), correctly reporting no unauthorized class, fewer than three sessions for the owner class, and no possible session-separated split. No model or biometric accuracy claim exists yet.
- The pre-existing uncommitted trailing-space/no-final-newline edit in `run_tests.ps1` was preserved as owner work; it is the only `git diff --check` finding and does not affect execution.

### Exact next action

Record at least **2 more owner sessions** and **3 consented `unknown_pool` sessions**, one invocation per genuinely separate time/environment, then run `train_voiceguard.py --check`. When it reports `READY`, run `train_voiceguard.py` to create the first honest held-out baseline. Separately, perform the live Companion visual/voice acceptance pass: automated tests verify the microphone-amplitude plumbing but cannot establish its response with the owner's physical microphone or visual fidelity on the owner's display, and the current TTS providers expose no real amplitude feed. Android build/device verification remains external because Java, Gradle wrapper/SDK, ADB, and a physical device are absent.

---

## Continuation update — Cinematic UI final freeze: minimal companion scene (2026-08-21)

### Final visual decision
- Workspace mode remains the normal ASHER management surface for conversation history, memory, confirmations, VoiceGuard/users, permissions, activity, settings and diagnostics.
- Active voice interaction switches to a deliberately minimal full-screen Companion scene. The previous sci-fi HUD cards and decorative telemetry were removed because they made the interface look synthetic and duplicated information already available in Workspace.
- Companion mode now keeps only: the ASHER identity, a truthful local/offline presence indicator, the persistent emergency Stop control, the state-driven energy sphere, real state/transcript text and the exact confirmation strip when a consequential action requires approval.
- The cinematic sphere remains original procedural PySide6 rendering connected to canonical ASHER state. It does not advance state, fabricate success or display invented metrics.
- Sphere scaling remains bounded and testable by desktop wheel input. Webcam hand tracking is intentionally **not** claimed as part of the visual UI; it remains a later interaction capability.

### Scope frozen
- Do not spend further implementation cycles on cosmetic orb/HUD tuning before the core ASHER roadmap is completed. Visual polish may be revisited only during final acceptance.
- Detailed provider, planner, executor, VoiceGuard and diagnostic data stay in Workspace instead of being copied into Companion mode.

### Verification in handoff environment
- `python -m compileall -q asher tests`: PASS.
- 95 tests are discovered after the final minimal-UI contract test. Qt tests require PySide6 and are skipped in this handoff environment.
- The environment still shows the already-known VoiceGuard import-safety false failure because optional numerical modules are preloaded here; the owner's Windows `run_tests.ps1` result is authoritative.

### Exact next action after Windows verification
Run the full Windows test suite and one live Companion smoke test. If green, commit the complete UI work as a single checkpoint and move immediately to **VoiceGuard real-data Phase VG-1: owner recording sessions, dataset manifest, session-separated train/validation/test split and first measured baseline**.

---

## Continuation update — Cinematic UI Phase UI-3B: immersive plasma HUD renderer (2026-08-21)

### Visual target clarified by owner
- The compact dashboard orb is **not** the target for active conversation.
- Workspace/dashboard mode remains for history, memory, confirmations, VoiceGuard, settings and diagnostics.
- Active voice conversation uses a separate immersive, near-black HUD scene inspired by the supplied reference video, implemented with original procedural rendering rather than copied assets.

### Implemented in UI-3B
- Added a dedicated cinematic renderer mode to `AsherOrbWidget` while preserving the compact workspace renderer.
- Cinematic sphere now uses a bright plasma shell, animated internal energy filaments, crossing energy orbits, star/particle field, central scan line, bloom and state-driven colours.
- State and message are painted inside the sphere instead of below it in Companion mode.
- Added truthful side HUD panels for microphone/session/state, safety boundary, provider/voice/privacy and live transcript/planner/executor activity. No fabricated CPU/network/accuracy numbers are shown.
- Companion mode enters full-screen only during the active voice scene and returns to the previous workspace window mode afterwards.
- Existing emergency stop and confirmation controls remain available in the immersive scene.
- Existing bounded mouse-wheel sphere scaling remains. Actual webcam/hand tracking is still **not implemented** and remains the next interaction milestone.

### Verification
- `python -m compileall -q asher tests`: PASS in the handoff environment.
- 94 tests are discovered after UI-3B additions. New Qt visual/fullscreen tests require PySide6 on the Windows project environment.
- The handoff environment still exhibits its known VoiceGuard import-safety false failure because optional numerical modules are preloaded by that environment; the owner Windows baseline is authoritative for this test.

### Exact next action
Apply UI-3B on the Windows `feature/cinematic-ui` branch, run `run_tests.ps1`, then launch `run_asher.ps1 -Mode ui -Live` and visually compare the active Companion scene with the supplied reference. Tune plasma/HUD geometry only after the functional/fullscreen tests are green.

---

# ASHER Project Progress


## Continuation update — Cinematic UI Phase UI-3A: dual-mode companion foundation (2026-08-21)

### Verified before UI-3A

- UI-1 is committed on `feature/cinematic-ui` as `e28a530`.
- UI-2 state-driven orb prototype was verified on the owner's Windows machine with **88 tests passed, 0 failed** and the legacy offline voice-accuracy smoke passing.
- The UI-2 visual result was intentionally rejected as the final Home design because the orb was embedded inside the permanent dashboard rather than becoming a dedicated immersive conversation scene.

### Implemented in UI-3A

- Split the desktop experience into two presentation modes without duplicating the assistant process:
  - **Workspace mode** preserves the current sidebar, history/conversation, confirmation, memory, VoiceGuard/users, permissions, activity, settings and diagnostics views.
  - **Companion mode** is a separate edge-to-edge dark scene used only during an active voice interaction.
- Added `should_use_companion_mode(state, microphone_active)` so text-only planning or idle wake monitoring does not force the immersive scene. The mode follows real controller state plus microphone activity rather than a fake timer.
- Added an immersive Companion view with a large state-driven orb, minimal telemetry, persistent emergency stop and an in-scene confirmation strip for pending consequential actions.
- Kept the existing Home/dashboard intact as the normal management surface.
- Added bounded orb display scaling (`340..900 px`) and desktop wheel resizing. The scaling hook is presentation-only and can later be driven by a webcam hand-gesture adapter; **actual hand tracking is not claimed implemented in UI-3A**.
- The same canonical state events update both Workspace Home and Companion mode. No second controller, planner, memory store or safety path was created.
- Added regression coverage for voice-only mode gating, workspace/companion separation and the bounded external orb-scale hook.

### Honest limitations after UI-3A

- The immersive orb still uses the UI-2 procedural ring/glow renderer. The reel-inspired plasma/electric filament renderer is the next visual milestone, not yet implemented.
- Real webcam hand tracking/pinch resizing is not yet implemented. UI-3A only establishes the safe bounded scale interface and mouse-wheel test path.
- Live microphone/TTS amplitude is not yet connected to the orb.
- Transition animation between Workspace and Companion modes is functional but not yet visually polished.
- Performance/latency measurements for the immersive mode are still pending.

### Required verification before committing UI-3A

Run `./run_tests.ps1` on the owner's Windows machine. Then launch `./run_asher.ps1 -Mode ui` and verify the normal Workspace still opens unchanged. Live voice mode should be tested only after the deterministic suite is green.

### Exact next action

After UI-3A is green and committed, implement UI-3B: the reel-inspired original plasma sphere/HUD renderer inside Companion mode, then wire live amplitude. Webcam hand-gesture resizing remains a later bounded input adapter after the visual/desktop interaction is stable.

---


## Continuation update — Cinematic UI Phase UI-2: procedural state-driven orb (2026-08-21)

### Verified before UI-2

- UI-1 committed on `feature/cinematic-ui` as `e28a530` (`UI-1 truthful cinematic state bridge`).
- Owner's Windows machine ran the complete suite after UI-1: **85 tests passed, 0 failed**.
- Deterministic legacy voice-accuracy smoke test also passed.

### Implemented in UI-2

- Added `asher/ui/orb_widget.py`: an original procedural PySide6/QPainter companion orb with bounded glow, arcs, particles and a radial core. No downloaded/copyrighted orb assets are used.
- Orb visuals map directly to the canonical `AssistantState`; the animation timer changes visual phase only and cannot advance ASHER state or fake success.
- Added distinct palettes/motion for standby, wake, authentication, listening, transcription, thinking, confirmation, execution, observation, speaking, success, error, offline, stopped and locked states, while keeping legacy fallback states renderable.
- Added a short-lived normalized `set_audio_level()` presentation hook without persisting raw audio. Real microphone/TTS level wiring remains a later milestone and is not claimed implemented.
- Animation is bounded, slows during low-activity states, and stops when hidden. Reduced-motion/intensity hooks are present for later settings integration.
- Reworked the Home view around the real orb while preserving text input, microphone toggle, provider status, voice profile status, sidebar navigation, emergency stop and all existing companion panels.
- Connected the existing canonical state subscription to Qt through a queued Signal bridge so state changes can update the orb immediately without touching Qt widgets from worker/voice threads. Polling remains as a safe status fallback.
- Added `tests/test_orb_widget.py` covering complete state-to-visual mapping, stopped/locked particle suppression, and optional offscreen Qt rendering across real states.

### Honest limitations after UI-2

- Audio-reactive animation is **PARTIAL**: the widget accepts a normalized level, but the current voice runtime does not yet publish live level events to the UI.
- Authentication/user-role identity is not yet surfaced in the minimalist Home status row. The existing security controller remains authoritative and unchanged.
- Conversation/memory/security/activity/settings panels retain the existing functional dashboard styling; cinematic panel redesign is a later companion-panel milestone.
- Reduced-motion and animation-intensity setters exist in the orb, but Settings is not yet connected to them.
- Performance/latency measurements for the cinematic UI have not yet been recorded.

### Required verification before committing UI-2

Run `./run_tests.ps1` on the owner's Windows machine. UI-2 is not complete until the complete deterministic suite and legacy voice smoke tests remain green and the offscreen Qt orb test passes when PySide6 is available. Then launch the desktop UI and visually confirm the orb renders without freezing the interface.

### Exact next action

After UI-2 is green and committed, wire safe live amplitude/state-detail signals for LISTENING/SPEAKING, surface authenticated role/provider/voice status truthfully, and add the confirmation surface around the orb without weakening the existing policy or emergency-stop path.

---

## Continuation update — Cinematic UI Phase UI-1: truthful state/event bridge (2026-08-21)

### Owner baseline before this change

- Local Git baseline commit: `28d581d` — `Baseline before cinematic UI - 82 tests passing`.
- Development branch: `feature/cinematic-ui`.
- Owner Windows verification before this phase: 82/82 unit tests passed and the legacy offline voice-accuracy smoke passed 16/16.

### Implemented in UI-1

- Expanded the canonical assistant state vocabulary for the cinematic companion flow:
  `STANDBY`, `WAKE_DETECTED`, `AUTHENTICATING`, `AUTHENTICATED`, `LISTENING`,
  `TRANSCRIBING`, `THINKING`, `AWAITING_CONFIRMATION`, `EXECUTING`, `OBSERVING`,
  `SPEAKING`, `SUCCESS`, `ERROR`, `OFFLINE`, `STOPPED`, and `LOCKED`.
- Preserved the old `UNDERSTANDING`, `ACTING`, and `COMPLETE` values for legacy/fallback integrations.
- `StateEvent` now records both previous and new state, giving the future orb an observable transition stream instead of relying on timed/fake animation.
- The real `CompanionController` now publishes `THINKING`, `LOCKED`, and conversational `SUCCESS` states around planning/session decisions.
- The real agent loop now distinguishes `EXECUTING` from `OBSERVING`, resumes `EXECUTING` after confirmation/retry, and publishes verified `SUCCESS` only after the tool loop completes.
- The real voice runtime now publishes standby/listen/transcribe/wake/auth/locked/speaking transitions through the same canonical state store used by the agent.
- Desktop microphone activity is now a separate `DesktopStatus.microphone_active` signal. Keeping the microphone runtime alive no longer masks deeper states such as thinking, executing, observing, confirmation, or speaking.
- The desktop adapter exposes the existing canonical state stream through `subscribe_state()` for the upcoming PySide6 orb bridge.
- Text-fallback TTS now publishes a truthful `SPEAKING` state and returns to `SUCCESS` or `AWAITING_CONFIRMATION` after speech completes.
- The existing dashboard listen button now follows microphone activity rather than assuming that semantic state `LISTENING` means the microphone worker is running.

### Tests added/updated

- Added `tests/test_cinematic_state_bridge.py` covering:
  - previous/new state publication;
  - availability of the complete cinematic state vocabulary;
  - real controller ordering through think -> execute -> observe -> success.
- Updated the UI adapter background-voice-runtime regression test to prove that microphone activity does not hide `THINKING`.

### Verification while preparing this phase

- `python -m compileall -q .`: PASS.
- Focused state/UI/agent/voice suite: 29 tests run, 28 passed, 1 optional PySide6 smoke skipped in the sandbox.
- `python -B test_voice_accuracy.py`: 16/16 PASS.
- Full sandbox discovery finds 85 tests. One pre-existing VoiceGuard import-laziness assertion cannot be evaluated in this ChatGPT execution host because the host preloads NumPy before Python user code starts. This phase does not modify VoiceGuard. The owner's pre-change Windows baseline passed that test, so the authoritative post-change result must come from `D:\VOICE AI` using `.\run_tests.ps1` after applying UI-1.

### Exact next action

Apply the UI-1 patch on `feature/cinematic-ui`, run `.\run_tests.ps1` on the owner's Windows machine, and do not continue to orb rendering until the complete suite is green. If green, commit UI-1 before starting the procedural orb widget.

---

Last updated: 2026-08-20 (Asia/Calcutta)

## Continuation update — live voice and communication fail-closed pass (2026-08-20)

### Repository baseline rechecked

- The project `.venv` is Python 3.12.3, `pip check` reports no broken
  requirements, and the optional PySide6/Faster-Whisper/OpenAI packages are
  now installed locally. No API key value or private runtime data was read.
- A clean safe aggregate run completed with 74 non-Qt unit tests passing;
  the legacy offline voice-normalization smoke remains 16/16 passing.
- `.git` is still a pre-existing empty directory, so Git correctly reports
  that this is not a repository. It has not been deleted, replaced, or
  initialized yet.

### Completed in this continuation

- Repaired the real voice state machine so a standalone `Hey Asher` creates
  an owner/trusted/guest session and the following utterance is accepted
  during a bounded active-listening window without repeating the wake phrase.
- Routed `Yes?`, replies, and clarifications through the existing
  provider-independent interruptible TTS manager. Detected speech stops active
  playback before turn capture, and ordinary voice-runtime shutdown no longer
  latches the global emergency stop.
- Removed the unsafe `create_voice_session()` owner fallback: callers must
  supply an explicitly identified non-guest actor; unidentified speakers use
  a guest session.
- Added regression coverage for standalone wake -> next command -> spoken
  reply and explicit-owner voice-session creation.
- Closed a live WhatsApp fail-open boundary. The built-in Windows adapter now
  refuses before paste/Enter unless an adapter can observe the exact recipient
  first; post-send failure can no longer be the first indication of a wrong
  chat. Added a regression test proving the send callback is never reached
  when recipient observation fails.

### Verification in this continuation

- `.venv\\Scripts\\python.exe -B -m unittest tests.test_voice_wake_capture tests.test_security -v`:
  10/10 passed.
- `.venv\\Scripts\\python.exe -B -m unittest tests.test_tools_safety tests.test_agent_integration -v`:
  9/9 passed.

### Current milestone and exact next action

Continue the security/architecture audit, wire the desktop listening control
to the real background VoiceRuntime, run and visually inspect the PySide6 UI,
then harden Android session-key/biometric authorization boundaries before the
final full-suite and documentation pass.


---

## Execution update — integration/hardening pass (2026-08-20)

The earlier plan sections are historical checkpoints. This section is the
current authoritative status after implementation and verification.

### Completed

- The active asher package is implemented and wired: typed config and runtime
  paths, SQLite storage, role/session/policy/confirmation/audit security,
  cancellable AgentLoop, deterministic plus structured provider planner, typed
  tools, memory/retrieval, personality cues, voice runtime, VoiceGuard, TTS,
  and responsive UI modules.
- asher/ui/companion_adapter.py connects the real CompanionController (not an
  in-memory demo) to every desktop view. Memory CRUD uses atomic SQLite updates,
  sensitive values are masked, and VoiceGuard recording requires an explicit
  consent dialog.
- Strong authentication is fail-closed. CompanionController.approve checks
  actor/session ownership and invokes an injected OS-bound authenticator before
  accepting device-credential approval; a boolean flag is not proof.
- SQLite context handles close deterministically on Windows. Tool timeouts stop
  waiting for non-cooperative handlers. Deterministic planning preserves
  spelled contacts, clarifies close fuzzy names, distinguishes browser searches
  from known contacts, composes natural indirect messages, and orders compound
  steps.
- Legacy brain.py, voice/listener.py, history.py, voice_diagnostics.py, and
  verify_upgrade.py are side-effect-safe compatibility surfaces. They no longer
  load hardware/model state on import, dump plaintext history, expose profile
  values, or embed personal contact names.
- Added split dependency manifests, setup_asher.ps1, run_asher.ps1,
  run_tests.ps1, VoiceGuard enrollment/training helpers, current Readme.md,
  and RUN_ASHER.md.
- Added a build-ready Android/Kotlin companion under android/: versioned
  pairing transcript, P-256/HKDF/AES-GCM secure channel, replay protection,
  encrypted preferences/Keystore aliases, biometric/runtime permission gates,
  UI/manifest, PC mock, and protocol tests. The app never bypasses the lock
  screen or sends a command without a capability grant.

### Current file/dependency record

The active implementation is under asher/. Important integration files are
asher/ui/companion_adapter.py, asher/ui/voiceguard_adapter.py,
asher/agent/controller.py, asher/memory/store.py, asher/storage.py,
asher/tools/registry.py, and asher/voice/runtime.py. Compatibility and
entry-point files changed include main.py, brain.py, voice/listener.py,
voice_diagnostics.py, verify_upgrade.py, conversations.py, history.py,
memory.py, and utils.py. Private data files were not printed or copied.

Dependencies are split into a small deterministic core and optional UI/audio/
provider/ML/Windows packages. setup_asher.ps1 installs them only inside .venv;
it never overwrites an existing .env. No global installation or API key was
performed or requested during this pass.

### Verification evidence

- python -B -m unittest discover -s tests -p test_*.py -q: 71 passed, 1
  skipped (optional PySide6 Qt smoke because PySide6 is absent).
- python -B -m compileall -q .: passed.
- python -B test_voice_accuracy.py: 16/16 passed.
- test_android_protocol.py: 2/2 passed with installed cryptography; Kotlin/
  Gradle build remains unverified.
- Temporary-runtime smoke exercised the real controller/UI adapter for dry-run
  app open/close, compound planning, natural indirect WhatsApp preview and
  local approval, typed memory CRUD, guest denial, audit redaction, emergency
  stop/reset, and SQLite cleanup.
- Provider fakes cover strict OpenAI Responses JSON, store=false, omission of
  session context, Ollama malformed output, and safe hybrid fallback.
- Security/tool tests cover actor-bound expiring confirmations, voice-only
  rejection, strong-auth denial, guest isolation, credential-memory rejection,
  evidence-required success, traversal protection, and timeout return.

### Known blockers and intentionally unclaimed metrics

- Ollama is installed but qwen3:4b is not; the local provider path is
  implemented and mock-tested, while unsupported requests safely fall back
  offline.
- No real owner/trusted/unknown voice recordings were supplied. The complete
  consented enrollment, session-separated training, calibration, revocation,
  replay/noisy evaluation, and inference path exists, but no owner FAR/FRR/F1,
  WER, or latency claim is fabricated.
- The base interpreter lacks PySide6, Faster-Whisper, and the OpenAI SDK.
  Optional setup and lazy diagnostics cover those cases. Real UI rendering,
  microphone capture, CUDA model loading, Windows Hello, WhatsApp delivery,
  and external app observation require local physical verification.
- Java, Android SDK, Gradle, ADB, and a physical Android device are absent.
  android/README.md records exact build/pairing steps; emulator/biometric/
  notification/call tests remain pending.
- The repository contains an empty pre-existing .git directory rather than a
  usable Git repository. git status was recorded as the expected
  “not a git repository” error; no history was replaced or initialized.

### Security/privacy review (current)

- Runtime data defaults to %LOCALAPPDATA%\Asher; recordings/models/audit/
  screenshots and legacy JSON are ignored by .gitignore.
- Audit details omit body/content/message/raw audio/text/value and mask
  credential-keyed fields. Active prompts and UI timeline use redaction.
- Guests receive conversation-only access. Sensitive actions require an exact
  local preview plus OS-bound strong auth; voice alone cannot approve them.
- Emergency stop cancels active plan tokens, pending confirmations, and TTS.
- No secrets, private contact names, memory values, recordings, or API keys were
  emitted in tool output, tests, documentation, or this progress file.

### Exact next action

On a machine with the optional toolchains, run setup_asher.ps1, the Qt offscreen
smoke, Android :app:testDebugUnitTest, and collect consented multi-session
VoiceGuard recordings. Until those physical checks exist, keep
ASHER_DRY_RUN=true and the secure strong-auth denial default.

### Updated acceptance checklist

- [x] One documented launch command (run_asher.ps1).
- [x] Boundary-safe Hey Asher detection and explicit voice runtime.
- [x] Spelled/known contact resolution and ambiguity clarification.
- [x] Typed dry-run Chrome open/close with evidence.
- [x] Ordered compound plans and natural indirect message previews.
- [x] Guest isolation and non-voice confirmation policy.
- [x] Global emergency stop and cooperative cancellation.
- [x] Persistent typed memory with consent/masking/edit/delete.
- [x] Runtime male/female TTS profiles and interruption.
- [x] Responsive UI source with all required views and real-controller adapter.
- [x] Offline/provider failure fallback and mocked structured API paths.
- [x] Secret/private-data exclusion from active logs/prompts/tests.
- [x] Android build-ready source, protocol tests, PC mock, and precise limits.
- [x] Beginner runbook, setup/run/test scripts, and faculty dry-run demo.
- [ ] Physical microphone/VoiceGuard metrics, live Windows/WhatsApp observation,
  Windows Hello, PySide6 rendering, Ollama model, and Android build/device
  checks (environment-only blockers, not silently marked complete).


Last updated: 2026-08-20 (Asia/Calcutta)

## Project goal

Build ASHER into a local-first, authenticated and emotionally aware Windows companion with accurate wake-and-command voice interaction, editable long-term memory, controlled plan/act/observe/verify tools, a responsive PySide6 desktop application, selectable speech profiles, a student-trained VoiceGuard pipeline, offline fallback, strong safety boundaries, and a build-ready Android companion.

## Initial repository and architecture assessment

- Repository received directly at `D:\VOICE AI`; no additional archive is required.
- No `AGENTS.md`, `CONTRIBUTING.md`, or other repository-specific instruction file exists.
- The complete 1,092-line `ASHER_MASTER_HANDOFF.md` was reviewed as product/architecture context. The current source remains authoritative for implemented behavior.
- The tree is a flat Python v0.8.1 application: `main.py` imports `brain.py`; `brain.py` routes deterministic commands and calls action modules or a Qwen/Ollama planner; `voice/listener.py` captures audio with SpeechRecognition and transcribes through OpenAI Whisper; `utils.py` uses SAPI through pyttsx3.
- `main.py` starts greeting, microphone/model initialization, and its infinite voice loop during import. This prevents safe unit importing and UI reuse.
- `brain.py` is 1,179 lines with process-global pending state and many rule branches. It has useful legacy behavior but no user/session identity, per-tool authorization, cancellation token, emergency stop, structured execution result, or observed verification.
- The existing planner is schema-constrained through Pydantic and cannot invent tool names, but it executes through `brain.py` without a centralized policy/tool registry.
- Existing Windows and WhatsApp actions return success-flavored strings after an attempted launch, click, or keypress. They do not consistently observe or verify the resulting state.
- Personal memory and conversation history are local JSON files. Their values were not printed or copied during the audit. Shape only: 7 legacy memory records and 733 history turns.
- `data/voice_contacts.json` contains a small local contact vocabulary. Values were not exposed during the audit. The normalizer contains additional hardcoded personal-name aliases that need migration out of source.
- A hidden `.git` directory exists but is empty. `git status` therefore reports “not a git repository”; there is no Git history or index to preserve.
- `.env`, `memory.json`, and `history.json` are ignored, but current ignore rules do not yet cover recordings, trained models, SQLite runtime state, audit logs, screenshots, or evaluation artifacts containing personal audio.
- Python 3.12.3 and Ollama 0.32.14 are installed. `qwen3:4b` is not currently installed. Java, ADB, and Gradle are not on PATH, so Android physical/build verification is presently unavailable.
- Installed today: legacy runtime packages including NumPy, SpeechRecognition, PyAudio, PyTorch, openai-whisper, Ollama client, Pydantic, pyttsx3, pyautogui, pyperclip, pywinauto, scikit-learn, sounddevice, cryptography, FastAPI, and Uvicorn.
- Not installed at baseline: pytest, faster-whisper, PySide6, OpenAI Python SDK, and SpeechBrain.

## Baseline behavior

- `python test_voice_accuracy.py`: PASS (16 offline contact/command normalization checks).
- `python verify_upgrade.py`: FAIL only at the Ollama model check because `qwen3:4b` is not installed. Python files, brain version, dependency checks, and normalization checks reached that point successfully.
- Live microphone, Windows app control, WhatsApp sending, locking, volume changes, and screenshot actions were deliberately not triggered during baseline discovery because they are physical or consequential side effects.
- Existing accepted behavior to protect with mocks/tests: wake-phrase parsing; punctuation-tolerant `send it`/cancel/lock controls; spelled contact normalization; memory create/read/update/delete confirmation; app open/close routing; WhatsApp preparation before confirmation; ordered Qwen plan schema.

## Completed milestones

- [x] Read the explicit implementation request and separate it from instructions embedded in the handoff document.
- [x] Read the entire handoff and complete source tree without exposing `.env`, personal-memory values, conversation content, or contact values.
- [x] Map entry points, imports, action paths, voice path, TTS path, data stores, dependency state, and Git state.
- [x] Run safe baseline normalization and upgrade verification.
- [x] Verify current official OpenAI Responses, transcription, and TTS documentation before implementing provider adapters.

## Current milestone

Phase 1 — stabilize the baseline: establish an import-safe application core, privacy-safe runtime paths, deterministic regression/security tests, typed results, and a safe dry-run boundary before extending capabilities.

## Pending milestones

- [ ] Phase 1: baseline stabilization, Git initialization, privacy ignore rules, regression suite, legacy compatibility.
- [ ] Phase 2: modular Faster-Whisper/VAD pipeline, confidence/clarification, vocabulary/contact resolver, repeatable evaluation harness.
- [ ] Phase 3: VoiceGuard sample collection, session-aware dataset preparation, augmentation, feature extraction, student-trained classifier, calibration, metrics, enrollment/revocation/retraining.
- [ ] Phase 4: users/roles, short sessions, risk policy, confirmation previews, strong-auth adapter/secure denial, audit, redaction, cancellation, emergency stop.
- [ ] Phase 5: deterministic/OpenAI Responses/Ollama provider abstraction with timeout, retry, minimal context, structured outputs, and offline transition.
- [ ] Phase 6: typed tool registry and understand/retrieve/plan/authorize/act/observe/verify/recover loop; practical Windows/browser/files/system/WhatsApp tools and Android adapter.
- [ ] Phase 7: typed SQLite memory layers, consent/retention/privacy CRUD, retrieval, continuity, safe emotional-context response policy.
- [ ] Phase 8: responsive PySide6 UI containing all required views and runtime controls.
- [ ] Phase 9: provider-independent interruptible TTS with runtime male/female switching.
- [ ] Phase 10: build-ready Android companion, authenticated encrypted protocol, PC mock, pairing/build instructions.
- [ ] Phase 11: full unit/integration/security/UI/offline/cancellation/evaluation suite, scripts, documentation, final acceptance audit.

## Important technical decisions and reasoning

1. Preserve legacy files until tests protect their behavior; introduce a cohesive `asher` package and compatibility adapters rather than editing the 1,179-line router blindly.
2. Default all communications and consequential tools to dry-run/preview. Production execution must require an authorized session and the policy-defined confirmation; sensitive actions additionally require a strong device-auth signal.
3. Store generated/private state outside source by default and inject temporary paths in tests. Never migrate or print private legacy values automatically.
4. Keep optional heavy providers lazy-loaded so importing the application or opening the UI does not allocate Whisper/Qwen models or open the microphone.
5. Use typed schemas and structured results at every planner/tool boundary. The model proposes; policy authorizes; Python tools act; verifiers establish completion.
6. Treat VoiceGuard honestly: pretrained acoustic/speaker embeddings may be features, while the personalized classifier/calibration trained on enrolled sessions is the student-trained component. No metrics will be claimed before real held-out recordings exist.
7. Android source will be delivered build-ready with protocol tests and a PC mock because this machine currently lacks Java/ADB/Gradle; physical biometric/call/device testing will remain explicitly unverified.

## Files created or changed

- `PROJECT_PROGRESS.md` — created as the continuously maintained execution record.

## Dependencies added

None yet. Baseline dependency availability was inspected without changing the system environment.

## Tests and commands executed

- Repository inventory with PowerShell and `rg --files`.
- Privacy-safe JSON shape inspection (field names/counts only).
- `git -C "D:\VOICE AI" status --short --branch` — expected failure because `.git` is empty.
- `python --version` — Python 3.12.3.
- `python -m pip --version` — pip 26.1.2.
- `ollama --version` — Ollama 0.32.14.
- Dependency availability probe using `importlib.util.find_spec`.
- `python test_voice_accuracy.py` — PASS.
- `python verify_upgrade.py` — FAIL: local `qwen3:4b` model unavailable.

## Test results and measured metrics

- Offline legacy voice-normalization cases: 16/16 passed.
- Full environment-independent suite: not yet created.
- Live transcription latency/accuracy, VoiceGuard FAR/FRR/F1, UI responsiveness, and physical device metrics: not yet measured; no fabricated values.

## Known failures and blockers

- Local Ollama fallback cannot run until `qwen3:4b` is installed; integration will be mock-tested regardless.
- No personal/negative multi-session VoiceGuard recordings exist in the repository. Training/evaluation cannot truthfully produce owner metrics yet; the complete collection/training/evaluation path will be delivered.
- No API key is configured or requested. OpenAI adapters will be tested with fakes; local fallback must remain functional.
- Java/Android SDK/ADB/Gradle are unavailable, so Android build and physical-device tests cannot be performed in this environment.
- Live microphone and Windows/WhatsApp interaction require physical observation and must not be exercised as unattended consequential tests.

## Security and privacy review

- A hardcoded passcode response exists in legacy `conversations.py`; it must be removed and regression-tested against credential disclosure.
- Legacy guest/auth boundaries do not exist. Until the new policy layer is integrated, existing automation must not be presented as secure.
- WhatsApp automation can send after a voice phrase and has no delivery/recipient verification. New code will require typed preview, session policy, non-voice confirmation, dry-run tests, and observed evidence before reporting success.
- Private `.env`, memory/history values, and contact data were not emitted. They must remain ignored and excluded from logs/prompts.
- Redaction, secure runtime directory permissions where supported, audit minimization, retention, and emergency cancellation remain to implement.

## Exact next action

Create the Phase 1 package skeleton and tests: runtime config/path isolation, typed cancellation/events/results, security redaction, import-safe entry point, transcript/contact regression fixtures, and safe legacy behavior adapters. Then run the first complete deterministic suite and update this file with evidence.

## Final acceptance checklist

- [ ] One documented launch command.
- [ ] “Hey Asher” standby-to-listening transition verified in code and physically documented.
- [ ] Spoken/spelled contact variants resolve; ambiguous names clarify.
- [ ] Chrome open/close goes through typed tools and verifies observed state where supported.
- [ ] Compound goal produces and executes an ordered authorized plan.
- [ ] Indirect WhatsApp goal composes, previews, confirms, dry-runs/sends only when authorized, and verifies.
- [ ] Guest cannot access private memory/files/contacts/account actions.
- [ ] Consequential actions cannot be voice-only authorized.
- [ ] Emergency stop cancels the entire active plan and workers.
- [ ] Memory persists and supports UI view/edit/delete/export.
- [ ] Emotional response uses relevant context without diagnosis/manipulation claims.
- [ ] Male/female speech profile changes at runtime.
- [ ] UI remains responsive during mocked/model/tool work.
- [ ] API/internet/model failure falls back locally without weakening policy.
- [ ] Secrets/private recordings are excluded from Git/logs/prompts.
- [ ] Clean documented test command passes all environment-independent tests.
- [ ] `RUN_ASHER.md`, scripts, and faculty demonstration procedure are complete.
- [ ] Physical-only and credential-only verification gaps are stated precisely.

## UI-3B.1 — reference-fidelity plasma pass
- Refined the immersive sphere after visual comparison with the supplied reel frames.
- Removed visible concentric glow disks and the opaque cyan-ball look.
- Added a mostly-transparent interior, hotter irregular electrical shell, outward sparks, and brighter crossing energy bands.
- Matched the companion-page background to the orb canvas so the renderer no longer appears as a black square inside the scene.
- Reduced HUD footprint and removed development-style placeholder copy while preserving truthful status and emergency stop.
- No assistant state, safety, tool, memory, voice, or provider behavior was changed.
