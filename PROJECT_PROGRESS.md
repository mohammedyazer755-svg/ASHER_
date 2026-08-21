# ASHER Project Progress

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
