# Run ASHER (Windows 11 beginner guide)

ASHER is a local-first personal companion with an authenticated session, a
typed planner/tool loop, editable SQLite memory, a consented VoiceGuard
subsystem, and a responsive PySide6 desktop shell. The project starts in
dry-run mode so an action can be inspected without sending a message, deleting
a file, or changing an external application.

## 1. What runs where

* `main.py` is the side-effect-free entry point.
* `asher/agent` owns the understand -> plan -> authorize -> act -> verify loop.
* `asher/brain` contains deterministic intent handling and structured OpenAI /
  Ollama provider adapters.
* `asher/tools` contains allow-listed, typed tools. There is no arbitrary shell
  tool.
* `asher/security` and `asher/core` provide sessions, policy, confirmations,
  redacted audit, cancellation, and the global emergency stop.
* `asher/memory` stores typed records in a private SQLite database.
* `asher/voice` contains wake matching, VAD turn capture, Faster-Whisper,
  vocabulary repair, confidence gating, and the explicit microphone runtime.
* `asher/voiceguard` stores consented WAV manifests and trains only the
  classifier head from enrolled sessions. Optional SpeechBrain/ECAPA features
  are labelled as pretrained; they are not described as student-trained.
* `asher/ui` contains the PySide6 views and the composition adapter connecting
  them to the real controller.
* `android/` is a build-ready companion protocol/app scaffold; Android SDK and
  a physical device were not available in this environment.

Runtime data defaults to `%LOCALAPPDATA%\Asher` (or
`ASHER_RUNTIME_DIR`) and is not stored in source control. Existing legacy
`memory.json`, `history.json`, and contact files are treated as private and
are never printed by the new runtime.

## 2. Prerequisites

Required:

* Windows 11 (Windows 10 is usually adequate for the Python core).
* Python 3.11 or newer. Python 3.12.3 was used for the checked-in tests.
* PowerShell 5+.

Optional capabilities:

* PySide6 for the desktop UI.
* `sounddevice` and a working microphone for voice mode.
* `faster-whisper` plus a downloaded model for local transcription.
* CUDA/CTranslate2 when available; the transcriber falls back to CPU.
* Ollama with the configured Qwen model for local free-form planning.
* An OpenAI API key and the `openai` package for opt-in cloud planning,
  transcription fallback, or online TTS. Never paste a key into chat or commit
  it.
* `numpy` and `scikit-learn` for VoiceGuard training.

Missing optional packages are reported in Diagnostics; they do not make text
dry-run mode unsafe or unusable.

## 3. Install the project

Open PowerShell in the repository directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_asher.ps1
```

The script creates `.venv`, installs `requirements-core.txt`, attempts the
optional stack, and copies `.env.example` to `.env` only when `.env` does
not already exist. It never overwrites an existing environment file. To skip
large optional packages on a first pass:

```powershell
.\setup_asher.ps1 -SkipOptional
```

Edit `.env` locally if needed. Keep `ASHER_DRY_RUN=true` initially:

```dotenv
ASHER_OWNER_NAME=YourName
ASHER_DRY_RUN=true
ASHER_OLLAMA_MODEL=qwen3:4b
ASHER_VOICE_PROFILE=asher_male
# OPENAI_API_KEY=   # optional; do not put a real value in source control
```

## 4. Start ASHER

The one documented launch command is:

```powershell
.\run_asher.ps1
```

It starts the UI when PySide6 is available. If it is not, `main.py` explains
the issue and falls back to the safe text shell. Explicit modes are useful for
demonstrations and diagnostics:

```powershell
.\run_asher.ps1 -Mode ui
.\run_asher.ps1 -Mode text
.\run_asher.ps1 -Mode voice
```

`--live` is intentionally explicit and does not bypass policy:

```powershell
.\run_asher.ps1 -Mode text -Live
```

During an active voice interaction, Companion opens fullscreen by default.
Press **F11** to toggle between fullscreen and a regular resizable Companion
window, or **Escape** to leave fullscreen without ending the conversation.
Returning to Workspace restores its previous normal, maximized, or fullscreen
window state.

Use `--runtime-dir C:\path\to\private\asher-data` when testing in an isolated
directory. Do not point it at the source tree or a directory containing
private files unless that is intentional.

## 5. Text and voice workflow

Text mode accepts commands until `exit`. Supported examples include:

```text
open chrome
close chrome
search Tharika
search T H A R I K A
ask Sai whether he is ready to hang out
send Hello there to Sai
emergency stop
```

`Hey Asher` is recognized with word boundaries; a word such as `washer` does
not wake the assistant. The voice runtime is never started on import. In
`--voice` mode it performs an energy gate, VAD turn capture, local
Faster-Whisper transcription, wake matching, optional VoiceGuard speaker
authentication, and then creates an actor-bound session. Without an enrolled
speaker model, a wake phrase receives only guest permissions.

The first local transcription run may download a model. Set
`ASHER_WHISPER_DEVICE=cpu` when CUDA/CTranslate2 is unavailable. Set
`ASHER_WHISPER_MODEL` to a model already present locally if network access is
restricted. A low-confidence cloud fallback is opt-in and disabled for
security-sensitive turns.

## 6. Roles, permissions, and confirmations

* **Owner**: the bootstrapped local account. Owner permissions cannot be
  silently removed.
* **Trusted**: an enrolled account with explicit per-capability grants.
* **Guest**: conversation only; no private memory, files, contacts, or account
  actions.

Sessions expire and are bound to the actor and authentication method. Risk 0
conversation is immediate. Risk 1 harmless local actions require an
authenticated session. Risk 2+ actions first return a preview containing the
exact target and effect. Approve that preview in the local UI or with the
text-mode `approve <id>` command. Voice alone, `approve` from an unrelated
session, and expired confirmations are rejected. Risk 3+ additionally asks an
injected OS-bound strong authenticator (Windows Hello adapter when configured);
the default when unavailable is denial.

The Workspace header shows **SESSION ACTIVE** or **SESSION EXPIRED**. An
expired desktop owner session never renews from a click alone: use the
Workspace-only **Re-authenticate** button and complete the real device-auth
prompt. Confirmations created under the expired session stay bound to it and
must be requested again; they are never transferred to the new session.

The red **EMERGENCY STOP** button and the `emergency stop` voice/text command
cancel the active plan, clear pending confirmations, stop speech, and latch a
global stop token. Reset is a deliberate local UI action.

## 7. Memory

The Memory page reads and edits typed SQLite records with source, confidence,
timestamps, type, sensitivity, consent, and optional expiry. Sensitive values
are masked until the local reveal checkbox is selected. Passwords, PINs, API
keys, tokens, and credential-like values are rejected. Create/update/delete are
explicit local UI operations and are audited without storing the value in the
audit log. The owner can use **Memory -> Export JSON** after fresh device
authentication; the export audit records only format/count metadata, never the
destination or memory values. No legacy conversation history is imported
automatically.

## 8. PreferenceCore feedback capture

PreferenceCore is the local personalization dataset for teaching ASHER how the
owner prefers the assistant to behave. It does **not** control permissions or
security policy, and it is disabled by default so normal conversations are not
silently collected as training data.

Enable explicit capture from an authenticated owner session:

```text
preference learning on
```

After an ordinary ASHER response, label only that immediately preceding
response with one of:

```text
feedback accept
feedback reject
feedback shorter
feedback more detailed
feedback more direct
feedback ask less
feedback ask more
feedback suggest more
feedback suggest less
feedback preferred: Opening VS Code.
```

Use `preference status` to see whether capture is enabled and how many explicit
examples are stored. `preference learning off` stops new capture without
deleting existing examples. Confirmation previews, local safety messages and
credential-like text are never eligible training examples. Feedback text stays
in the local SQLite runtime store and is not sent to a model trainer yet.

## 9. VoiceGuard enrollment and training

VoiceGuard recording is consent-first. The UI's **Users & VoiceGuard** page
asks for consent before a microphone sample is retained. If `sounddevice` or a
device is unavailable, it reports that fact and does not fabricate a sample.
Each UI click adds one clip to a six-clip collection session; partial sessions
remain unregistered and cannot be used as artificial train/validation/test
partitions. After an adapter restart, fresh recording consent reopens the one
matching partial session and reports its saved clip count; ambiguous partials
are surfaced for explicit recovery and are never silently registered. Revoking
the user also revokes matching unregistered partial manifests. Session updates
are serialized across desktop instances and processes, refresh disk state before
mutation, and bind consent to the observed enrollment generation, so a stale
collector cannot overwrite revocation or silently re-enroll a user.

For a controlled WAV import, put several clips from one real recording sitting
in the same session (repeat `--wav`):

```powershell
.\.venv\Scripts\python.exe -B voiceguard_enroll.py demo-user --role trusted --consent `
  --wav C:\private\clip-1.wav --wav C:\private\clip-2.wav --wav C:\private\clip-3.wav
```

The imported/recorded sample is a negative wake-word example unless
`--wake-phrase` is supplied. Collect both positive (`--wake-phrase`) and
negative sessions before training the optional wake-word task; speaker
authentication labels are stable user IDs, never only broad roles.

For the guided real speaker-auth workflow, use the collector in the final
section of this runbook. One invocation is one multi-clip session. Do not turn
sequential clips from one sitting into artificial train/validation/test
sessions.

Inspect aggregate-only readiness without importing ML libraries or exposing
recording paths/identities:

```powershell
.\.venv\Scripts\python.exe -B train_voiceguard.py --check
```

The command exits with status `2` while more real data is required and lists
only structural deficits. A first two-class speaker baseline needs at least
three independent owner sessions and three independent sessions for the same
consented negative identity, with at least three unique clips per session. Six
clips per session is the guided default. Replay trials are evaluation-only and
never replace the required negative identity or enter classifier fitting.
For repeated same-environment sessions, readiness requires at least a 30-minute
metadata interval; a declared different environment is separate collection
evidence. The desktop collector pauses an immediate repeat and directs the user
to wait or use the guided collector in a genuinely new environment. These
timestamp/environment checks are conservative quality guards, not proof that
recordings were physically independent. Rapid same-environment extras are
deterministically excluded from both the readiness split and the actual training
dataset, so correlated recordings cannot cross held-out partitions.

When the check reports `READY`, train and activate the private model:

```powershell
.\.venv\Scripts\python.exe -B train_voiceguard.py
```

Training uses session-separated partitions, calibrates a threshold on held-out
data, rejects cross-session duplicate audio, and activates the model only after
the versioned model plus validation/test reports are saved under ASHER's private
runtime directory. It reports measured accuracy/F1/FAR/FRR plus authorized
identity errors; a speaker is authorized only as the single identity the model
actually predicted. Real replay/noisy metrics remain unavailable until those
real conditions are recorded. VoiceGuard is never sufficient proof for
payments, account security,
or other high-risk actions; device authentication still applies.

Versioned model/report paths are immutable: a matching destination aborts the
retrain instead of overwriting prior evidence, and an incomplete bundle is not
activated. Finalizing or revoking biometric enrollment immediately clears stale
speaker/wake bindings. The runtime revalidates every model-authorized identity
against active biometric and application enrollment, the content-addressed
model, and the verified full finalized-dataset fingerprint before and after
inference, so a cached verifier fails closed after revocation or manifest/audio
change. Model fitting runs outside the lifecycle lock; activation occurs only
after a registry-and-dataset compare-and-swap, keeping revocation responsive and
preventing a stale fit from resurrecting a model.

## 10. Male/female voice switching and TTS

Open **Settings -> Speech output** in the UI and choose `asher_male` or
`asher_female`. The profile registry hides provider-specific voice IDs, so the
selection applies without restarting. Windows SAPI is the offline fallback;
OpenAI TTS profiles are optional, disclose that speech is AI-generated, and
fall back safely when unavailable. Pressing emergency stop interrupts active
speech. The same interface is available through `asher.voice.tts.speak()`.

## 11. Ollama and OpenAI

Install Ollama separately, then pull the local model (only if you want that
optional route):

```powershell
ollama pull qwen3:4b
ollama serve
```

ASHER catches model/network failures and returns a safe offline response; it
does not weaken permissions. OpenAI is similarly opt-in via a local `.env` and
uses a structured Responses API plan with minimal context and `store=false`.
The code also supports the official audio transcription and TTS adapters, but
no key or paid request is required for the tests.

## 12. Tests and faculty demonstration

Run everything:

```powershell
.\run_tests.ps1
```

Or use the standard-library command directly:

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -s tests -p 'test_*.py' -v
```

The environment-independent suite passes; the Qt smoke test is skipped when
PySide6 is absent. `test_voice_accuracy.py` runs the deterministic legacy
voice smoke test without recording.

Suggested dry-run demo:

1. Run `.\run_asher.ps1 -Mode text`.
2. Show `open chrome` and `close chrome`; explain that evidence is verified
   and dry-run does not change the desktop.
3. Run `ask Sai whether he is ready to hang out`; show the exact confirmation
   preview and reject it, then run it again and approve in the local UI.
4. Show the Memory, Users, Permissions, Activity, Diagnostics, and Settings
   views when PySide6 is installed.
5. Press **EMERGENCY STOP**, demonstrate that a pending action disappears,
   then reset locally.
6. Explain that no real WhatsApp message, deletion, booking, or payment is
   performed during the demo.

## 13. Troubleshooting

* **UI unavailable**: install `PySide6` in `.venv`; text mode is the supported
  fallback. On a headless test machine set `QT_QPA_PLATFORM=offscreen` only
  for smoke tests.
* **Microphone unavailable**: install `sounddevice`, check Windows privacy
  settings and the Diagnostics microphone index, then try text mode. No audio
  is retained when setup fails.
* **Faster-Whisper/CUDA failure**: set `ASHER_WHISPER_DEVICE=cpu`, use a
  smaller model, or install the matching CUDA/CTranslate2 build. The loader
  automatically retries CPU after a CUDA load/transcription failure.
* **Ollama unavailable**: verify `ollama serve` and `ollama list`; missing Qwen
  is a normal offline blocker and deterministic commands still work.
* **OpenAI unavailable**: check that the key is configured locally and the
  optional package is installed. Never paste the key into logs or chat.
* **WhatsApp**: dry-run only simulates preparation/delivery. Live mode needs
  the explicit `ASHER_ENABLE_LIVE_WHATSAPP` gate, a UIA adapter, and an
  observable delivery verifier; otherwise it fails closed.
* **VoiceGuard training error**: collect multiple sessions for every class,
  include both authorized and unauthorized examples, and install NumPy plus
  scikit-learn. The trainer rejects insufficient data rather than inventing
  metrics.
* **Android build**: open `android/` in Android Studio after installing its JDK
  and SDK. This checkout does not contain Gradle wrapper scripts/JAR, and the
  local environment has no Java, Gradle, SDK, ADB, or physical device, so the
  command-line build/device test remains pending.

## 14. Adding a tool safely

1. Create a focused module under `asher/tools/`.
2. Define a JSON input schema with bounded lengths and
   `additionalProperties: false`.
3. Assign a capability and `RiskLevel`; require confirmation for external or
   sensitive effects.
4. Implement timeout-aware cancellation and return `ToolResult` with concrete
   evidence. Never report success because a click/keystroke was merely sent.
5. Register it in `asher/tools/catalog.py` and add a deterministic unit test for
   authorization, dry-run, failure, and verification.
6. Keep secrets and payloads out of `AuditLog`, prompts, tests, and UI details.

## 15. Known limits

The source includes an Android protocol/app scaffold, but this checkout has no
Gradle wrapper and the local machine has no Java/Android SDK/device toolchain,
so an Android build or device claim cannot yet be made. A complete held-out
VoiceGuard dataset, optional model downloads, Ollama's Qwen model, an OpenAI
key, Windows Hello, and live external-action observation are intentionally not
fabricated. See `PROJECT_PROGRESS.md` for exact evidence and next actions.

## 16. VoiceGuard real-data collection

Use the guided collector for real speaker-auth sessions. It keeps raw audio in ASHER's private runtime directory; do not copy those WAV files into the project or upload them.

Owner session (the script resolves the real persistent owner user ID automatically):

```powershell
.\.venv\Scripts\python.exe -B voiceguard_collect.py --speaker owner --environment quiet_room --samples 6 --consent
```

A consented non-owner/negative speaker can be collected into the dataset-only `unknown_pool` class:

```powershell
.\.venv\Scripts\python.exe -B voiceguard_collect.py --speaker unknown --speaker-id unknown_pool --environment quiet_room --samples 6 --consent
```

Each invocation is one recording session. Record at least three sessions per
class across different times/environments so train, validation and test remain
session-separated. Obtain the other speaker's consent every time; raw audio
never belongs in Git or chat. Then run `train_voiceguard.py --check`. Never
treat VoiceGuard as sufficient authorization for high-risk actions.
