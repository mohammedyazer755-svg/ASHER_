# ASHER

ASHER is an authenticated, local-first Windows companion. It listens only
when an explicit voice/text mode is started, routes simple requests through a
deterministic planner, and sends harder planning to an optional structured
provider (Ollama or OpenAI). Every external or sensitive action passes a typed
tool registry, role/risk policy, a preview, and observable verification.

The safe default is dry-run/offline mode. No API key, private contact, memory,
recording, or legacy chat history is required to run the testable core.

## Start here

On Windows 11, run PowerShell in this directory:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\setup_asher.ps1
.\run_asher.ps1                 # UI when PySide6 is installed; text fallback otherwise
```

Useful explicit modes:

```powershell
.\run_asher.ps1 -Mode text
.\run_asher.ps1 -Mode voice       # microphone opens only here
.\run_asher.ps1 -Mode ui
.\run_tests.ps1
```

Read [RUN_ASHER.md](RUN_ASHER.md) for setup, Ollama/OpenAI configuration,
VoiceGuard enrollment, confirmations, troubleshooting, and the faculty demo.
`PROJECT_PROGRESS.md` is the continuously maintained engineering record.

## Architecture

```text
main.py
  -> CompanionController (session + planner + cancellable AgentLoop)
       -> HybridPlanner (deterministic -> OpenAI Responses -> Ollama -> safe fallback)
       -> ToolRegistry (schema -> policy -> confirmation -> timeout -> evidence)
       -> SQLite MemoryStore / redacted append-only AuditLog

voice/runtime.py -> VAD/turn capture -> Faster-Whisper -> vocabulary/wake gate
                  -> optional VoiceGuard speaker gate -> authenticated session
ui/               -> responsive PySide6 views through companion_adapter.py
voiceguard/       -> consented WAV manifests, session splits, trained classifier
android/          -> build-ready encrypted companion protocol scaffold
```

The older flat modules remain compatibility shims, but the `asher/` package is
the active composition root. Importing `main`, the voice package, or the UI
does not open a microphone, start TTS, contact a network service, or dump
private history.

## Safety and privacy

* Owner, trusted, and guest roles use short-lived actor-bound sessions.
* Guests can converse but cannot access private memory, files, contacts, or
  account-connected tools.
* Risk 2+ actions show an exact target/effect preview. Voice alone cannot
  approve them; risk 3+ requires an injected OS-bound strong authenticator and
  otherwise fails closed.
* Emergency stop cancels the active plan, pending confirmations, and speech.
* Audit entries contain event metadata and redacted outcomes, never message
  bodies, credentials, memory values, or raw audio.
* Runtime data defaults to `%LOCALAPPDATA%\Asher` and is excluded from Git.

Keep `ASHER_DRY_RUN=true` until you have reviewed the policy and verified the
target application. Live WhatsApp delivery additionally requires an explicit
environment gate and an adapter that can observe delivery; an attempted click
is never reported as success.

## Optional providers

Ollama is local and is used as a fallback when the configured model is present.
OpenAI is opt-in through `OPENAI_API_KEY` in a local, ignored `.env`; never put
the value in chat, source, tests, or issue reports. The provider sends only a
minimal request and uses structured JSON output with `store=false`.

## Tests

The environment-independent suite uses Python's standard unittest runner:

```powershell
python -B -m unittest discover -s tests -p 'test_*.py' -v
```

Optional Qt smoke tests are skipped when PySide6 is not installed. VoiceGuard
metrics intentionally report unavailable conditions when real noisy/replay
recordings are absent; no accuracy or authentication result is fabricated.

## Status

The implementation and test evidence, including environment-only blockers
(microphone, API key, Ollama model, Android SDK/device), are recorded in
[PROJECT_PROGRESS.md](PROJECT_PROGRESS.md).
