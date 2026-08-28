# VOICE-2 Continuation

## 1. CURRENT GOAL

Deliver VOICE-2: a reliable local-first daily-use voice pipeline with a dedicated wake boundary, separate VoiceGuard speaker authentication, one complete command turn, measured local STT, observable microphone failures, the existing Qwen3.5:9b controller path, and truthful TTS/UI lifecycle states. Do not resume UI polish or PreferenceCore work.

## 2. CURRENT SUB-MILESTONE

VOICE-2A - AUDIT + BASELINE is complete. The next sub-milestone is VOICE-2B - WAKE BOUNDARY; no production edit for 2B has started yet.

## 3. WHAT WAS ALREADY COMPLETED

- Read both user milestone briefs completely.
- Ran the required repository status and recent-history checks.
- Inspected the current voice runtime, wake binding, VAD capture, Faster-Whisper wrapper/config, dynamic vocabulary, microphone backend, desktop adapter, controller routing, TTS lifecycle, and principal voice tests.
- Inspected the pre-existing uncommitted diff before editing anything.
- Ran the complete existing test command once.
- Completed three independent read-only audits covering low-level voice code, UI/controller/security integration, and regression/benchmark coverage.
- Ran the legacy deterministic text-normalization smoke directly because the full runner stopped before reaching it.
- Enumerated the real local environment without opening the microphone:
  - Python voice packages present: Faster-Whisper 1.2.1, CTranslate2 4.8.1, sounddevice 0.5.6, openWakeWord 0.6.0.
  - GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8188 MiB total; it was idle when sampled.
  - Windows/sounddevice default input index: 1 (Realtek microphone array). `.env` has no `ASHER_MIC_INDEX`, so automatic selection is active.
  - Only `small.en` is currently cached locally; no stronger Faster-Whisper candidate is cached.
  - The VoiceGuard registry exists but has 0/3 wake binding keys, no wake model file, and no active speaker model file. The actual runtime therefore uses text wake fallback and guest access today.
- Established these verified implementation facts:
  - `VoiceRuntime.run_forever()` currently captures and transcribes every energy-triggered utterance before checking `TextWakeDetector`.
  - A trained `WakeWordModelBinding` is currently invoked only after exact transcript wake matching, so acoustic wake acceptance cannot rescue a mis-transcribed “Hey Asher”.
  - `VoiceRuntime` emits a permanent `transcript` event before wake acceptance/rejection; `CompanionDesktopController._on_voice_event()` appends that event to Conversation history.
  - Wake and speaker model artifacts/loaders are separate, and wake rejection already prevents speaker authentication once the text gate has been passed.
  - `TurnCapture` does return a single buffer and ends after configured silence, but the runtime creates a new capture from the first energy frame with no retained standby pre-roll. Its 550 ms endpoint can split natural pauses.
  - The same `CompanionController.handle_text()` method is used by voice and desktop text. Voice replies already call asynchronous TTS.
  - `ASHER_MIC_INDEX` already parses a non-negative numeric index or exact non-empty device name, and the desktop adapter passes it lazily to `SoundDeviceBackend`.
  - Microphone exceptions stop the adapter’s listening flag and publish ERROR, but the visible message currently exposes only the exception class, not actionable device/context detail.
  - The direct CLI/diagnostic construction path ignores `config.microphone_index`; only the desktop adapter explicitly builds `SoundDeviceBackend(device=...)`.
  - The runtime continues command capture while TTS is speaking, so speaker echo can be mistaken for barge-in and stop TTS.
  - The window status timer does not refresh Conversation rows, explaining reports where accepted voice text does not visibly appear until another UI refresh.
  - A voice-issued emergency-stop command latches the agent loop but does not currently stop the voice runtime/TTS; the local UI emergency button does.

## 4. EXACT FILES CHANGED

- `VOICE2_CONTINUATION.md` — created as the authoritative resumable VOICE-2 handoff.

No production or test source file has been changed for VOICE-2 yet.

## 5. WHY EACH FILE WAS CHANGED

- `VOICE2_CONTINUATION.md`: required by the continuation protocol so the milestone can resume from repository state alone.

## 6. TESTS ALREADY RUN

- `& .\run_tests.ps1`
- `.\.venv\Scripts\python.exe -B test_voice_accuracy.py`
- Read-only audit agent focused run: `python -B -m unittest tests.test_voice_wake_capture tests.test_wake_word_runtime_binding tests.test_ui_adapter tests.test_tts tests.test_agent_integration tests.test_security`
- Read-only audit agent broader focused voice/evaluation/security run (exact module selection recorded in the agent audit): 67 tests.

## 7. EXACT TEST RESULTS / COUNTS

- Syntax/import compilation completed successfully before unittest discovery.
- Unittest discovery ran 214 tests in 120.537 seconds.
- Result: 213 passed, 1 failed.
- Failure: `tests.test_companion_mode.CompanionMinimalUiContractTests.test_fullscreen_is_scoped_to_companion_mode` at `tests/test_companion_mode.py:155` because `window.mode_stack.currentWidget()` was not `window.companion_mode`.
- The runner stopped at that unittest failure, so the deterministic legacy voice smoke step was not reached in this baseline invocation.
- Direct legacy normalization smoke result: PASS (exit 0). This checks static text normalization only; it is not an acoustic wake/STT accuracy measurement.
- Focused integration audit result: 54/54 passed in 11.692 seconds.
- Broader focused audit result: 67/67 passed. These focused runs overlap and must not be added together as unique tests.

## 8. CURRENT KNOWN FAILURE

Primary VOICE-2 failure: standby wake acceptance is ordered as energy/VAD capture → Whisper → exact text wake match → acoustic wake binding. This makes Whisper’s spelling of “Hey Asher” mandatory even when a trained acoustic wake verifier exists.

Secondary failures: pre-wake/rejected transcripts can pollute permanent Conversation history; endpointing begins without retained pre-roll and can cut/split natural turns; microphone failure copy is not actionable; real start-stop-start and full end-to-end voice acceptance have not been physically verified.

Additional verified lifecycle failures: capture remains active during TTS and can self-trigger on speaker echo; a two-second cooperative stop does not guarantee the old blocking PortAudio reader released the device before restart; direct CLI mode ignores configured microphone selection; voice emergency stop does not terminate the listening runtime.

Unrelated baseline failure: the single companion fullscreen test listed above. Do not turn this milestone into UI work; re-run it later only to distinguish deterministic failure from baseline flakiness.

## 9. ROOT CAUSE IF KNOWN

- `asher/voice/runtime.py:823-849`: `pipeline.process()` performs expensive STT before `wake_detector.detect(heard)` and before `WakeWordModelBinding.verify()`.
- `asher/voice/runtime.py:830`: `transcript` is emitted before wake acceptance.
- `asher/ui/companion_adapter.py:285-294`: every event named `transcript` is persisted to the in-memory Conversation list.
- `asher/voice/runtime.py:930-978`: standby discards silence frames until the first energy hit and then creates a fresh default `TurnCapture`; earlier onset/pre-roll and learned ambient noise are unavailable.
- `asher/voice/capture.py:22`: 550 ms end silence is aggressive for natural sentences and can yield separate turns around ordinary pauses.
- `asher/ui/companion_adapter.py:251-270`: runtime errors are observable as ERROR but the user-facing text is only `Voice input unavailable: <ExceptionType>`.
- `asher/voice/runtime.py:787-803,948-954`: the loop still captures while TTS is speaking and any energy stops speech, making laptop-speaker echo a command candidate.
- `asher/ui/companion_adapter.py:190-217` and `asher/voice/runtime.py:83-97`: the adapter drops runtime ownership before a bounded join, while the backend has no explicit close/unblock hook.
- `asher/ui/window.py:1052-1078,1105-1119`: periodic refresh updates status but not Conversation data.
- `main.py:74-83` and `voice_diagnostics.py:36-45`: default `VoiceRuntime()` construction does not pass `config.microphone_index` to its default backend.
- `tests/test_wake_word_runtime_binding.py:162-176`: an existing regression explicitly enforces the wrong text-before-acoustic order and must be replaced for VOICE-2B.

## 10. WHAT IS STILL UNIMPLEMENTED

- VOICE-2B dedicated acoustic-before-STT wake path and clearly labelled text fallback.
- VOICE-2C robust full-turn buffering/pre-roll/endpointing and final-transcript-only persistence.
- VOICE-2D real current-vs-stronger STT benchmark, model selection, resource observations, and explicit “Asher” vocabulary bias.
- VOICE-2E actionable microphone diagnostics and deterministic Start → Stop → Start coverage.
- VOICE-2F end-to-end regression coverage, full suite, documentation, and real-machine acceptance commands/results.
- No real personalized wake success-rate measurement exists yet; never claim the 9/10 target.

## 11. EXACT NEXT ACTION

Implement VOICE-2B as the smallest coherent change: make a trained acoustic `WakeWordModelBinding` decide a standby candidate before STT, keep exact text matching only as a clearly emitted fallback when no trained artifact exists, and retain speaker authentication as a later independent step. Replace the regression that currently requires text before the acoustic verifier.

## 12. EXACT NEXT FILE/FUNCTION TO INSPECT OR EDIT

- Edit `VoiceRuntime.run_forever()` in `asher/voice/runtime.py`, extracting a small standby wake-decision helper so `WakeWordModelBinding.verify()` can run before `VoiceAccuracyPipeline.process()`.
- Update `tests/test_wake_word_runtime_binding.py:162-209` to assert trained acoustic acceptance does not depend on exact Whisper wake text and rejected acoustic wake never calls STT or speaker authentication.
- Do not edit the planner/provider files.

## 13. EXACT COMMAND TO RUN NEXT

```powershell
.\.venv\Scripts\python.exe -B -m unittest tests.test_wake_word_runtime_binding tests.test_voice_wake_capture -v
```

After the focused tests, run:

```powershell
git diff --check
```

## 14. ANY UNCOMMITTED WORK

Pre-existing user work (present before VOICE-2; preserve and do not include in VOICE-2 commits):

- `asher/brain/providers.py`
- `asher/config.py`
- `asher/ui/window.py`
- `tests/test_providers.py`
- `tests/test_ui.py`

VOICE-2 work currently untracked:

- `VOICE2_CONTINUATION.md`

## 15. ANY IMPORTANT ARCHITECTURAL DECISIONS

- Actual code/tests are authoritative over old progress notes.
- A trained acoustic wake artifact must be evaluated before Whisper and independently from speaker authentication.
- VOICE-2B will first use the already security-bound `WakeWordModelBinding` on a complete standby candidate. The openWakeWord adapter remains a valid future streaming provider, but it must not be wired without a real compatible personalized model, correct frame sizing, cooldown, and measured behavior.
- With no trained wake artifact, text wake matching remains an explicit degraded fallback; it must not be described as equivalent to a personalized detector.
- Wake acceptance activates listening but never authorizes private/high-risk actions.
- Permanent Conversation history receives only a final accepted command transcript, never standby candidates, rejected wakes, or unstable partials.
- Voice commands continue through the exact existing `CompanionController.handle_text()` path and its security/tool policy.
- Do not touch cinematic UI or PreferenceCore except for strictly necessary truthful state/error wiring.

## 16. THINGS THE NEXT SESSION MUST NOT REDO

- Do not repeat the initial source-tree inventory or the 214-test baseline unless code has changed enough to justify a new full-suite run.
- Do not rerun hardware/package/microphone enumeration; the exact audit result is recorded above.
- Do not re-audit or overwrite the five pre-existing modified files.
- Do not add wake-string alias lists.
- Do not claim STT or wake accuracy without measured recordings.
- Do not merge wake acceptance with VoiceGuard speaker authorization.

## 17. CURRENT GIT COMMIT HASH

`7c1947e03c4a29719d8ad93c1636d1df33a1601b`
