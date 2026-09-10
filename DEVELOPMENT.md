# Development notes

Things that were expensive to discover. Most of them look like small details
and are not: several cost a whole working session, and a few produced output
that looked correct while being wrong.

## Layout

```
avilistener/            Python package
  audio.py              device discovery (soundcard/WASAPI)
  recorder.py           silence segmenter, adaptive gate, WAV writing
  file_transcriber.py   WAV directory -> transcript, filename parsing
  transcriber.py        faster-whisper wrapper, hallucination filters, and
                        build_transcriber(): settings dict -> Transcriber
  writer.py             transcript files, and clearing them before a re-run
  meeting.py            diarization, speaker-accurate line splitting
  combine.py            merge a diarized loopback with other transcripts
  timeline.py           read-only inputs, compact audio, per-part clock mapping
  processing.py         staged publication, multipart jobs, content-keyed caches
  speakers.py           per-speaker audio samples for identification
  cli.py                dashboard, list-devices, levels — nothing else
  server/               dashboard: workspace model, jobs, FastAPI
discord-receiver/       Node.js Discord bot, one WAV per participant
web/                    React dashboard (Vite)
e2e/                    Playwright tests driving the real dashboard
scripts/                diarization engines, each run in its own virtualenv
```

Three virtualenvs: `.venv` for everything, `.venv-pyannote` and `.venv-nemo`
for the diarization engines, because their dependencies conflict. The scripts
in `scripts/` therefore cannot import `avilistener`, which is why diarization
preprocessing lives in `meeting.py`.

## The filename contract

Every producer writes `<ISO timestamp>Z-<name>-<digits>.wav`, and
`file_transcriber.py` parses the name back into a speaker and a start time.
That one convention is why the Discord receiver and the loopback recorder feed
the same transcription code with no adapters.

The trailing field must be digits. Session recordings deliberately break the
pattern (`-continuous.wav`) and live in a subdirectory, so an hour-long file is
never mistaken for a single utterance. The filename carries each part's UTC
start. `timeline.py` uses this to map compact-audio seconds back to wall time;
`combine.py` still accepts the older single-recording RTTM convention.

## Resumed sessions and safe reprocessing

`Meeting.shared_parts()` enumerates all continuous shared-source recordings,
not just the most recent. `processing.diarize_meeting()` streams them into a
mono 16 kHz `session-audio.wav` in a fresh staging directory. No silence is
inserted for outages; `timeline.json` records source, original filename, SHA-256,
original rate/frame count, absolute start/end, compact offset and duration.
Word lines are split at part boundaries before mapping to wall time, even when
the same speaker speaks on both sides. RTTM turns and speaker samples use the
compact audio clock, while events also carry absolute start/end for merging.

Only clips fully covered by a diarized part of the same source are excluded
from the merge. Boundary/uncovered clips stay; this conservatively favours
retaining speech if alignment is uncertain. Small clock tolerances account for
the recorder's clip timestamps. Ordinary simultaneous microphone/remote speech
remains overlapping in the merged output.

Every step writes a new `.processing/` directory and publishes after success.
Previous outputs are moved to `artifact-history/`, never erased first. These
paths contain derived data only; `recordings/` is strictly read-only during
dashboard processing. Transcription includes legacy `processed/` clips and
caches each result by filename, content and full transcription configuration.
Diarization word reuse is keyed by all part hashes, timeline version and
transcription configuration, independently of the diarization engine/count.
Changing any input invalidates reuse. Speaker names reset after a new split;
old names are preserved in `previous-speaker-names.json` for reference.

`tests/test_multipart.py` covers both sides of a restart, gap mapping, samples,
sample-rate changes, cache invalidation, repeatability, failed publication and
byte-for-byte preservation of input recordings. Use `scripts/verify_recordings.py`
to verify a real recording directory against a separate backup and persist an
integrity manifest beside that backup.

## Audio capture

**Segments and the session file are written at the same time**, not as
alternatives. Clips are transcribed; the session file is what diarization needs,
because splitting by voice depends on the real timeline including silences.
They are also independent, which matters when something breaks: clips are closed
as written and survive a crash, while the session file's header is only written
on close.

**Never kill a recording process.** The WAV header is rewritten at close, so a
killed process leaves a file claiming zero length and the whole session is
unreadable. This is why the dashboard owns recorder threads in-process and stops
them with an event: Windows cannot deliver a polite signal to a child process.

**The speech gate adapts to each source's noise floor.** A quiet microphone and
a speaker loopback differ by more than 10x, so one fixed threshold either forces
shouting into the microphone or records hiss from the loopback. Measured here:
microphone speech 0.004-0.008 RMS over a 0.0002 floor; loopback 0.04 over
0.00003. Use `avilistener levels` to see real numbers before tuning anything.

**Recording captures the default device unless told otherwise**, and a wrong
device is silent in a way nothing notices until the transcript is empty. Both
the recorder log and the dashboard warn when a source peaks below 0.01.

**Chrome-tab capture is push-based.** The dashboard calls `getDisplayMedia`,
downmixes the chosen tab through Web Audio and sends ordered float32 PCM chunks
to the local API. `BrowserStreamRecorder` presents the same lifecycle and
metrics as the WASAPI threads, then feeds `SilenceSegmenter` and
`ContinuousWriter` directly. The WAV sample rate comes from the browser's actual
`AudioContext`; assuming the requested 16 kHz would make timestamps drift on a
browser that chose another rate.

## Why PC audio, and not an integration with the meeting app

Capturing the speaker output looks cruder than talking to the meeting platform,
and is the reason this works everywhere. Google Meet was tried properly and is
the cautionary tale:

- **Meet decodes remote audio in WebAssembly.** It ships its own NetEQ and
  renders through WebAudio, so the remote `MediaStreamTrack` a page can reach
  carries no audio. An extension can capture your own microphone and nothing
  else, no matter how the track is wired. Patching `AudioNode.prototype.connect`
  to tap what Meet renders crashes the renderer outright
  (`STATUS_BREAKPOINT`), because that graph is driven from WASM on the audio
  thread.
- **Google blocks automated browsers.** An anonymous Playwright client is
  refused outright, and even when it joins it is ejected within about a minute.
  A bot that joins meetings is not a workable approach.

Recording the machine's audio output has none of these problems and does not
care which application is making the sound.

The optional **Chrome tab** source is deliberately different from the failed
Meet integration above. It uses Chrome's user-approved tab sharing output, not
Meet's internal remote track or WebAudio graph. This isolates one tab cleanly,
but Chrome must show its picker for every recording and the user must enable
tab audio.

## Diarization

**Whisper segments are not speaker turns.** Whisper emits 10-15 second segments
that routinely span several turns, so labelling whole segments loses every
speaker change inside one and a participant with only short turns disappears.
Lines are cut using word timestamps, across the whole transcript rather than per
segment, otherwise a speaker's line breaks wherever Whisper ended a segment.

**Words are cached (`words.json`).** Transcription is the expensive step, and
the words depend only on the audio. Re-running a split with another engine or
another speaker count re-splits the stored words instead of touching the GPU.

**NeMo's VAD judges absolute level.** Whisper normalises internally and
transcribes quiet audio fine, so a quiet recording produces a normal transcript
and almost no speaker turns, and every line falls back to `SPEAKER_UNKNOWN`.
Audio is boosted before diarization; only diarization sees the boosted copy,
since amplifying Whisper's input invites hallucination over silence. Gain comes
from a high percentile rather than the peak, because dropped samples produce
clicks and one full-scale sample would otherwise suppress the gain entirely.

**Never reuse a NeMo output directory.** `pred_rttms/` accumulates, and
selecting the result with `sorted(...)[0]` returned the *first recording ever
diarized*. Every run after the first silently reused old speaker turns: the
transcript looked plausible while every label belonged to another meeting. The
RTTM is now selected by name and the work directory is cleared first.

**pyannote community-1 is the default because it was measured better.** On a
2h51m five-voice session it found all five speakers unaided, handled overlapping
speech and ignored background music, in 2 minutes. NeMo with the count forced to
five produced three usable clusters in 18 minutes. NeMo's auto counting
under-counts on short or similar-sounding audio; with NeMo, set the count when
you know it.

## Discord

The receiver writes one WAV per participant, already in the filename contract,
so each Discord username becomes the speaker with **no diarization at all**.
When Discord applies it is the best of the sources.

The token comes from `DISCORD_BOT_TOKEN` in the environment; the dashboard falls
back to it and only stores a token if one is typed.

## Testing

```powershell
.\.venv\Scripts\python.exe -m pytest      # unit tests
cd e2e; npm install; npx playwright test  # dashboard, end to end
```

The end-to-end tests start the real server and drive the real interface with
Playwright. They run against a throwaway workspace in the system temp folder
(`AVILISTENER_WORKSPACE`), so they never touch the one holding real meetings.

Recording, transcription and diarization are seeded on disk rather than run:
they need a microphone, a GPU and a Whisper model, and a test that needs those
is a test nobody runs. What is left is the part that is this project's own
logic — what may be done next, merging two clocks into one timeline, naming
speakers, and keeping tokens out of the browser.

Levels used in unit tests come from real recordings rather than round numbers,
so a regression is measured against audio that actually occurred.

## Known gaps

- Nothing can be deleted from the dashboard, and the workspace only grows.
  Roughly 115 MB per source per hour, doubled because clips and the session
  file are both kept.
- The job registry never evicts, so a long-running server slowly leaks.
- No authentication. It binds to localhost; exposing it would expose the
  Discord token and every recording.
- Recording is Windows-only (WASAPI loopback). The transcription, diarization
  and merging code has no such dependency.
