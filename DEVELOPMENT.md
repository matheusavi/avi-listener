# Development notes

Things that were expensive to discover. Most of them look like small details
and are not: several cost a whole working session, and a few produced output
that looked correct while being wrong.

## Layout

```
avilistener/            Python package
  audio.py              device discovery (soundcard/WASAPI)
  recorder.py           silence segmenter, adaptive gate, WAV writing
  live.py               prototype: transcribe each clip as it is recorded
  file_transcriber.py   WAV directory -> transcript, filename parsing
  transcriber.py        faster-whisper wrapper, hallucination filters, and
                        build_transcriber(): settings dict -> Transcriber
  writer.py             transcript files, and clearing them before a re-run
  meeting.py            diarization, speaker-accurate line splitting
  combine.py            merge a diarized loopback with other transcripts
  timeline.py           read-only inputs, compact audio, per-part clock mapping
  processing.py         staged publication, multipart jobs, content-keyed caches
  speakers.py           per-speaker audio samples for identification
  cli.py                dashboard, list-devices, levels, live — nothing else
  server/               dashboard: workspace model, jobs, FastAPI
  server/live.py        live sessions for the dashboard: clip discovery,
                        watermark, one session at a time
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

## Live transcription prototype

`avilistener live` exists to answer one question before anything is built on
top of it: on this machine, how long after someone stops speaking does their
line appear? The answer depends on the model size, the device and how long
people talk for, so it has to be measured rather than estimated. Every printed
line carries its own latency, and the run ends with an average and a worst case.

Recording is unchanged. Clips are still written to disk under the same filename
contract, so the ordinary pipeline can be run over the same session afterwards
and remains the source of truth; live output is a read-only extra.

`live.py` sits between the two. `LiveTranscriber.submit` has exactly the
signature of `SourceRecorder.on_saved` and does nothing but put the path on a
queue; a separate worker thread reads the WAV, transcribes it and calls back
with a `LiveLine`. That decoupling is the whole design: transcription takes
seconds and the recorder thread must never wait for it, because a stalled
recorder loses audio that cannot be recovered. If the model cannot keep up, the
queue grows and lines arrive late while the recording stays intact. A clip that
fails is reported and skipped; it never kills the worker. On Ctrl+C the recorder
threads are joined rather than killed (same reason as above) and the queue is
drained, so the last utterances still appear.

**Diarization is not available live.** It needs the unbroken session audio and a
pass over the whole recording, so live lines are labelled by source (`mic`,
`system`, ...) only. Who said what inside a shared source is still an offline
question.

```powershell
.\.venv\Scripts\avilistener.exe live --model small --device cuda
```

`--model`, `--device` and `--compute-type` override the config for that run
only, which is the point: a smaller model for live latency, the large one kept
in `config.yaml` for the offline transcript. Clips land in
`workspace/live/<timestamp>/` unless `--output` says otherwise.

### Live in the dashboard

The measurement answered yes, so the same machinery is offered from a meeting
page: pick a model, press start, watch lines arrive. `server/live.py` holds it,
and the reason it is a module of its own rather than a few lines inside the
recorder is that live has to work when this process is not the one recording.

**Output goes to `live/`, a sibling of `recordings/`, never inside it.**
`recordings/` is read-only for everything except the recorders. The meeting page
counts clips per source from that folder, and `processing.py` keys its caches on
what it holds, so one extra file in there re-labels the meeting, invalidates
artifacts and offers to re-transcribe work that was already done. A live copy
dropped beside the originals would do exactly that. `live/` holds
`live-transcript.txt`, `state.json` and `clips/`, and nothing downstream reads
it.

**Clips are found by scanning, not pushed by the recorder.** A callback on
`SourceRecorder` was the obvious design and is wrong: the Discord receiver is a
separate Node process writing into the same folder, so half of a Discord meeting
would never reach the model. The filename contract is the one thing every
producer shares, so discovery goes through it - `timestamp_from_discord_wav`
decides what is new, `source_name_from_discord_wav` decides who said it. That
also keeps live off the recorder threads entirely, which matters because a
stalled recorder loses audio that cannot be recovered; live can be started,
stopped and pointed at a different model without any of them noticing. The
found clip is then *copied* into `live/clips/`, so what was actually fed to the
live model is kept whatever later happens to the original.

**A clip is ready when it is a second old and its header agrees with its
size.** Both recorders write a whole WAV in one go once the silence gate closes
it - the write itself measures 1-2 ms here - so there is no window in which a
file sits half-written for long. An mtime at least `MIN_CLIP_AGE` old covers
that window with three orders of magnitude to spare, and the header check
(`nframes > 0`, `st_size >= 44 + nframes * nchannels * sampwidth`) catches the
case where it does not, because a file caught mid-write claims more audio than
it has. Anything that raises while checking means "not ready, try next scan",
never "seen": marking a clip seen on an error silently drops an utterance, and
nothing would ever say so. A lock file or a write-then-rename protocol would be
stronger and would need every producer to cooperate, including the Node
receiver - not worth it for a window this small.

**Start, stop and catch up.** Starting sets a watermark: only clips whose
filename timestamp is at or after it are eligible, so pressing start in the
middle of a meeting does not replay the first hour. Stopping ends the scan, lets
the clip already inside the model finish, and drops the queue *without* marking
those clips seen - so `catch_up` picks them up rather than the user waiting out
a backlog they no longer want. `state.json` carries the watermark, the model and
the seen filenames across both a stop and a server restart, which is what makes
"Resume (catch up)" mean "everything since I first pressed start", and what
makes changing model free: stop, pick another, catch up. A clip joins `seen`
only once the worker is finished with it, line or silence alike. One session
runs at a time for the whole dashboard, because what is being shared is the GPU.

**The fake transcriber.** With `AVILISTENER_LIVE_FAKE_TRANSCRIBER=1` the manager
is built with `FakeLiveTranscriber`, which loads no model and answers
`simulated transcript of <source> clip`. It exists so the Playwright suite (and
a developer poking at the interface) can exercise the whole path without a GPU
or a model download. The variable is read in exactly one place, where the
manager is constructed in `server/app.py`, so the offline pipeline can never
pick it up by accident.

Routes, all under `/api/projects/{p}/meetings/{m}`: `GET live` (status plus
lines after an index), `POST live/start`, `POST live/stop`, and
`GET live/transcript` for the raw file. The meeting payload carries the same
status object without lines, so opening a page already knows. `GET /api/live` is
the global one, like `/api/recording`.

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
