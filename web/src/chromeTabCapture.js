const BUFFER_FRAMES = 4096;

function downmix(buffer) {
  const frames = buffer.length;
  const channels = buffer.numberOfChannels;
  const mono = new Float32Array(frames);
  for (let channel = 0; channel < channels; channel += 1) {
    const input = buffer.getChannelData(channel);
    for (let index = 0; index < frames; index += 1) mono[index] += input[index] / channels;
  }
  return mono;
}

/**
 * Ask Chrome for one tab and prepare a local PCM stream.
 *
 * getDisplayMedia intentionally cannot remember or silently repeat the user's
 * choice. Chrome shows its picker for every recording, and the user must pick
 * a tab with "Share tab audio" enabled.
 */
export async function prepareChromeTabCapture({ onChunk, onEnded }) {
  if (!navigator.mediaDevices?.getDisplayMedia) {
    throw new Error("Chrome tab capture is unavailable in this browser. Open the dashboard in Chrome or Edge.");
  }

  const stream = await navigator.mediaDevices.getDisplayMedia({
    video: true,
    audio: true,
    selfBrowserSurface: "exclude",
    surfaceSwitching: "exclude",
    systemAudio: "exclude"
  });
  const audioTrack = stream.getAudioTracks()[0];
  if (!audioTrack) {
    stream.getTracks().forEach((track) => track.stop());
    throw new Error('No tab audio was shared. Pick a Chrome tab and enable "Share tab audio".');
  }

  let context = null;
  let source = null;
  let processor = null;
  let silentOutput = null;
  let pending = [];
  let pendingFrames = 0;
  let uploadError = null;
  let uploadChain = Promise.resolve();
  let stopping = false;
  let started = false;

  const enqueue = (audio, sampleRate) => {
    uploadChain = uploadChain.then(async () => {
      if (uploadError) return;
      try {
        await onChunk(audio, sampleRate);
      } catch (error) {
        uploadError = error;
      }
    });
  };

  const flush = (sampleRate, force = false) => {
    if (!pendingFrames || (!force && pendingFrames < sampleRate / 2)) return;
    const audio = new Float32Array(pendingFrames);
    let offset = 0;
    for (const chunk of pending) {
      audio.set(chunk, offset);
      offset += chunk.length;
    }
    pending = [];
    pendingFrames = 0;
    enqueue(audio, sampleRate);
  };

  const handleEnded = () => {
    if (!stopping) onEnded?.();
  };
  audioTrack.addEventListener("ended", handleEnded);

  return {
    async start() {
      if (started) return;
      started = true;
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      if (!AudioContext) throw new Error("Web Audio is unavailable in this browser.");

      // Chrome normally honours this rate. The server still accepts the
      // context's actual rate, so another Chromium build cannot distort time.
      context = new AudioContext({ sampleRate: 16000 });
      await context.resume();
      source = context.createMediaStreamSource(new MediaStream([audioTrack]));
      processor = context.createScriptProcessor(BUFFER_FRAMES, Math.max(1, source.channelCount || 1), 1);
      silentOutput = context.createGain();
      silentOutput.gain.value = 0;
      processor.onaudioprocess = (event) => {
        const audio = downmix(event.inputBuffer);
        pending.push(audio);
        pendingFrames += audio.length;
        flush(context.sampleRate);
      };
      source.connect(processor);
      processor.connect(silentOutput);
      silentOutput.connect(context.destination);
    },

    async stop() {
      if (stopping) return uploadChain;
      stopping = true;
      audioTrack.removeEventListener("ended", handleEnded);
      if (processor) processor.onaudioprocess = null;
      if (context) flush(context.sampleRate, true);
      processor?.disconnect();
      source?.disconnect();
      silentOutput?.disconnect();
      stream.getTracks().forEach((track) => track.stop());
      await uploadChain;
      await context?.close();
      if (uploadError) throw uploadError;
    }
  };
}
