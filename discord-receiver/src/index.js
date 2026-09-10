import fs from "node:fs";
import path from "node:path";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";

import {
  EndBehaviorType,
  VoiceConnectionStatus,
  entersState,
  getVoiceConnection,
  joinVoiceChannel
} from "@discordjs/voice";
import { Client, GatewayIntentBits } from "discord.js";
import OpusScript from "opusscript";
import YAML from "yaml";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const projectRoot = path.resolve(__dirname, "..", "..");

const configPath = process.argv[2] || path.join(projectRoot, "config.discord-node.yaml");
const config = readConfig(configPath);
const tokenEnv = config.token_env || "DISCORD_BOT_TOKEN";
const token = process.env[tokenEnv] || readWindowsUserEnv(tokenEnv);

if (!token) {
  console.error(`Set your bot token in $${tokenEnv} first.`);
  process.exit(1);
}

const outputDir = path.resolve(projectRoot, config.output_dir || "discord-user-audio");
fs.mkdirSync(outputDir, { recursive: true });
const activeUserStreams = new Map();
const speakingUsers = new Set();
const audioSettings = {
  sampleRate: 48000,
  channels: 2,
  bytesPerSample: 2,
  silenceRmsThreshold: Number(config.silence_rms_threshold || 0.006),
  silenceDurationMs: Number(config.silence_duration_ms || config.after_silence_ms || 1500),
  prerollMs: Number(config.preroll_ms || 300),
  minSegmentMs: Number(config.min_segment_ms || 700)
};

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.GuildVoiceStates,
    GatewayIntentBits.MessageContent
  ]
});

client.once("ready", () => {
  console.log(`Discord receiver logged in as ${client.user.tag}`);
  console.log(`Audio output: ${outputDir}`);
  if (config.auto_join_voice_channel_id) {
    autoJoin(String(config.auto_join_voice_channel_id)).catch((error) => {
      console.error("Auto-join failed:", error);
    });
  }
});

client.on("messageCreate", async (message) => {
  if (message.author.bot || !message.content.startsWith(config.command_prefix || "!")) {
    return;
  }

  const [command] = message.content.slice((config.command_prefix || "!").length).trim().split(/\s+/);

  if (command === "ping") {
    await message.reply("AviListener Node receiver is online.");
    return;
  }

  if (command === "listen") {
    const channel = message.member?.voice?.channel;
    if (!channel) {
      await message.reply("Join a voice channel first, then run `!listen`.");
      return;
    }

    try {
      await startListening(channel, message);
      await message.reply(`Listening in **${channel.name}**. Writing per-user audio files.`);
    } catch (error) {
      console.error(error);
      await message.reply(`Could not listen: \`${error.message}\``);
    }
    return;
  }

  if (command === "stop") {
    const connection = getVoiceConnection(message.guild.id);
    if (connection) {
      connection.destroy();
      await message.reply("Stopped listening.");
    } else {
      await message.reply("I am not connected to voice.");
    }
    return;
  }

  if (command === "where") {
    await message.reply(`Writing per-user audio to \`${outputDir}\``);
    return;
  }

  if (command === "status") {
    await message.reply(buildStatus(message));
  }
});

async function startListening(channel, message) {
  const existing = getVoiceConnection(channel.guild.id);
  if (existing) {
    existing.destroy();
  }

  const connection = joinVoiceChannel({
    channelId: channel.id,
    guildId: channel.guild.id,
    adapterCreator: channel.guild.voiceAdapterCreator,
    selfDeaf: false,
    selfMute: false
  });

  connection.on("error", (error) => console.error("Voice connection error", error));
  await entersState(connection, VoiceConnectionStatus.Ready, 30_000);

  const receiver = connection.receiver;
  receiver.speaking.on("start", (userId) => {
    speakingUsers.add(userId);
  });
  receiver.speaking.on("end", (userId) => {
    speakingUsers.delete(userId);
    console.log(`speaker end ${labelForUserId(userId)}`);
  });
  receiver.speaking.on("start", (userId) => {
    startUserStream(receiver, userId, "speaking");
  });

  for (const member of channel.members.values()) {
    if (!member.user.bot) {
      startUserStream(receiver, member.id, "initial");
    }
  }

  if (config.auto_stop_after_seconds) {
    setTimeout(() => {
      const active = getVoiceConnection(channel.guild.id);
      if (active) {
        active.destroy();
        console.log("auto stopped voice connection");
      }
    }, Number(config.auto_stop_after_seconds) * 1000);
  }
}

function startUserStream(receiver, userId, reason) {
  if (activeUserStreams.has(userId)) {
    return;
  }

  const name = safeName(labelForUserId(userId));
  const streamId = `${Date.now()}-${userId}`;
  console.log(`speaker stream ${reason} ${name} (${userId})`);

  const opusStream = receiver.subscribe(userId, {
    end: {
      behavior: EndBehaviorType.AfterInactivity,
      duration: Number(config.after_inactivity_ms || 300000)
    }
  });
  opusStream.setMaxListeners(50);
  activeUserStreams.set(userId, opusStream);

  const decoder = new OpusScript(audioSettings.sampleRate, audioSettings.channels, OpusScript.Application.AUDIO);
  const segmenter = createSilenceSegmenter(name, userId, streamId);
  let badPackets = 0;
  let finished = false;

  opusStream.on("data", (packet) => {
    try {
      const pcm = decoder.decode(packet);
      if (pcm && pcm.length) {
        segmenter.push(Buffer.from(pcm));
      }
    } catch (error) {
      badPackets += 1;
      if (badPackets <= 3 || badPackets % 100 === 0) {
        console.error(`bad opus packet for ${name}: ${error.message}${badPackets > 3 ? ` (${badPackets} total)` : ""}`);
      }
    }
  });

  opusStream.once("error", (error) => {
    console.error(`audio stream error for ${name}:`, error.message);
    finalizeUserStream();
  });
  opusStream.once("end", finalizeUserStream);
  opusStream.once("close", finalizeUserStream);

  function finalizeUserStream() {
    if (finished) {
      return;
    }
    finished = true;
    activeUserStreams.delete(userId);
    segmenter.finish();
  }
}

function createSilenceSegmenter(name, userId, streamId) {
  const preRollBuffers = [];
  const maxPreRollBytes = Math.round(
    audioSettings.sampleRate *
      audioSettings.channels *
      audioSettings.bytesPerSample *
      (audioSettings.prerollMs / 1000)
  );
  let preRollBytes = 0;
  let segment = null;

  return {
    push(pcm) {
      const now = Date.now();
      const rms = pcmRms(pcm);
      const isVoice = rms >= audioSettings.silenceRmsThreshold;

      if (isVoice) {
        if (!segment) {
          segment = openSegment(name, userId, streamId);
          for (const buffer of preRollBuffers) {
            segment.stream.write(buffer);
            segment.bytes += buffer.length;
          }
          console.log(`speaker audio ${name} (${userId}) rms=${rms.toFixed(4)}`);
        }
        segment.lastVoiceAt = now;
      }

      if (segment) {
        segment.stream.write(pcm);
        segment.bytes += pcm.length;
        if (!isVoice && now - segment.lastVoiceAt >= audioSettings.silenceDurationMs) {
          closeSegment(segment, name, userId);
          segment = null;
          rememberPreRoll(pcm);
        }
      } else {
        rememberPreRoll(pcm);
      }
    },
    finish() {
      if (segment) {
        closeSegment(segment, name, userId);
        segment = null;
      }
    }
  };

  function rememberPreRoll(pcm) {
    preRollBuffers.push(pcm);
    preRollBytes += pcm.length;
    while (preRollBytes > maxPreRollBytes && preRollBuffers.length > 1) {
      const removed = preRollBuffers.shift();
      preRollBytes -= removed.length;
    }
  }
}

function openSegment(name, userId, streamId) {
  const started = timestamp();
  const rawPath = path.join(outputDir, `${streamId}-${started}.pcm`);
  return {
    started,
    rawPath,
    stream: fs.createWriteStream(rawPath),
    bytes: 0,
    lastVoiceAt: Date.now()
  };
}

function closeSegment(segment, name, userId) {
  segment.stream.end(() => {
    if (!fs.existsSync(segment.rawPath) || fs.statSync(segment.rawPath).size === 0) {
      cleanupPartial(segment.rawPath);
      return;
    }

    const durationMs = pcmDurationMs(segment.bytes);
    if (durationMs < audioSettings.minSegmentMs) {
      cleanupPartial(segment.rawPath);
      console.log(`speaker dropped short segment ${name} (${Math.round(durationMs)}ms)`);
      return;
    }

    const wavPath = path.join(outputDir, `${segment.started}-${name}-${userId}.wav`);
    writeWavFromPcm(segment.rawPath, wavPath, audioSettings.sampleRate, audioSettings.channels);
    fs.unlinkSync(segment.rawPath);
    console.log(`speaker saved ${wavPath} (${(durationMs / 1000).toFixed(1)}s)`);
  });
}

function pcmRms(pcm) {
  if (!pcm.length) {
    return 0;
  }

  let sumSquares = 0;
  const samples = Math.floor(pcm.length / 2);
  for (let offset = 0; offset + 1 < pcm.length; offset += 2) {
    const value = pcm.readInt16LE(offset) / 32768;
    sumSquares += value * value;
  }
  return Math.sqrt(sumSquares / samples);
}

function pcmDurationMs(bytes) {
  const bytesPerSecond = audioSettings.sampleRate * audioSettings.channels * audioSettings.bytesPerSample;
  return (bytes / bytesPerSecond) * 1000;
}

function buildStatus(message) {
  const channel = message.member?.voice?.channel;
  const connection = getVoiceConnection(message.guild.id);
  const lines = [];
  lines.push(`connected: ${connection ? "yes" : "no"}`);
  lines.push(`your voice channel: ${channel ? `${channel.name} (${channel.id})` : "none"}`);
  if (channel) {
    lines.push("channel members:");
    for (const member of channel.members.values()) {
      const states = [];
      if (member.id === client.user.id) states.push("bot");
      if (member.voice.selfMute) states.push("self-muted");
      if (member.voice.serverMute) states.push("server-muted");
      if (member.voice.selfDeaf) states.push("self-deaf");
      if (member.voice.serverDeaf) states.push("server-deaf");
      if (member.voice.streaming) states.push("streaming");
      if (member.voice.suppress) states.push("suppressed");
      lines.push(`- ${member.displayName} (${member.id}) ${states.length ? `[${states.join(", ")}]` : ""}`);
    }
  }
  lines.push(`active receive streams: ${[...activeUserStreams.keys()].map(labelForUserId).join(", ") || "none"}`);
  lines.push(`currently speaking: ${[...speakingUsers].map(labelForUserId).join(", ") || "none"}`);
  lines.push(
    `audio split: rms>=${audioSettings.silenceRmsThreshold}, silence=${audioSettings.silenceDurationMs}ms, preroll=${audioSettings.prerollMs}ms`
  );
  return `\`\`\`\n${lines.join("\n").slice(0, 1800)}\n\`\`\``;
}

function labelForUserId(userId) {
  const user = client.users.cache.get(userId);
  return user?.globalName || user?.username || userId;
}

function cleanupPartial(filePath) {
  try {
    if (fs.existsSync(filePath)) {
      fs.unlinkSync(filePath);
    }
  } catch {
    // Best-effort cleanup only.
  }
}

async function autoJoin(channelId) {
  const channel = await client.channels.fetch(channelId);
  await startListening(channel, null);
  console.log(`Auto listening in ${channel.name}`);
}

function readConfig(filePath) {
  if (!fs.existsSync(filePath)) {
    return {};
  }
  return YAML.parse(fs.readFileSync(filePath, "utf8")) || {};
}

function readWindowsUserEnv(name) {
  if (process.platform !== "win32") {
    return "";
  }
  try {
    const output = execFileSync("reg.exe", ["query", "HKCU\\Environment", "/v", name], {
      encoding: "utf8",
      windowsHide: true
    });
    const line = output.split(/\r?\n/).find((item) => item.includes(name));
    if (!line) {
      return "";
    }
    const parts = line.trim().split(/\s{2,}/);
    return parts.at(-1) || "";
  } catch {
    return "";
  }
}

function safeName(value) {
  return value.replace(/[<>:"/\\|?*\x00-\x1f]+/g, "_").replace(/\s+/g, " ").trim().slice(0, 80) || "unknown";
}

function timestamp() {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function writeWavFromPcm(rawPath, wavPath, sampleRate, channels) {
  const pcm = fs.readFileSync(rawPath);
  const header = Buffer.alloc(44);
  const byteRate = sampleRate * channels * 2;
  const blockAlign = channels * 2;

  header.write("RIFF", 0);
  header.writeUInt32LE(36 + pcm.length, 4);
  header.write("WAVE", 8);
  header.write("fmt ", 12);
  header.writeUInt32LE(16, 16);
  header.writeUInt16LE(1, 20);
  header.writeUInt16LE(channels, 22);
  header.writeUInt32LE(sampleRate, 24);
  header.writeUInt32LE(byteRate, 28);
  header.writeUInt16LE(blockAlign, 32);
  header.writeUInt16LE(16, 34);
  header.write("data", 36);
  header.writeUInt32LE(pcm.length, 40);

  fs.writeFileSync(wavPath, Buffer.concat([header, pcm]));
}

client.login(token);
