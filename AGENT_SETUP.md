Set up AviListener from https://github.com/matheusavi/avi-listener on this
Windows machine. Follow README.md and AGENT_SETUP.md in the repository.

1. Clone the repository into a folder I choose (ask me where; default to the
   current directory) and work from its root.
2. Check prerequisites and tell me what is missing before installing anything:
   Python 3.11 from python.org with the `py` launcher, Node.js 22.12 or newer,
   and whether this machine has an NVIDIA GPU with CUDA.
3. Run `.\setup.ps1`. Use `.\setup.ps1 -Cpu` if there is no NVIDIA GPU. Do not
   add `-WithNemo` unless I ask for it. If the script fails, show me the error
   and fix the cause rather than retrying blindly.
4. Run `.\.venv\Scripts\avilistener.exe list-devices` and show me the list.
   Ask me which microphone I speak into and which output I hear calls through.
5. Put those two device names into `config.yaml` (created by setup.ps1 from
   config.example.yaml) under `sources.mic.device` and
   `sources.system.device`.
6. Run `.\.venv\Scripts\avilistener.exe levels --config config.yaml
   --duration 10` while I talk, and tell me whether the microphone clears the
   gate. If it does not, tell me to raise the input volume in Windows sound
   settings before touching thresholds.
7. Start the dashboard with `.\.venv\Scripts\avilistener.exe dashboard` and
   tell me to open http://127.0.0.1:8000.
8. Tell me what to set in the dashboard's Settings panel: the same two
   devices, my transcription language, "Run on" (cuda or cpu, matching step
   3), and a Hugging Face token if I want speaker splitting. Never ask me to
   paste a token into the chat; I will enter it in the dashboard myself.

Do not commit, push, or change anything outside the cloned folder.
