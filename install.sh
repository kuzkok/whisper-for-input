#!/bin/bash
# Install whisper-for-input systemd user services.
# Run once after cloning / moving the project.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Models directory ─────────────────────────────────────────────────────────
mkdir -p "$HOME/.local/share/whisper"
echo "Models dir: $HOME/.local/share/whisper"

# ── Quadlet container unit ────────────────────────────────────────────────────
QUADLET_DIR="$HOME/.config/containers/systemd"
mkdir -p "$QUADLET_DIR"
cp "$SCRIPT_DIR/whisper-for-input.container" "$QUADLET_DIR/"
echo "Installed: $QUADLET_DIR/whisper-for-input.container"

# ── voice-input script → ~/.local/bin ────────────────────────────────────────
BIN_DIR="$HOME/.local/bin"
mkdir -p "$BIN_DIR"
cp "$SCRIPT_DIR/voice-input.py" "$BIN_DIR/voice-input"
chmod +x "$BIN_DIR/voice-input"
echo "Installed: $BIN_DIR/voice-input"

# ── voice-input service ───────────────────────────────────────────────────────
SERVICE_DIR="$HOME/.config/systemd/user"
mkdir -p "$SERVICE_DIR"
cp "$SCRIPT_DIR/voice-input.service" "$SERVICE_DIR/"
echo "Installed: $SERVICE_DIR/voice-input.service"

# ── Reload daemon ─────────────────────────────────────────────────────────────
systemctl --user daemon-reload
echo "Daemon reloaded."

# ── Build the image if not present ───────────────────────────────────────────
if ! podman image exists localhost/whisper-for-input:latest; then
    echo "Building whisper-for-input image (this will take a few minutes)..."
    podman build -t whisper-for-input:latest "$SCRIPT_DIR"
else
    echo "Image localhost/whisper-for-input:latest already exists, skipping build."
    echo "To rebuild: podman build -t whisper-for-input:latest $SCRIPT_DIR"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
cat <<'EOF'

Done. Next steps:

  # Start (and enable on login):
  systemctl --user enable --now whisper-for-input
  systemctl --user enable --now voice-input

  # Check status:
  systemctl --user status whisper-for-input
  systemctl --user status voice-input

  # Live logs:
  journalctl --user -fu whisper-for-input
  journalctl --user -fu voice-input

  # One-time: add yourself to the input group, then re-login:
  sudo usermod -a -G input $USER

  # If WAYLAND_DISPLAY is not wayland-0, override:
  systemctl --user edit voice-input
  # Add:
  #   [Service]
  #   Environment=WAYLAND_DISPLAY=wayland-1
EOF
