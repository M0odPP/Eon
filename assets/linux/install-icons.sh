#!/usr/bin/env bash
set -euo pipefail

# User-local Freedesktop (XDG) installer for Eon wing icons, MIME types (.eon / .hal.eon),
# and desktop entry (Wayland app_id `dev.eon.app` + file manager icons). Zero root required.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
XDG_DATA="${XDG_DATA_HOME:-${HOME}/.local/share}"

APP_ICON_DIR="${XDG_DATA}/icons/hicolor/scalable/apps"
MIME_ICON_DIR="${XDG_DATA}/icons/hicolor/scalable/mimetypes"
MIME_PKG_DIR="${XDG_DATA}/mime/packages"
DESKTOP_DIR="${XDG_DATA}/applications"

mkdir -p "${APP_ICON_DIR}" "${MIME_ICON_DIR}" "${MIME_PKG_DIR}" "${DESKTOP_DIR}"

cp "${REPO_ROOT}/assets/brand/eon-app-icon.svg" "${APP_ICON_DIR}/dev.eon.app.svg"
cp "${REPO_ROOT}/assets/brand/eon-app-icon.svg" "${APP_ICON_DIR}/eon-lang.svg"
cp "${REPO_ROOT}/editors/vscode/icons/eon-file-dark.svg" "${MIME_ICON_DIR}/text-x-eon.svg"
cp "${REPO_ROOT}/editors/vscode/icons/eon-hal-dark.svg" "${MIME_ICON_DIR}/text-x-eon-hal.svg"
cp "${SCRIPT_DIR}/eon-mime.xml" "${MIME_PKG_DIR}/eon-lang.xml"
cp "${SCRIPT_DIR}/dev.eon.app.desktop" "${DESKTOP_DIR}/dev.eon.app.desktop"

if command -v update-mime-database >/dev/null 2>&1; then
    update-mime-database "${XDG_DATA}/mime" || true
fi
if command -v gtk-update-icon-cache >/dev/null 2>&1; then
    gtk-update-icon-cache -f -t "${XDG_DATA}/icons/hicolor" 2>/dev/null || true
fi
if command -v update-desktop-database >/dev/null 2>&1; then
    update-desktop-database "${DESKTOP_DIR}" 2>/dev/null || true
fi

echo "Installed Eon wing icons, MIME types (*.eon, *.hal.eon), and dev.eon.app.desktop into ${XDG_DATA}"
