#!/bin/sh
# Installs the pinned Booth kath (Linux build) for Kernelcore's WSL fallback mode.
# Destination: ~/.local/share/kernelcore/booth-<version>/kath  (user-local, nothing system-wide)
#
#   sh scripts/install_kath_wsl.sh                # download the release tarball, verify SHA-256
#   sh scripts/install_kath_wsl.sh --from-source  # build the pinned commit with make instead
#
# Run inside WSL, e.g. from Windows:  wsl.exe -e sh scripts/install_kath_wsl.sh
set -eu

VERSION=0.6.1
COMMIT=fc128cb50234a9975858ad17c056e30ca4111441
TGZ_SHA256=526dbccc3d0e05a110a0555ea6cf2d5cef54fd22c68bdfb0ef03d606e3c24fa9
URL="https://github.com/Zaneham/Booth/releases/download/v${VERSION}/booth-${VERSION}-linux-x86_64.tar.gz"
DEST="$HOME/.local/share/kernelcore/booth-${VERSION}"

tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT

fetch() {
    if command -v curl >/dev/null 2>&1; then curl -fsSL -o "$2" "$1"
    elif command -v wget >/dev/null 2>&1; then wget -q -O "$2" "$1"
    else echo "error: need curl or wget" >&2; exit 1
    fi
}

if [ "${1:-}" = "--from-source" ]; then
    command -v git >/dev/null 2>&1 || { echo "error: git not found" >&2; exit 1; }
    command -v make >/dev/null 2>&1 || { echo "error: make not found" >&2; exit 1; }
    git clone --quiet https://github.com/Zaneham/Booth.git "$tmp/src"
    git -C "$tmp/src" checkout --quiet "$COMMIT"
    make -C "$tmp/src" >/dev/null
    mkdir -p "$DEST"
    install -m 0755 "$tmp/src/kath" "$DEST/kath"
    [ -d "$tmp/src/lang" ] && cp -R "$tmp/src/lang" "$DEST/"
else
    fetch "$URL" "$tmp/booth.tgz"
    echo "${TGZ_SHA256}  $tmp/booth.tgz" | sha256sum -c - >/dev/null || {
        echo "error: SHA-256 mismatch for $URL" >&2; exit 1; }
    mkdir -p "$tmp/x"
    tar -xzf "$tmp/booth.tgz" -C "$tmp/x"
    kath=$(find "$tmp/x" -type f -name kath | head -n 1)
    [ -n "$kath" ] || { echo "error: kath not found in archive" >&2; exit 1; }
    mkdir -p "$DEST"
    install -m 0755 "$kath" "$DEST/kath"
    langdir="$(dirname "$kath")/lang"
    [ -d "$langdir" ] && cp -R "$langdir" "$DEST/"
fi

"$DEST/kath" --version | grep -q "Booth ${VERSION}" || { echo "error: installed kath is not Booth ${VERSION}" >&2; exit 1; }
echo "installed $DEST/kath ($("$DEST/kath" --version | head -n 1))"
