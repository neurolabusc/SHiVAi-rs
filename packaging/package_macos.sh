#!/bin/sh
# macOS installer (Apple silicon, macOS 13+): /usr/local/bin/shivai-rs, /usr/local/bin/niimath and the
# SHiVA models in /usr/local/share/shivai-rs/models, with the licences.
#   packaging/package_macos.sh VERSION
# MACOS_SIGN_IDENTITY ("Developer ID Application: …", default: ad hoc) signs the binaries with the
# hardened runtime; MACOS_INSTALLER_IDENTITY ("Developer ID Installer: …") signs the package.
set -eu
[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "package_macos.sh requires Apple silicon macOS" >&2; exit 2; }
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
version=${1:?usage: package_macos.sh VERSION}
identity=${MACOS_SIGN_IDENTITY:--}
installer_identity=${MACOS_INSTALLER_IDENTITY:-}
suffix=; [ "$identity" = - ] && suffix=-adhoc
pkg="$root/dist/shivai-rs-$version-macos-arm64$suffix.pkg"
shivai="$root/target/release/shivai-rs"; niimath="$root/target/niimath/niimath"; models="$root/target/models"
for bin in "$shivai" "$niimath"; do
	[ -x "$bin" ] || { echo "package_macos.sh: $bin is not built (make pkg builds it)" >&2; exit 1; }
	foreign=$(otool -L "$bin" | tail -n +2 | awk '{ print $1 }' | grep -v '^/usr/lib/' | grep -v '^/System/' || true)
	[ -z "$foreign" ] || { echo "package_macos.sh: $bin has non-system dependencies: $foreign" >&2; exit 1; }
	minos=$(otool -l "$bin" | awk '$1 == "minos" { print $2; exit }')
	[ "$minos" = 13.0 ] || { echo "package_macos.sh: $bin targets macOS $minos, expected 13.0" >&2; exit 1; }
done
payload=$(mktemp -d "${TMPDIR:-/tmp}/shivai-pkg.XXXXXX")
trap 'rm -rf "$payload"' EXIT HUP INT TERM
bin="$payload/root/usr/local/bin"; share="$payload/root/usr/local/share/shivai-rs"
mkdir -p "$bin" "$share/models" "$root/dist"
cp "$shivai" "$niimath" "$bin/"
cp "$models"/*.shiva "$share/models/"
cp "$root/LICENSE" "$root/NOTICE" "$root/models.json" "$share/"
[ -f "${NIIMATH_SRC:-$root/target/niimath-src}/LICENSE" ] && cp "${NIIMATH_SRC:-$root/target/niimath-src}/LICENSE" "$share/LICENSE-niimath"
chmod 755 "$bin"/*; xattr -cr "$payload/root"
for b in "$bin/niimath" "$bin/shivai-rs"; do
	if [ "$identity" = - ]; then codesign --force --sign - "$b"; else codesign --force --options runtime --timestamp --sign "$identity" "$b"; fi
	codesign --verify --strict --verbose=2 "$b"
done
"$bin/shivai-rs" --version
pkgbuild --root "$payload/root" --identifier io.github.neurolabusc.shivai-rs --version "$version" --install-location / "$payload/component.pkg"
if [ -n "$installer_identity" ]; then
	productbuild --package "$payload/component.pkg" --sign "$installer_identity" "$pkg"
else
	echo "package_macos.sh: MACOS_INSTALLER_IDENTITY unset; the package is unsigned (local testing only)" >&2
	productbuild --package "$payload/component.pkg" "$pkg"
fi
shasum -a 256 "$pkg" | sed "s#$root/dist/##" > "$pkg.sha256"
echo "wrote $pkg"
