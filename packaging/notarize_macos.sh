#!/bin/sh
# Notarize, staple and validate a signed package.
#   packaging/notarize_macos.sh FILE.pkg [KEYCHAIN_PROFILE]   (profile from `xcrun notarytool store-credentials`)
set -eu
pkg=${1:?usage: notarize_macos.sh FILE.pkg [KEYCHAIN_PROFILE]}
profile=${2:-${NOTARY_PROFILE:-synthsr-notary}}
pkgutil --check-signature "$pkg" | grep -q "signed by a developer certificate" || { echo "$pkg is not signed with a Developer ID Installer identity" >&2; exit 2; }
xcrun notarytool submit "$pkg" --keychain-profile "$profile" --wait
for attempt in 1 2 3 4; do xcrun stapler staple "$pkg" && break; sleep $((attempt * 20)); done
xcrun stapler validate "$pkg"
spctl --assess --type install --verbose=2 "$pkg"
# Stapling changes the file: checksum the final package.
(cd "$(dirname "$pkg")" && shasum -a 256 "$(basename "$pkg")" > "$(basename "$pkg").sha256")
