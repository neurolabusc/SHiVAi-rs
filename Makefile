# SHiVAi-rs: `make help` lists targets.
VERSION ?= $(shell sed -n 's/^version = "\(.*\)"/\1/p' Cargo.toml | head -1)
MACOS_SIGN_IDENTITY ?= $(shell security find-identity -v -p codesigning 2>/dev/null | sed -n 's/.*"\(Developer ID Application: [^"]*\)".*/\1/p' | head -1)
MACOS_INSTALLER_IDENTITY ?= $(shell security find-identity -v 2>/dev/null | sed -n 's/.*"\(Developer ID Installer: [^"]*\)".*/\1/p' | head -1)
# The notarytool Keychain profile (xcrun notarytool store-credentials NAME); this Mac's is synthsr-notary.
NOTARY_PROFILE ?= synthsr-notary
PKG = dist/shivai-rs-$(VERSION)-macos-arm64.pkg
export MACOSX_DEPLOYMENT_TARGET = 13.0
# Release binaries otherwise carry this machine's home path in panic locations.
export RUSTFLAGS ?= --remap-path-prefix=$(HOME)=~

.PHONY: help build test lint web smoke smoke-gpu niimath models pkg-adhoc pkg notarize release clean

help:
	@echo "build       release build of the shivai-rs executable (target/release/shivai-rs)"
	@echo "test        Rust unit and conformance tests (CPU, Metal), resampling vs scipy"
	@echo "lint        cargo fmt --check, cargo clippy -D warnings"
	@echo "web         build the web app (site/)"
	@echo "smoke       web app in Chromium (SwiftShader): boot + WebGPU conformance graph"
	@echo "smoke-gpu   the same on this GPU, plus the T1 + FLAIR example against SHiVAi"
	@echo "pkg-adhoc   unsigned macOS installer for local testing"
	@echo "pkg         signed installer (MACOS_SIGN_IDENTITY, MACOS_INSTALLER_IDENTITY)"
	@echo "notarize    notarize + staple $(PKG) (NOTARY_PROFILE=$(NOTARY_PROFILE))"
	@echo "release     pkg + notarize"

build:
	cargo build --release -p shivai-rs

test:
	cargo test --release --workspace --exclude shivai-wasm

lint:
	cargo fmt --all --check
	cargo clippy --release --workspace --all-targets -- -D warnings

web:
	npm ci && npm run build

smoke: web
	node test/smoke.mjs

smoke-gpu: web
	SHIVAI_HARDWARE_GPU=1 node test/smoke.mjs

niimath:
	packaging/build_niimath.sh

models:
	packaging/fetch_models.sh target/models

pkg-adhoc: build niimath models
	MACOS_SIGN_IDENTITY=- MACOS_INSTALLER_IDENTITY= packaging/package_macos.sh $(VERSION)

pkg: build niimath models
	MACOS_SIGN_IDENTITY="$(MACOS_SIGN_IDENTITY)" MACOS_INSTALLER_IDENTITY="$(MACOS_INSTALLER_IDENTITY)" packaging/package_macos.sh $(VERSION)

notarize:
	packaging/notarize_macos.sh $(PKG) $(NOTARY_PROFILE)

release: pkg notarize

clean:
	cargo clean
	rm -rf site src/wasm
