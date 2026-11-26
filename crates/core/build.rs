fn main() {
    println!("cargo:rerun-if-changed=src/sme.c");
    if std::env::var_os("CARGO_FEATURE_SME").is_some()
        && std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos")
        && std::env::var("CARGO_CFG_TARGET_ARCH").as_deref() == Ok("aarch64")
    {
        let mut build = cc::Build::new();
        build.file("src/sme.c").compile("shivai_sme");
        // Rust suppresses Clang's runtime libraries, including SME's ZA save helper.
        let runtime = build
            .get_compiler()
            .to_command()
            .arg("--print-runtime-dir")
            .output();
        // Not Clang (e.g. GCC): sme.c compiles to its stub, which needs no runtime.
        let Some(runtime) = runtime.ok().filter(|r| r.status.success()) else {
            println!("cargo:warning=no Clang runtime directory; SME disabled");
            return;
        };
        println!(
            "cargo:rustc-link-search=native={}",
            String::from_utf8(runtime.stdout).unwrap().trim()
        );
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }
}
