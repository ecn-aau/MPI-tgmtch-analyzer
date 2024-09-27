fn main() {
    println!("cargo:rerun-if-changed=src/og_hash.c");
    cc::Build::new().file("src/og_hash.c").compile("hash");
}
