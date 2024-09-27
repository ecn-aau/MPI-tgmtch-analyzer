pub trait ByteHash {
    fn hash_u8(buf: &[u8]) -> u8;
}
