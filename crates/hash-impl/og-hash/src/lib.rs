use hash::ByteHash;

const T: [u8; 256] = [
    1, 87, 49, 12, 176, 178, 102, 166, 121, 193, 6, 84, 249, 230, 44, 163, 14, 197, 213, 181, 161,
    85, 218, 80, 64, 239, 24, 226, 236, 142, 38, 200, 110, 177, 104, 103, 141, 253, 255, 50, 77,
    101, 81, 18, 45, 96, 31, 222, 25, 107, 190, 70, 86, 237, 240, 34, 72, 242, 20, 214, 244, 227,
    149, 235, 97, 234, 57, 22, 60, 250, 82, 175, 208, 5, 127, 199, 111, 62, 135, 248, 174, 169,
    211, 58, 66, 154, 106, 195, 245, 171, 17, 187, 182, 179, 0, 243, 132, 56, 148, 75, 128, 133,
    158, 100, 130, 126, 91, 13, 153, 246, 216, 219, 119, 68, 223, 78, 83, 88, 201, 99, 122, 11, 92,
    32, 136, 114, 52, 10, 138, 30, 48, 183, 156, 35, 61, 26, 143, 74, 251, 94, 129, 162, 63, 152,
    170, 7, 115, 167, 241, 206, 3, 150, 55, 59, 151, 220, 90, 53, 23, 131, 125, 173, 15, 238, 79,
    95, 89, 16, 105, 137, 225, 224, 217, 160, 37, 123, 118, 73, 2, 157, 46, 116, 9, 145, 134, 228,
    207, 212, 202, 215, 69, 229, 27, 188, 67, 124, 168, 252, 42, 4, 29, 108, 21, 247, 19, 205, 39,
    203, 233, 40, 186, 147, 198, 192, 155, 33, 164, 191, 98, 204, 165, 180, 117, 76, 140, 36, 210,
    172, 41, 54, 159, 8, 185, 232, 113, 196, 231, 47, 146, 120, 51, 65, 28, 144, 254, 221, 93, 189,
    194, 139, 112, 43, 71, 109, 184, 209,
];

pub struct OgHashU8;

impl ByteHash for OgHashU8 {
    fn hash_u8(buf: &[u8]) -> u8 {
        if buf.len() > u16::MAX as usize {
            panic!("length in original hash cannot exeed u16::MAX");
        }

        let mut length = buf.len() as u16;

        if length == 1 {
            return T[buf[0] as usize];
        }

        let mut len = length >> 1;
        length -= len * 2;

        let mut prev = 0u8;
        let mut next;

        let mut index = 0usize;

        while len > 0 {
            next = T[(prev ^ buf[index]) as usize];
            prev = T[(next ^ buf[index + 1]) as usize];

            index += 2;
            len -= 1;
        }

        if length == 1 {
            T[(prev ^ buf[index]) as usize]
        } else {
            prev
        }
    }
}

#[cfg(test)]
mod tests {
    use hash::ByteHash;
    use rand::{distributions::Standard, Rng};

    use crate::OgHashU8;

    #[link(name = "hash")]
    extern "C" {
        fn dpatm_hash_8(buffer: *const u8, length: u16) -> u8;
    }

    fn og_hash_8(buf: &[u8]) -> u8 {
        if buf.len() > u16::MAX as usize {
            panic!("length in hash FFI cannot exeed u16::MAX");
        }

        unsafe { dpatm_hash_8(buf.as_ptr(), buf.len() as u16) }
    }

    #[test]
    fn simple_cases() {
        let single_element = [0u8];
        assert_eq!(
            og_hash_8(&single_element),
            OgHashU8::hash_u8(&single_element)
        );

        let u8_ordered = (0..255).collect::<Vec<_>>();
        assert_eq!(og_hash_8(&u8_ordered), OgHashU8::hash_u8(&u8_ordered));
    }

    #[test]
    fn max_buffer_size() {
        let max_buffer_size = (0..u8::MAX)
            .cycle()
            .take(u16::MAX as usize)
            .collect::<Vec<_>>();
        assert_eq!(
            og_hash_8(&max_buffer_size),
            OgHashU8::hash_u8(&max_buffer_size)
        );
    }

    #[test]
    fn random_max_buffer() {
        let mut rng = rand::thread_rng();

        let rand_max_buffer = (&mut rng)
            .sample_iter(Standard)
            .take(u16::MAX as usize)
            .collect::<Vec<u8>>();

        assert_eq!(
            og_hash_8(&rand_max_buffer),
            OgHashU8::hash_u8(&rand_max_buffer)
        );
    }

    #[test]
    #[should_panic]
    fn greater_than_max() {
        let max_buffer_size = (0..u8::MAX)
            .cycle()
            .take(u16::MAX as usize + 1)
            .collect::<Vec<_>>();

        OgHashU8::hash_u8(&max_buffer_size);
    }
}
