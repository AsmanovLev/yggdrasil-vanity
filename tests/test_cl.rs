use ocl::{Platform, ProQue};

#[test]
fn test_length_approx() {
    let src = r#"
        static uchar subnet_length_approx(uchar height, __private uchar *pk) {
            uchar buf[16];
            buf[0] = 0x02 | 0x01;
            buf[1] = height;
            for (int i = 8; i < 16; i++) buf[i] = 0;

            int shift = (height + 1) % 8;
            int byte_off = height / 8;
            for (int j = 0; j < 6; j++) {
                int src = byte_off + j;
                if (src + 1 < 32) {
                    buf[2 + j] = (uchar)(
                        (pk[src]   << shift) ^
                        (pk[src+1] >> (8 - shift)) ^
                        0xFF
                    );
                } else {
                    buf[2 + j] = 0xFF;
                }
            }
            buf[8] = 0;

            int len = 0;
            len += 2;
            for (int g = 0; g < 4; g++) {
                uint16_t grp = ((uint)buf[g*2] << 8) | (uint)buf[g*2+1];
                if (grp == 0) {
                    len += 1;
                } else if (grp < 0x10) {
                    len += 1;
                } else if (grp < 0x100) {
                    len += 2;
                } else if (grp < 0x1000) {
                    len += 3;
                } else {
                    len += 4;
                }
                len += 1;
            }
            return (uchar)len;
        }

        __kernel void test_len(__global uchar *pks, __global uchar *heights, __global uchar *lens) {
            size_t id = get_global_id(0);
            uchar pk[32];
            for(int i=0; i<32; i++) pk[i] = pks[id*32 + i];
            lens[id] = subnet_length_approx(heights[id], pk);
        }
    "#;

    let pro_que = ProQue::builder()
        .src(src)
        .dims(2)
        .build().unwrap();

    let mut pks = vec![0u8; 64];
    for i in 0..64 { pks[i] = (i * 7 % 256) as u8; }
    
    let pks_buf = pro_que.buffer_builder::<u8>().len(64).copy_host_slice(&pks).build().unwrap();
    let heights_buf = pro_que.buffer_builder::<u8>().len(2).copy_host_slice(&[35, 31]).build().unwrap();
    let lens_buf = pro_que.buffer_builder::<u8>().len(2).build().unwrap();

    let kernel = pro_que.kernel_builder("test_len")
        .arg(&pks_buf).arg(&heights_buf).arg(&lens_buf)
        .build().unwrap();

    unsafe { kernel.enq().unwrap(); }

    let mut lens = vec![0u8; 2];
    lens_buf.read(&mut lens).enq().unwrap();
    println!("GPU computed lens: {:?}", lens);
}
