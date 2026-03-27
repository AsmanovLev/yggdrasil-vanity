/*
 * entry.cl — GPU-side keygen + scoring + candidate filtering
 *
 * Each work-item generates one ed25519 keypair, computes the Yggdrasil
 * address score, and — if the score beats the threshold — appends a
 * candidate record to the output buffer using an atomic counter.
 *
 * Output layout (candidates buffer):
 *   [0..3]   uint  — number of candidates written (atomic counter)
 *   [4..N]   candidate records, each CANDIDATE_STRIDE bytes:
 *               [0..31]  seed    (32 bytes)
 *               [32..63] pubkey  (32 bytes)
 *               [64..71] score   (int64, little-endian, fixed-point *1000000)
 *
 * Inputs:
 *   seeds    — flat array: thread * 32 .. thread * 32 + 31
 *   threshold_buf — int64 LE: current best_score - capture_range (fixed *1e6)
 *   mode       — uint: selects scoring preset
 *   weight_h / weight_l  — float: preset weights passed as kernel args
 */

#define MAX_CANDIDATES  65536
#define CANDIDATE_STRIDE 72   /* 32 seed + 32 pubkey + 8 score */
#define CANDIDATES_HEADER 4   /* uint counter at start */

/* ------------------------------------------------------------------ */
/* leading-zero count of pubkey (Yggdrasil "height")                  */
/* ------------------------------------------------------------------ */
static uchar leading_zeros_pubkey(__private uchar *pk) {
    uchar zeros = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) {
        uchar b = pk[i];
        if (b == 0) {
            zeros += 8;
        } else {
            /* clz on uchar via shift */
            uchar z = 0;
            if ((b & 0xF0) == 0) { z += 4; b <<= 4; }
            if ((b & 0xC0) == 0) { z += 2; b <<= 2; }
            if ((b & 0x80) == 0) { z += 1; }
            zeros += z;
            break;
        }
    }
    return zeros;
}

/* ------------------------------------------------------------------ */
/* Yggdrasil subnet string length estimation                           */
/*                                                                     */
/* Exact length calculator, accurately mirroring std::net::Ipv6Addr   */
/* ------------------------------------------------------------------ */
static uchar subnet_length_approx(uchar height, __private uchar *pk) {
    uchar buf[16];
    buf[0] = 0x02 | 0x01;   /* subnet: first byte has bit 0 set */
    buf[1] = height;
    /* lower 8 bytes zeroed for /64 subnet */
    for (int i = 8; i < 16; i++) buf[i] = 0;

    /* fill buf[2..7] with the same XOR shift as Rust address_for_pubkey */
    int shift = (height + 1) % 8;
    int byte_off = height / 8;
    #pragma unroll
    for (int j = 0; j < 6; j++) {
        int src = byte_off + j;
        if (src + 1 < 32) {
            buf[2 + j] = (uchar)(
                (pk[src]   << shift) ^
                (shift == 0 ? 0 : (pk[src+1] >> (8 - shift))) ^
                0xFF
            );
        } else {
            buf[2 + j] = 0xFF;
        }
    }
    buf[8] = 0; /* /64 subnet lower half zeroed */

    int lnz = 0;
    #pragma unroll
    for (int g = 3; g >= 0; g--) {
        ushort grp = ((uint)buf[g*2] << 8) | (uint)buf[g*2+1];
        if (grp != 0 && lnz == 0) {
            lnz = g;
        }
    }

    int len = 0;
    #pragma unroll
    for (int g = 0; g < 4; g++) {
        if (g <= lnz) {
            ushort grp = ((uint)buf[g*2] << 8) | (uint)buf[g*2+1];
            if (grp == 0) len += 1;
            else if (grp < 0x10) len += 1;
            else if (grp < 0x100) len += 2;
            else if (grp < 0x1000) len += 3;
            else len += 4;
        }
    }
    len += lnz; /* colons */
    len += 2;   /* trailing :: */
    return (uchar)len;
}

/* ------------------------------------------------------------------ */
/* Score formula dispatch                                               */
/* mode:                                                                */
/*   0  →  h               (Height)                                   */
/*   1  →  -l              (Length)                                   */
/*   2  →  h*wh - l*wl     (HeightLength)                             */
/* Score is returned as int64 fixed-point * 1000000                   */
/* ------------------------------------------------------------------ */
static long compute_score(uchar h, uchar l, uint mode, float wh, float wl) {
    float hf = (float)h;
    float lf = (float)l;
    float score;
    switch (mode) {
        case 1:  score = -lf; break;
        case 2:  score = hf * wh - lf * wl; break;
        default: score = hf; break;   /* mode == 0 */
    }
    return (long)(score * 1000000.0f);
}

/* ------------------------------------------------------------------ */
/* Main kernel                                                          */
/* ------------------------------------------------------------------ */
__kernel void generate_pubkey(
    __global uchar  *candidates,   /* output: header(4B) + records     */
    __global uchar  *base_seed,    /* input:  base seed 32B            */
    __global uchar  *threshold_buf,/* input:  int64 LE threshold        */
    uint             mode,
    float            weight_h,
    float            weight_l,
    ulong            global_offset,
    uchar            minimal_height
) {
    size_t thread = get_global_id(0);
    ulong nonce = global_offset + thread;

    /* Read threshold (int64 little-endian) */
    long threshold = 0;
    __global uchar *tb = threshold_buf;
    for (int i = 0; i < 8; i++) {
        threshold |= ((long)tb[i]) << (i * 8);
    }

    /* Construct seed: 24 bytes base_seed + 8 bytes nonce */
    uchar key[32];
    #pragma unroll
    for (int i = 0; i < 24; i++) {
        key[i] = base_seed[i];
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        key[24 + i] = (uchar)(nonce >> (i * 8));
    }

    /* Ed25519 keygen */
    uchar hash[64];
    sha512_hash(key, hash);
    hash[0] &= 248;
    hash[31] &= 63;
    hash[31] |= 64;

    bignum256modm a;
    ge25519 ALIGN(16) A;
    expand256_modm(a, hash, 32);
    ge25519_scalarmult_base_niels(&A, a);

    uchar pubkey[32];
    ge25519_pack(pubkey, &A);

    /* Score */
    uchar h = leading_zeros_pubkey(pubkey);
    if (h < minimal_height) return;

    uchar l = subnet_length_approx(h, pubkey);
    long  score = compute_score(h, l, mode, weight_h, weight_l);

    if (score < threshold) return;

    /* Atomically claim a slot in the candidates buffer */
    __global uint *counter = (__global uint *)candidates;
    uint slot = atomic_inc(counter);
    if (slot >= MAX_CANDIDATES) return;

    /* Write candidate: seed[32] + pubkey[32] + score[8] */
    __global uchar *out = candidates + CANDIDATES_HEADER + slot * CANDIDATE_STRIDE;
    #pragma unroll
    for (int i = 0; i < 32; i++) out[i]      = key[i];
    #pragma unroll
    for (int i = 0; i < 32; i++) out[32 + i] = pubkey[i];

    /* score as int64 little-endian */
    long s_val = score;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        out[64 + i] = (uchar)(s_val & 0xFF);
        s_val >>= 8;
    }
}
