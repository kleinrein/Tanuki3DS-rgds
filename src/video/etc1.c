#include "etc1.h"

const u8 etc1table[8][2] = {
    {2, 8},   {5, 17},  {9, 29},   {13, 42},
    {18, 60}, {24, 80}, {33, 106}, {47, 183},
};

#define CLAMP(x) ((x) < 0 ? 0 : (x) > 255 ? 255 : (x))

void flip_vertical(u32 height, u32 pitch, u8 (*tex)[pitch]) {
    u8 tmp[pitch];
    for (int y = 0; y < height / 2; y++) {
        memcpy(tmp, tex[y], pitch);
        memcpy(tex[y], tex[height - 1 - y], pitch);
        memcpy(tex[height - 1 - y], tmp, pitch);
    }
}

void etc1_decompress_block(u64 block, u32 width, int Bpp,
                           u8 (*dst)[width][Bpp]) {
    etc1block blk = {block};

    u8 r1, r2, g1, g2, b1, b2;
    if (blk.diff) {
        r1 = blk.dr1 * 0x21 / 4;
        r2 = (blk.dr1 + blk.dr2) * 0x21 / 4;
        g1 = blk.dg1 * 0x21 / 4;
        g2 = (blk.dg1 + blk.dg2) * 0x21 / 4;
        b1 = blk.db1 * 0x21 / 4;
        b2 = (blk.db1 + blk.db2) * 0x21 / 4;
    } else {
        r1 = blk.r1 * 0x11;
        r2 = blk.r2 * 0x11;
        g1 = blk.g1 * 0x11;
        g2 = blk.g2 * 0x11;
        b1 = blk.b1 * 0x11;
        b2 = blk.b2 * 0x11;
    }

    for (int i = 0; i < 8; i++) {
        int x, y;
        if (blk.flip) {
            x = i % 4;
            y = i / 4;
        } else {
            x = i / 4;
            y = i % 4;
        }
        int pixel = y + x * 4;
        int mod = etc1table[blk.table1][(blk.modidx >> pixel) & 1];
        if ((blk.modneg >> pixel) & 1) {
            dst[y][x][0] = CLAMP(r1 - mod);
            dst[y][x][1] = CLAMP(g1 - mod);
            dst[y][x][2] = CLAMP(b1 - mod);
        } else {
            dst[y][x][0] = CLAMP(r1 + mod);
            dst[y][x][1] = CLAMP(g1 + mod);
            dst[y][x][2] = CLAMP(b1 + mod);
        }
    }
    for (int i = 0; i < 8; i++) {
        int x, y;
        if (blk.flip) {
            x = i % 4;
            y = i / 4 + 2;
        } else {
            x = i / 4 + 2;
            y = i % 4;
        }
        int pixel = y + x * 4;
        int mod = etc1table[blk.table2][(blk.modidx >> pixel) & 1];
        if ((blk.modneg >> pixel) & 1) {
            dst[y][x][0] = CLAMP(r2 - mod);
            dst[y][x][1] = CLAMP(g2 - mod);
            dst[y][x][2] = CLAMP(b2 - mod);
        } else {
            dst[y][x][0] = CLAMP(r2 + mod);
            dst[y][x][1] = CLAMP(g2 + mod);
            dst[y][x][2] = CLAMP(b2 + mod);
        }
    }
}

void etc1_decompress_texture(u32 width, u32 height, u64 (*src)[width / 8][2][2],
                             u8 (*dst)[width][3]) {

    for (int tx = 0; tx < width / 8; tx++) {
        for (int ty = 0; ty < height / 8; ty++) {
            for (int ttx = 0; ttx < 2; ttx++) {
                for (int tty = 0; tty < 2; tty++) {
                    etc1_decompress_block(
                        src[ty][tx][tty][ttx], width, 3,
                        (void*) &dst[8 * ty + 4 * tty][8 * tx + 4 * ttx]);
                }
            }
        }
    }

    flip_vertical(height, width * 3, (void*) dst);
}

void etc1a4_decompress_texture(u32 width, u32 height,
                               u64 (*src)[width / 8][2][2][2],
                               u8 (*dst)[width][4]) {

    for (int tx = 0; tx < width / 8; tx++) {
        for (int ty = 0; ty < height / 8; ty++) {
            for (int ttx = 0; ttx < 2; ttx++) {
                for (int tty = 0; tty < 2; tty++) {
                    u64 etcblk = src[ty][tx][tty][ttx][1];
                    u64 ablk = src[ty][tx][tty][ttx][0];
                    etc1_decompress_block(
                        etcblk, width, 4,
                        (void*) &dst[8 * ty + 4 * tty][8 * tx + 4 * ttx]);
                    for (int fx = 0; fx < 4; fx++) {
                        for (int fy = 0; fy < 4; fy++) {
                            int pixel = fy + fx * 4;
                            u8 a = ((ablk >> (pixel * 4)) & 0xf) * 0x11;
                            dst[8 * ty + 4 * tty + fy][8 * tx + 4 * ttx + fx]
                               [3] = a;
                        }
                    }
                }
            }
        }
    }

    flip_vertical(height, width * 4, (void*) dst);
}

// A block for the GPU's own ETC decoder: 3DS blocks are the same bits in the
// other byte order. Textures are stored bottom row first here, so each block is
// also flipped top to bottom. A difference of -4 between the top and bottom
// colours has no +4 to flip into; it becomes +3, one step off on those pixels.
u64 etc1_block_flip_y(u64 block, bool* clamped) {
    etc1block b = {block};
    u32 m = b.modidx;
    m = (m & 0x5555) << 1 | (m >> 1 & 0x5555);
    b.modidx = (m & 0x3333) << 2 | (m >> 2 & 0x3333);
    m = b.modneg;
    m = (m & 0x5555) << 1 | (m >> 1 & 0x5555);
    b.modneg = (m & 0x3333) << 2 | (m >> 2 & 0x3333);
    if (b.flip) {
        // the two sub-blocks are above each other: they swap
        int t = b.table1;
        b.table1 = b.table2;
        b.table2 = t;
        if (b.diff) {
#define FLIPDIFF(c)                                                            \
    {                                                                          \
        int d = b.d##c##2;                                                     \
        b.d##c##1 = b.d##c##1 + d;                                             \
        if (d == -4) {                                                         \
            d = -3;                                                            \
            *clamped = true;                                                   \
        }                                                                      \
        b.d##c##2 = -d;                                                        \
    }
            FLIPDIFF(r)
            FLIPDIFF(g)
            FLIPDIFF(b)
#undef FLIPDIFF
        } else {
            t = b.r1, b.r1 = b.r2, b.r2 = t;
            t = b.g1, b.g1 = b.g2, b.g2 = t;
            t = b.b1, b.b1 = b.b2, b.b2 = t;
        }
    }
    return b.d;
}

// false when a block would mean something else to an ETC2 decoder
bool etc1_block_is_plain(u64 block) {
    etc1block b = {block};
    if (!b.diff) return true;
    int r = b.dr1 + b.dr2, g = b.dg1 + b.dg2, bl = b.db1 + b.db2;
    return r >= 0 && r <= 31 && g >= 0 && g <= 31 && bl >= 0 && bl <= 31;
}

// w and h in pixels, multiples of 8. dst holds w * h / 2 bytes. False when the
// texture has to be decoded instead.
bool etc1_to_gpu(u32 width, u32 height, const u64* src, int blockstep, u64* dst,
                 int* clamped) {
    u32 bw = width / 4, bh = height / 4;
    for (u32 ty = 0; ty < height / 8; ty++) {
        for (u32 tx = 0; tx < width / 8; tx++) {
            for (u32 tty = 0; tty < 2; tty++) {
                for (u32 ttx = 0; ttx < 2; ttx++) {
                    u64 blk = *src;
                    src += blockstep;
                    if (!etc1_block_is_plain(blk)) return false;
                    bool c = false;
                    blk = etc1_block_flip_y(blk, &c);
                    *clamped += c;
                    u32 by = bh - 1 - (ty * 2 + tty), bx = tx * 2 + ttx;
                    dst[by * bw + bx] = __builtin_bswap64(blk);
                }
            }
        }
    }
    return true;
}
