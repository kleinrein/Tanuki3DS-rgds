// The volume bar, fast-forward badge and FPS counter as RG DS Core draws them for
// every engine. The headers in platform/ are copies of src/platform/ in the RG DS repo.
#include <algorithm>
#include <cmath>
#include <cstring>

#include "platform/frame_stats.hpp"
#include "platform/shader_catalog.hpp"
#include "platform/volume_hud.hpp"

extern "C" void rgds_hud_draw(unsigned char* pixels, int volume, bool fast_forward, double fps) {
    std::memset(pixels, 0, 640 * 480 * 4);
    if (volume >= 0) rgds::draw_volume_hud(pixels, volume);
    if (fast_forward) rgds::draw_fast_forward_hud(pixels);
    if (fps >= 0) rgds::draw_fps_hud(pixels, fps, false);
}

extern "C" const char* rgds_shader_file(int id) {
    return id > 0 && id < rgds::shader_count ? rgds::shader_files[id] : nullptr;
}

extern "C" const char* rgds_shader_defines(int id) {
    return rgds::shader_grid_defines(id);
}

extern "C" bool rgds_shader_uses_target(int id) {
    return rgds::shader_uses_target_size(id);
}

// The 3DS lid around its top screen, in the strips the screen leaves on a 4:3 panel. Shapes only,
// as RG DS Core's other bezels (play/src/ui/frame_styles.hpp keeps the same slugs and names).
namespace {
struct Lid {
    const char* slug;
    unsigned top, bottom, glass, rim;
};
constexpr Lid lids[] = {
    {"ctr-dark", 0x15161A, 0x0F1013, 0x0F1013, 0x15161A},
    {"ctr-cosmos", 0x34373D, 0x1E2024, 0x08090B, 0x4A4E56},
    {"ctr-aqua", 0x4FB6D6, 0x2F8FB0, 0x08090B, 0x7FD0E8},
    {"ctr-flame", 0xD0303A, 0xA01C26, 0x08090B, 0xE8606A},
    {"ctr-pink", 0xF6C6D4, 0xE6A4B8, 0x120C0E, 0xFBDDE6},
    {"ctr-white", 0xF2F2EF, 0xD8D8D4, 0xE9E9E6, 0xFFFFFF},
    {"ctr-midnight", 0x4E3A78, 0x322450, 0x08090B, 0x6E58A0},
};

struct Paint {
    unsigned char* px;
    void set(int x, int y, unsigned rgb, float a = 1) {
        if (x < 0 || x >= 640 || y < 0 || y >= 480 || a <= 0) return;
        auto* p = px + (y * 640 + x) * 4;
        for (int i = 0; i < 3; i++) {
            float v = (rgb >> (16 - 8 * i)) & 255;
            p[i] = (unsigned char) (v * a + p[i] * (1 - a) + 0.5f);
        }
        p[3] = 255;
    }
    void box(int x, int y, int w, int h, unsigned rgb, float a = 1) {
        for (int yy = y; yy < y + h; yy++)
            for (int xx = x; xx < x + w; xx++) set(xx, yy, rgb, a);
    }
    void disc(float cx, float cy, float r, unsigned rgb, float a = 1) {
        for (int y = int(cy - r - 1); y <= int(cy + r + 1); y++)
            for (int x = int(cx - r - 1); x <= int(cx + r + 1); x++) {
                float dx = x + 0.5f - cx, dy = y + 0.5f - cy, d = std::sqrt(dx * dx + dy * dy);
                set(x, y, rgb, a * std::clamp(r - d + 0.5f, 0.f, 1.f));
            }
    }
};

unsigned blend(unsigned a, unsigned b, float t) {
    unsigned out = 0;
    for (int shift = 16; shift >= 0; shift -= 8) {
        float x = (a >> shift) & 255, y = (b >> shift) & 255;
        out |= unsigned(x + (y - x) * t + 0.5f) << shift;
    }
    return out;
}
}

extern "C" bool rgds_bezel_known(const char* slug) {
    for (auto& lid : lids)
        if (!std::strcmp(lid.slug, slug)) return true;
    return false;
}

// (sx, sy, sw, sh): where the game's picture lies. A whole panel of opaque pixels, first row at the top.
extern "C" void rgds_bezel_draw(unsigned char* pixels, const char* slug, int sx, int sy, int sw, int sh) {
    const Lid* lid = &lids[0];
    for (auto& l : lids)
        if (!std::strcmp(l.slug, slug)) lid = &l;
    std::memset(pixels, 255, 640 * 480 * 4);
    Paint p{pixels};
    // Clean dark: nothing but a quiet ground and a hairline around the picture
    if (!std::strcmp(lid->slug, "ctr-dark")) {
        for (int y = 0; y < 480; y++) p.box(0, y, 640, 1, blend(lid->top, lid->bottom, y / 479.f));
        p.box(sx - (sw < 640), sy - 1, sw + 2 * (sw < 640), sh + 2, 0x25272C);
        p.box(sx, sy, sw, sh, 0);
        return;
    }
    // the lid, lit from above
    for (int y = 0; y < 480; y++) p.box(0, y, 640, 1, blend(lid->top, lid->bottom, y / 479.f));
    p.box(0, 0, 640, 2, lid->rim);

    // the glass over the screen, as far as the lid's edge and the hinge leave room
    const int edge = 10, hinge = 26;
    int gx = sw >= 640 ? 0 : edge, gw = sw >= 640 ? 640 : 640 - edge * 2;
    int gy = edge, gh = 480 - edge - hinge;
    p.box(gx, gy, gw, gh, lid->glass);
    const bool pale = ((lid->glass >> 16) & 255) > 128;
    p.box(gx, gy, gw, 1, blend(lid->glass, pale ? 0x000000 : 0xFFFFFF, 0.16f));
    // a soft sheen across the glass above the picture
    for (int y = gy + 1; y < sy; y++) {
        float t = float(y - gy) / std::max(1, sy - gy);
        p.box(gx, y, gw, 1, pale ? 0x000000 : 0xFFFFFF, (pale ? 0.05f : 0.07f) * (1 - t));
    }

    const int mid = (gy + sy) / 2;
    // the camera over the screen
    p.disc(320, mid, 8.5f, blend(lid->glass, pale ? 0x808080 : 0x505660, 0.8f));
    p.disc(320, mid, 6.5f, 0x05060A);
    p.disc(320, mid, 3.5f, 0x141C34);
    p.disc(318.2f, mid - 1.8f, 1.3f, 0xB0C0E0, 0.9f);
    // the 3D light beside it
    p.disc(372, mid, 2.5f, pale ? 0xA8A8A8 : 0x2A2C30);
    // speaker holes at both ends
    const unsigned hole = pale ? 0x8A8A88 : 0x34363C;
    for (int side = 0; side < 2; side++)
        for (int row = -1; row <= 1; row++)
            for (int col = 0; col < 7; col++) {
                int x = (side ? 640 - 58 : 30) + col * 4 + ((row & 1) ? 2 : 0);
                p.disc(x + 0.5f, mid + row * 5.0f, 1.2f, hole);
            }

    // a thin frame where the glass meets the picture
    if (sw < 640) p.box(sx - 2, sy - 2, sw + 4, sh + 4, pale ? 0xB4B4B0 : 0x24262A);
    p.box(sx, sy, sw, sh, 0);

    // the hinge along the bottom, its two barrels
    const int hy = 480 - hinge;
    p.box(0, hy, 640, 2, blend(lid->bottom, 0x000000, 0.35f));
    p.box(0, hy + 2, 640, 1, blend(lid->rim, lid->top, 0.3f));
    for (int side = 0; side < 2; side++) {
        int x = side ? 640 - 150 : 40;
        p.box(x, hy + 7, 110, 12, blend(lid->bottom, 0x000000, 0.28f));
        p.box(x, hy + 7, 110, 1, blend(lid->rim, lid->bottom, 0.4f));
    }
}
