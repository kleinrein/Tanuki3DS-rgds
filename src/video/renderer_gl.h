#ifndef RENDERER_GL_H
#define RENDERER_GL_H

#include <glad/glad.h>

#include "common.h"

#define MAX_PROGRAM 1024

typedef struct _GPU GPU;

typedef struct _ProgCacheEntry {
    union {
        struct {
            GLuint vs;
            GLuint fs;
        };
        u64 key;
    };
    GLuint prog;

    struct _ProgCacheEntry *next, *prev;
} ProgCacheEntry;

typedef struct {
    GLuint main_vao;
    GLuint main_vbo;
    GLuint main_program;

    GLuint gpu_vao_sw;
    GLuint gpu_vao_hw;
    GLuint gpu_vbos[12];
    GLuint gpu_ebo;

    GLuint gpu_vs;
    GLuint gpu_uberfs;

    LRUCache(ProgCacheEntry, MAX_PROGRAM) progcache;

    GLuint screentex[2];
    GLuint screenfbo[2];

    GLuint swrendertex;
    GLuint swrenderfbo;

    GLuint blanktex;

    GLuint pbo;

    GLuint lightluttex;
    GLuint fogluttex;
    GLuint proctexmaptex;
    GLuint proctexnoisetex;

    union {
        GLuint ubos[4];
        struct {
            GLuint vert_ubo;
            GLuint uber_ubo;
            GLuint frag_ubo;
            GLuint freecam_ubo;
        };
    };

} GLState;

extern bool g_wireframe;

void renderer_gl_init(GLState* state, GPU* gpu);
void renderer_gl_destroy(GLState* state, GPU* gpu);

void gpu_gl_start_frame(GPU* gpu);
void render_gl_main(GLState* state);
void renderer_gl_set_filter(GLState* state, int filter);
extern bool (*gpu_gl_screen_pass)(int screen);
extern bool (*gpu_gl_panel_ground)();
void renderer_gl_update_freecam(GLState* state);

void gpu_gl_display_transfer(GPU* gpu, u32 paddr, int yoff, bool scalex,
                          bool scaley, bool vflip, int screenid);
void gpu_gl_render_lcd_fb(GPU* gpu, u32 paddr, u32 fmt, int screenid);
void gpu_gl_texture_copy(GPU* gpu, u32 srcpaddr, u32 dstpaddr, u32 size,
                      u32 srcpitch, u32 srcgap, u32 dstpitch, u32 dstgap);
void gpu_gl_clear_fb(GPU* gpu, u32 paddr, u32 len, u32 value, u32 datasz);

// spike: where the main thread put one draw's guest vertex and index data
typedef struct {
    u32 vbo[12];
    u32 index;
    u32 minind, maxind;
    u32 vhash, ihash;
} DrawSnap;

u32 gpu_gl_vstream_put(const void* data, size_t size, size_t align);
void gpu_gl_frame_fence();
extern u32 g_vstream_presents;
extern volatile int g_draw_dump;
extern int g_lut_misses, g_tex_creates, g_tex_partial, g_draws, g_links, g_tex_same, g_tex_changed, g_tex_again;
extern u64 g_ns_shaders, g_ns_compile, g_ns_textures, g_ns_progwait;
extern int g_tex_diffs, g_tex_difftiles, g_tex_new, g_tex_stalefull, g_tex_rehashfull;
extern u64 g_ns_texdiff;
void gpu_gl_shadows_drop(GPU* gpu);
extern int g_prog_ready, g_prog_total;
void gpu_gl_draw(GPU* gpu, bool elements, bool immediate, DrawSnap* snap);

#endif
