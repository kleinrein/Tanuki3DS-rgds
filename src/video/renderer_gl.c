#include <SDL3/SDL.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <unistd.h>
#include "renderer_gl.h"

#include "3ds.h"
#include "emulator.h"

#include "etc1.h"
#include "gpu.h"
#include "gpu_hash.h"

#include "gpuptr.inc"

const char mainvertsource[] = {
#embed "hostshaders/main.vert"
    , '\0'};
const char mainfragsource[] = {
#embed "hostshaders/main.frag"
    , '\0'};
const char mainfragsharpsource[] = {
#embed "hostshaders/main_sharp_linear.frag"
    , '\0'};

const char gpuvertsource[] = {
#embed "hostshaders/gpu.vert"
    , '\0'};
const char gpufragsource[] = {
#embed "hostshaders/gpu.frag"
    , '\0'};

bool g_wireframe;

// spike: a ring instead of glBufferData per draw (Panfrost makes a new BO for each of those)
#define STREAM_SIZE (16u << 20)
#define VSTREAM_SIZE (32u << 20)
// g_vstream is written by the emulation thread, read by the gl thread's draws
static GLuint g_vstream;
static u8* g_vstream_ptr;
static size_t g_vstream_off;
static GLuint g_stream;
static size_t g_stream_off;
static GLint g_ubo_align = 256;
static u8* g_stream_ptr;
static GLsync g_frame_fence;
void gpu_gl_frame_fence() {
    if (g_frame_fence) glDeleteSync(g_frame_fence);
    g_frame_fence = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
}
static size_t stream_put(GLenum target, const void* data, size_t size, size_t align) {
    glBindBuffer(target, g_stream);
    size_t off = (g_stream_off + align - 1) & ~(align - 1);
    if (off + size > STREAM_SIZE) {
        fprintf(stderr, "spike ubo ring wrap\n");
        // what is at the start of the ring belongs to old frames, not the open batch
        if (g_frame_fence) {
            glClientWaitSync(g_frame_fence, 0, 1000000000);
        } else {
            glFinish();
        }
        off = 0;
    }
    if (size) memcpy(g_stream_ptr + off, data, size);
    g_stream_off = off + size;
    return off;
}

u32 g_vstream_presents;
volatile int g_draw_dump;
static u32 g_vstream_wrap_at;
u32 gpu_gl_vstream_put(const void* data, size_t size, size_t align) {
    if (size > VSTREAM_SIZE / 2) {
        // only a broken index or a wild pointer asks for this much
        lwarnonce("vertex data of %zu bytes left out", size);
        size = 0;
    }
    size_t off = (g_vstream_off + align - 1) & ~(align - 1);
    if (off + size > VSTREAM_SIZE) {
        // the gpu is never this many frames behind, so only short rings wait
        if (g_vstream_presents - g_vstream_wrap_at < 8) gpu_worker_sync(true);
        g_vstream_wrap_at = g_vstream_presents;
        off = 0;
    }
    if (size && data) memcpy(g_vstream_ptr + off, data, size);
    g_vstream_off = off + size;
    return off;
}
u8* gpu_gl_vstream_ptr() { return g_vstream_ptr; }

// spike: light tables by content
#define LUT_CACHE 32
static struct {
    GLuint tex;
    u32 used;
    u64 hash;
} g_lut[LUT_CACHE];
static GLuint g_lut_cur;
static u32 g_lut_tick;
int g_lut_misses;

// spike: glGetUniformLocation per draw makes glthread sync, so keep the answers per program
static struct {
    GLint loc[3];
    float offset, scale;
    int wbuffer;
    bool known, sent;
} g_depth_uni[8192];

static void prog_builder_start();
static void gpu_gl_etc_start();
static void prog_unbind();

static void main_program_build(GLState* state) {
    auto mainvs = glCreateShader(GL_VERTEX_SHADER);
    auto mainfs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(mainvs, 1, &(const char*) {mainvertsource}, nullptr);
    glShaderSource(mainfs, 1,
                   &(const char*) {ctremu.outputfilter == FILTER_SHARP_LINEAR
                                       ? mainfragsharpsource
                                       : mainfragsource},
                   nullptr);
    glCompileShader(mainvs);
    glCompileShader(mainfs);

    state->main_program = glCreateProgram();
    glAttachShader(state->main_program, mainvs);
    glAttachShader(state->main_program, mainfs);
    glLinkProgram(state->main_program);
    glDeleteShader(mainvs);
    glDeleteShader(mainfs);
}

// how the 3DS screens are scaled onto the panels, while the game runs
void renderer_gl_set_filter(GLState* state, int filter) {
    if (filter == ctremu.outputfilter) return;
    ctremu.outputfilter = filter;
    GLuint old = state->main_program;
    main_program_build(state);
    glDeleteProgram(old);
    glUseProgram(state->main_program);
    glUniform1i(glGetUniformLocation(state->main_program, "screen"), 0);
    GLint f = filter == FILTER_NEAREST ? GL_NEAREST : GL_LINEAR;
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, state->screentex[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, f);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, f);
    }
}

void renderer_gl_init(GLState* state, GPU* gpu) {
    main_program_build(state);
    glUseProgram(state->main_program);
    glUniform1i(glGetUniformLocation(state->main_program, "screen"), 0);

    glGenVertexArrays(1, &state->main_vao);
    glBindVertexArray(state->main_vao);

    glGenBuffers(1, &state->main_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, state->main_vbo);
    glBufferData(GL_ARRAY_BUFFER, 0, nullptr, GL_STATIC_DRAW);

    state->gpu_vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(state->gpu_vs, 1, &(const char*) {gpuvertsource}, nullptr);
    glCompileShader(state->gpu_vs);

    state->gpu_uberfs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(state->gpu_uberfs, 1, &(const char*) {gpufragsource},
                   nullptr);
    glCompileShader(state->gpu_uberfs);

    LRU_init(state->progcache);
    prog_builder_start();
    gpu_gl_etc_start();

    glGenBuffers(countof(state->ubos), state->ubos);
    for (int i = 0; i < countof(state->ubos); i++) {
        glBindBufferBase(GL_UNIFORM_BUFFER, i, state->ubos[i]);
    }
    // freecam buffer contains a matrix and a bool
    glBindBuffer(GL_UNIFORM_BUFFER, state->freecam_ubo);
    glBufferData(GL_UNIFORM_BUFFER, 17 * 4, nullptr, GL_STATIC_DRAW);
    renderer_gl_update_freecam(state);

    glGenBuffers(12, state->gpu_vbos);
    glGenBuffers(1, &state->gpu_ebo);
    glGenBuffers(1, &g_stream);
    glBindBuffer(GL_ARRAY_BUFFER, g_stream);
    {
        void (*bufferStorage)(GLenum, GLsizeiptr, const void*, GLbitfield) =
            (void*) SDL_GL_GetProcAddress("glBufferStorage");
        const GLbitfield flags = GL_MAP_WRITE_BIT | 0x0040 /* PERSISTENT */ | 0x0080 /* COHERENT */;
        bufferStorage(GL_ARRAY_BUFFER, STREAM_SIZE, nullptr, flags);
        g_stream_ptr = glMapBufferRange(GL_ARRAY_BUFFER, 0, STREAM_SIZE, flags);
        glGenBuffers(1, &g_vstream);
        glBindBuffer(GL_ARRAY_BUFFER, g_vstream);
        bufferStorage(GL_ARRAY_BUFFER, VSTREAM_SIZE, nullptr, flags);
        g_vstream_ptr = glMapBufferRange(GL_ARRAY_BUFFER, 0, VSTREAM_SIZE, flags);
        // spike: the same again for vertex data the renderer thread lays out aligned
    }
    g_stream_off = 0;
    g_vstream_off = 0;
    glGetIntegerv(GL_UNIFORM_BUFFER_OFFSET_ALIGNMENT, &g_ubo_align);

    // we have 2 vaos since with geoshaders we need to fallback to sw shaders
    glGenVertexArrays(1, &state->gpu_vao_sw);
    glBindVertexArray(state->gpu_vao_sw);
    glBindBuffer(GL_ARRAY_BUFFER, state->gpu_vbos[0]);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state->gpu_ebo);

    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, pos));
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, color));
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, texcoord0));
    glEnableVertexAttribArray(2);
    glVertexAttribPointer(3, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, texcoord1));
    glEnableVertexAttribArray(3);
    glVertexAttribPointer(4, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, texcoord2));
    glEnableVertexAttribArray(4);
    glVertexAttribPointer(5, 1, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, texcoordw));
    glEnableVertexAttribArray(5);
    glVertexAttribPointer(6, 4, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, normquat));
    glEnableVertexAttribArray(6);
    glVertexAttribPointer(7, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex),
                          (void*) offsetof(Vertex, view));
    glEnableVertexAttribArray(7);

    // for hw vshaders attributes are setup at run time
    glGenVertexArrays(1, &state->gpu_vao_hw);
    glBindVertexArray(state->gpu_vao_hw);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, state->gpu_ebo);

    glGenTextures(2, state->screentex);
    glGenFramebuffers(2, state->screenfbo);
    GLint filter =
        ctremu.outputfilter == FILTER_NEAREST ? GL_NEAREST : GL_LINEAR;
    for (int i = 0; i < 2; i++) {
        glBindTexture(GL_TEXTURE_2D, state->screentex[i]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     SCREEN_HEIGHT * ctremu.videoscale,
                     SCREEN_WIDTH(i) * ctremu.videoscale, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, filter);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, filter);
        glBindFramebuffer(GL_FRAMEBUFFER, state->screenfbo[i]);
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                             state->screentex[i], 0);
    }

    glGenTextures(1, &state->swrendertex);
    glGenFramebuffers(1, &state->swrenderfbo);
    glBindTexture(GL_TEXTURE_2D, state->swrendertex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glBindFramebuffer(GL_FRAMEBUFFER, state->swrenderfbo);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                         state->swrendertex, 0);

    // create a blank texture for when games read textures
    // from invalid memory
    // the actual behavior is impossible to emulate in a hw renderer
    // usually its fine to just use a transparent texture instead
    glGenTextures(1, &state->blanktex);
    glBindTexture(GL_TEXTURE_2D, state->blanktex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 8, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                 nullptr);
    // blank texture has no mipmaps
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
    // make is always read 0 for all components
    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA,
                     (GLint[4]) {GL_ZERO, GL_ZERO, GL_ZERO,
                                 GL_ZERO}); // compound literals are cursed

    GLuint fbos[FB_MAX];
    glGenFramebuffers(FB_MAX, fbos);
    GLuint colorbufs[FB_MAX];
    glGenTextures(FB_MAX, colorbufs);
    GLuint depthbufs[FB_MAX];
    glGenTextures(FB_MAX, depthbufs);

    for (int i = 0; i < FB_MAX; i++) {
        gpu->fbs.d[i].fbo = fbos[i];
        gpu->fbs.d[i].color_tex = colorbufs[i];
        gpu->fbs.d[i].depth_tex = depthbufs[i];
    }

    GLuint textures[TEX_MAX];
    glGenTextures(TEX_MAX, textures);
    for (int i = 0; i < TEX_MAX; i++) {
        gpu->textures.d[i].tex = textures[i];
    }

    glGenTextures(1, &gpu->proctex.tex);
    glBindTexture(GL_TEXTURE_1D, gpu->proctex.tex);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);

    glGenTextures(1, &state->lightluttex);
    glBindTexture(GL_TEXTURE_1D_ARRAY, state->lightluttex);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage2D(GL_TEXTURE_1D_ARRAY, 0, GL_R16, countof(gpu->lightLuts[0]),
                 countof(gpu->lightLuts), 0, GL_RED, GL_UNSIGNED_SHORT,
                 nullptr);
    glGenTextures(1, &state->fogluttex);
    glBindTexture(GL_TEXTURE_1D, state->fogluttex);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R16, countof(gpu->fogLut), 0, GL_RED,
                 GL_UNSIGNED_SHORT, nullptr);
    glGenTextures(1, &state->proctexmaptex);
    glBindTexture(GL_TEXTURE_1D, state->proctexmaptex);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_RG16, countof(gpu->proctexMapLut), 0,
                 GL_RG, GL_UNSIGNED_SHORT, nullptr);
    glGenTextures(1, &state->proctexnoisetex);
    glBindTexture(GL_TEXTURE_1D, state->proctexnoisetex);
    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
    glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
    glTexImage1D(GL_TEXTURE_1D, 0, GL_R16, countof(gpu->proctexNoiseLut), 0,
                 GL_RED, GL_UNSIGNED_SHORT, nullptr);

    glGenBuffers(1, &state->pbo);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, state->pbo);
    glBufferData(GL_PIXEL_PACK_BUFFER,
                 4 * 512 * ctremu.videoscale * 512 * ctremu.videoscale, nullptr,
                 GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
}

void renderer_gl_destroy(GLState* state, GPU* gpu) {
    glDeleteProgram(state->main_program);
    glDeleteShader(state->gpu_vs);
    glDeleteShader(state->gpu_uberfs);
    for (int i = 0; i < MAX_PROGRAM; i++) {
        glDeleteProgram(state->progcache.d[i].prog);
    }
    for (int i = 0; i < VSH_MAX; i++) {
        glDeleteShader(gpu->vshaders_hw.d[i].vs);
    }
    for (int i = 0; i < FSH_MAX; i++) {
        glDeleteShader(gpu->fshaders.d[i].fs);
    }
    glDeleteVertexArrays(1, &state->main_vao);
    glDeleteVertexArrays(1, &state->gpu_vao_sw);
    glDeleteVertexArrays(1, &state->gpu_vao_hw);
    glDeleteBuffers(1, &state->main_vbo);
    glDeleteBuffers(12, state->gpu_vbos);
    glDeleteBuffers(1, &g_stream);
    glDeleteBuffers(1, &g_vstream);
    glDeleteBuffers(4, state->ubos);
    glDeleteBuffers(1, &state->gpu_ebo);
    glDeleteTextures(2, state->screentex);
    glDeleteFramebuffers(2, state->screenfbo);
    glDeleteTextures(1, &state->swrendertex);
    glDeleteFramebuffers(1, &state->swrenderfbo);
    for (int i = 0; i < FB_MAX; i++) {
        glDeleteFramebuffers(1, &gpu->fbs.d[i].fbo);
        glDeleteTextures(1, &gpu->fbs.d[i].color_tex);
        glDeleteTextures(1, &gpu->fbs.d[i].depth_tex);
    }
    for (int i = 0; i < TEX_MAX; i++) {
        glDeleteTextures(1, &gpu->textures.d[i].tex);
    }
    glDeleteTextures(1, &gpu->proctex.tex);
    glDeleteTextures(1, &state->lightluttex);
    glDeleteTextures(1, &state->fogluttex);
    glDeleteTextures(1, &state->proctexmaptex);
    glDeleteTextures(1, &state->proctexnoisetex);
    glDeleteBuffers(1, &state->pbo);
}

// call before emulating gpu drawing
u32 g_tex_frame; // spike: counts frames, so a texture's whole-content check happens once a frame
static void tsq_begin(void);
static void gls_invalidate(void);
static void gls_textures_unknown(void);
void gpu_gl_start_frame(GPU* gpu) {
    g_tex_frame++;
    gls_invalidate();
    tsq_begin();
    prog_unbind();
    glBindFramebuffer(GL_FRAMEBUFFER, gpu->curfb->fbo);
    if (g_wireframe) glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
}

// call at end of frame
// leaves framebuffer 0 bound at the end because on mac
// swap buffers wont work if it is not
bool (*gpu_gl_screen_pass)(int screen);
bool (*gpu_gl_panel_ground)();

void render_gl_main(GLState* state) {
    // reset gl for drawing the main window
    gls_invalidate();
    glUseProgram(state->main_program);
    glBindVertexArray(state->main_vao);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColorMask(true, true, true, true);
    glDisable(GL_BLEND);
    glDisable(GL_COLOR_LOGIC_OP);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);

    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);

    glActiveTexture(GL_TEXTURE0);

    if (gpu_gl_panel_ground && gpu_gl_panel_ground()) glUseProgram(state->main_program);

    for (int i = 0; i < 2; i++) {
        glViewport(ctremu.screens[i].x,
                   ctremu.windowH - ctremu.screens[i].y - ctremu.screens[i].h,
                   ctremu.screens[i].w, ctremu.screens[i].h);
        glBindTexture(GL_TEXTURE_2D, state->screentex[i]);
        if (ctremu.screens[i].w && gpu_gl_screen_pass && gpu_gl_screen_pass(i)) {
            glUseProgram(state->main_program);
            glBindVertexArray(state->main_vao);
            continue;
        }
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
}

// spike: present sets. The renderer thread copies the two screens into a set (two small blits)
// and the present thread draws the panels from it, so the renderer never waits for a vblank.
#define PSETS 8
static GLuint g_pset_tex[PSETS][2], g_pset_fbo[PSETS][2];
static GLsync g_pset_ready[PSETS]; // the writes into the set are done (renderer thread)
static GLsync g_pset_done[PSETS];  // the panels were drawn from it (present thread)
static int g_pset_last = -1, g_pset_filter = -1;
// direct mode: the display transfers write into a set themselves, no copy; a frame hands over
// the set that holds each screen's latest image (a screen the game did not redraw keeps its set)
static bool g_pset_direct;
static int (*g_pset_busy)(int* out);
static int g_pset_write = -1, g_pset_screen[2] = {-1, -1};

void gpu_gl_present_direct(bool on, int (*busy)(int* out)) {
    g_pset_direct = on && !getenv("SPIKE_PRESENT_COPY");
    g_pset_busy = busy;
}

// spike: GPU time per frame from timestamp queries at frame start and hand-over
u64 g_gpu_ns;
int g_gpu_n;
static GLuint g_tsq[8][2];
static bool g_tsq_open[8], g_tsq_off;
static int g_tsq_cur = -1;
static void tsq_poll() {
    for (int i = 0; i < 8; i++) {
        if (!g_tsq_open[i] || i == g_tsq_cur) continue;
        GLuint avail = 0;
        glGetQueryObjectuiv(g_tsq[i][1], GL_QUERY_RESULT_AVAILABLE, &avail);
        if (!avail) continue;
        GLuint64 t0 = 0, t1 = 0;
        glGetQueryObjectui64v(g_tsq[i][0], GL_QUERY_RESULT, &t0);
        glGetQueryObjectui64v(g_tsq[i][1], GL_QUERY_RESULT, &t1);
        if (t1 > t0 && t1 - t0 < 1000000000ull) {
            g_gpu_ns += t1 - t0;
            g_gpu_n++;
        }
        g_tsq_open[i] = false;
    }
}
static void tsq_begin() {
    // opt-in: polling a timestamp query's result waits in the driver (12% of the renderer
    // thread in a race) and ending one starts a fresh batch
    static int on = -1;
    if (on < 0) on = getenv("SPIKE_GPUTIME") != nullptr;
    if (g_tsq_off || !on) return;
    if (!g_tsq[0][0]) {
        glGetError();
        glGenQueries(16, &g_tsq[0][0]);
        glQueryCounter(g_tsq[0][0], GL_TIMESTAMP);
        if (glGetError() != GL_NO_ERROR) {
            g_tsq_off = true;
            fprintf(stderr, "spike: no timestamp queries\n");
            return;
        }
    }
    tsq_poll();
    int i = -1;
    for (int k = 0; k < 8; k++) {
        if (!g_tsq_open[k]) {
            i = k;
            break;
        }
    }
    if (i < 0) return;
    g_tsq_cur = i;
    g_tsq_open[i] = true;
    glQueryCounter(g_tsq[i][0], GL_TIMESTAMP);
}
static void tsq_end() {
    if (g_tsq_cur < 0) return;
    glQueryCounter(g_tsq[g_tsq_cur][1], GL_TIMESTAMP);
    g_tsq_cur = -1;
}

static void pset_init(GLState* state) {
    glGenTextures(PSETS * 2, &g_pset_tex[0][0]);
    glGenFramebuffers(PSETS * 2, &g_pset_fbo[0][0]);
    for (int s = 0; s < PSETS; s++) {
        for (int i = 0; i < 2; i++) {
            glBindTexture(GL_TEXTURE_2D, g_pset_tex[s][i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCREEN_HEIGHT * ctremu.videoscale,
                         SCREEN_WIDTH(i) * ctremu.videoscale, 0, GL_RGBA, GL_UNSIGNED_BYTE,
                         nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
            glBindFramebuffer(GL_FRAMEBUFFER, g_pset_fbo[s][i]);
            glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, g_pset_tex[s][i], 0);
        }
    }
}

// renderer thread: a set that no queued or drawing frame refers to and that holds no screen's
// latest image; the GPU must be done drawing from it before it is written again
static int pset_pick(GLState* state) {
    if (!g_pset_tex[0][0]) pset_init(state);
    int avoid[6], n = g_pset_busy ? g_pset_busy(avoid) : 0;
    avoid[n++] = g_pset_screen[0];
    avoid[n++] = g_pset_screen[1];
    int s = g_pset_last;
    for (int tries = 0; tries < PSETS; tries++) {
        s = (s + 1) % PSETS;
        bool bad = false;
        for (int i = 0; i < n; i++) bad |= avoid[i] == s;
        if (!bad) break;
    }
    g_pset_last = s;
    // Writing a set the present thread drew from a few frames ago needs no wait here: Panfrost's
    // kernel driver orders the write after earlier reads of the same buffer (implicit sync),
    // and a client wait is a glthread sync point that drains the whole queue, so the driver
    // thread and this one stop overlapping once a frame (at the clock the chip throttles to
    // after ten minutes of racing that cost frames: 99.5% -> 99.8% drawn without it, and the
    // driver thread 39% -> 34% of a core). SPIKE_PSETWAIT=1 waits anyway, for comparison.
    static int wait = -1;
    if (wait < 0) wait = getenv("SPIKE_PSETWAIT") != nullptr;
    if (g_pset_done[s]) {
        if (wait) glClientWaitSync(g_pset_done[s], GL_SYNC_FLUSH_COMMANDS_BIT, 100000000);
        glDeleteSync(g_pset_done[s]);
        g_pset_done[s] = nullptr;
    }
    return s;
}

// where a display transfer draws screen `screen` this frame
static GLuint screen_target_fbo(GPU* gpu, int screen) {
    if (!g_pset_direct) return gpu->gl.screenfbo[screen];
    if (g_pset_write < 0) g_pset_write = pset_pick(&gpu->gl);
    g_pset_screen[screen] = g_pset_write;
    return g_pset_fbo[g_pset_write][screen];
}

// renderer thread, at hand-over: the sets holding each screen's latest image (direct mode), or
// one set the screens are copied into
void gpu_gl_present_finish(GLState* state, int sets[2]) {
    tsq_end();
    int s;
    if (g_pset_direct) {
        s = g_pset_write;
        if (s < 0) {
            // nothing was drawn to the screens this frame: hand the last images over again
            s = g_pset_screen[0] >= 0 ? g_pset_screen[0] : pset_pick(state);
            if (g_pset_screen[0] < 0) g_pset_screen[0] = s;
            if (g_pset_screen[1] < 0) g_pset_screen[1] = s;
        }
        g_pset_write = -1;
        sets[0] = g_pset_screen[0];
        sets[1] = g_pset_screen[1];
    } else {
        s = pset_pick(state);
        gls_invalidate();
        glDisable(GL_SCISSOR_TEST);
        glColorMask(true, true, true, true);
        for (int i = 0; i < 2; i++) {
            int w = SCREEN_HEIGHT * ctremu.videoscale, h = SCREEN_WIDTH(i) * ctremu.videoscale;
            glBindFramebuffer(GL_READ_FRAMEBUFFER, state->screenfbo[i]);
            glBindFramebuffer(GL_DRAW_FRAMEBUFFER, g_pset_fbo[s][i]);
            glBlitFramebuffer(0, 0, w, h, 0, 0, w, h, GL_COLOR_BUFFER_BIT, GL_NEAREST);
        }
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        sets[0] = sets[1] = s;
    }
    if (g_pset_ready[s]) glDeleteSync(g_pset_ready[s]);
    g_pset_ready[s] = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    glFlush();
}

// present thread, in its own context: render_gl_main from the frame's sets
void gpu_gl_present_draw(GLState* state, const int sets[2], GLuint vao) {
    for (int i = 0; i < 2; i++) {
        if (g_pset_ready[sets[i]]) glWaitSync(g_pset_ready[sets[i]], 0, GL_TIMEOUT_IGNORED);
    }
    glUseProgram(state->main_program);
    glBindVertexArray(vao);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    glColorMask(true, true, true, true);
    glDisable(GL_BLEND);
    glDisable(GL_COLOR_LOGIC_OP);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_CULL_FACE);
    glDisable(GL_SCISSOR_TEST);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    if (gpu_gl_panel_ground && gpu_gl_panel_ground()) glUseProgram(state->main_program);
    if (g_pset_filter != ctremu.outputfilter) {
        g_pset_filter = ctremu.outputfilter;
        GLint f = g_pset_filter == FILTER_NEAREST ? GL_NEAREST : GL_LINEAR;
        for (int k = 0; k < PSETS; k++) {
            for (int i = 0; i < 2; i++) {
                glBindTexture(GL_TEXTURE_2D, g_pset_tex[k][i]);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, f);
                glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, f);
            }
        }
    }
    for (int i = 0; i < 2; i++) {
        glViewport(ctremu.screens[i].x, ctremu.windowH - ctremu.screens[i].y - ctremu.screens[i].h,
                   ctremu.screens[i].w, ctremu.screens[i].h);
        glBindTexture(GL_TEXTURE_2D, g_pset_tex[sets[i]][i]);
        if (ctremu.screens[i].w && gpu_gl_screen_pass && gpu_gl_screen_pass(i)) {
            glUseProgram(state->main_program);
            glBindVertexArray(vao);
            continue;
        }
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
}

void gpu_gl_present_done(const int sets[2]) {
    GLsync f = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    for (int i = 0; i < 2; i++) {
        int s = sets[i];
        if (i == 1 && sets[1] == sets[0]) break;
        if (g_pset_done[s]) glDeleteSync(g_pset_done[s]);
        // one fence object per set: a second set gets its own
        g_pset_done[s] = i == 0 ? f : glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    }
    glFlush();
}

void renderer_gl_update_freecam(GLState* state) {
    glBindBuffer(GL_UNIFORM_BUFFER, state->freecam_ubo);
    if (ctremu.freecam_enable) {
        glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof ctremu.freecam_mtx,
                        ctremu.freecam_mtx);
        glBufferSubData(GL_UNIFORM_BUFFER, 16 * 4, 4, &(int) {1});
    } else {
        glBufferSubData(GL_UNIFORM_BUFFER, 16 * 4, 4, &(int) {0});
    }
}

u64 g_ns_compile;
static GLuint compile_shader(GLuint type, char* source) {
    struct timespec ts0, ts1;
    clock_gettime(CLOCK_MONOTONIC, &ts0);
    extern GLuint compile_shader_real(GLuint type, char* source);
    GLuint r = compile_shader_real(type, source);
    clock_gettime(CLOCK_MONOTONIC, &ts1);
    g_ns_compile += (ts1.tv_sec - ts0.tv_sec) * 1000000000ull + ts1.tv_nsec - ts0.tv_nsec;
    return r;
}
GLuint compile_shader_real(GLuint type, char* source) {
    auto sh = glCreateShader(type);
    glShaderSource(sh, 1, &(const char*) {source}, nullptr);
    glCompileShader(sh);
    int res;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &res);
    if (!res) {
        char log[512];
        glGetShaderInfoLog(sh, sizeof log, nullptr, log);
        lerror("failed to compile shader: %s", log);
        printf("%s\n", source);
    }
    return sh;
}

int g_links;
u64 g_ns_shaders, g_ns_textures;
static u64 diag_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}
static GLuint link_program(GLuint vs, GLuint fs, bool sw, bool uber) {
    auto prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    int res;
    glGetProgramiv(prog, GL_LINK_STATUS, &res);
    if (!res) {
        char log[512];
        glGetProgramInfoLog(prog, sizeof log, nullptr, log);
        lerror("failed to link program: %s", log);
    }
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "tex0"), 0);
    glUniform1i(glGetUniformLocation(prog, "tex0shadow"), 0);
    glUniform1i(glGetUniformLocation(prog, "tex1"), 1);
    glUniform1i(glGetUniformLocation(prog, "tex2"), 2);
    glUniform1i(glGetUniformLocation(prog, "tex3"), 3);
    glUniform1i(glGetUniformLocation(prog, "lightLuts"), 4);
    glUniform1i(glGetUniformLocation(prog, "fogLut"), 5);
    glUniform1i(glGetUniformLocation(prog, "proctexMapLut"), 6);
    glUniform1i(glGetUniformLocation(prog, "proctexNoiseLut"), 7);

    if (!sw) {
        glUniformBlockBinding(prog,
                              glGetUniformBlockIndex(prog, "VertUniforms"), 0);
        glUniformBlockBinding(
            prog, glGetUniformBlockIndex(prog, "FreecamUniforms"), 3);
    }
    if (uber)
        glUniformBlockBinding(prog, glGetUniformBlockIndex(prog, "FragConfig"),
                              1);
    glUniformBlockBinding(prog, glGetUniformBlockIndex(prog, "FragUniforms"),
                          2);
    return prog;
}

// Programs are built on a helper thread with its own shared context, and each
// game keeps a list of them so the next run has them ready before they are
// drawn. Bump PROGLIST_VERSION whenever the shader generators change: a list
// maps config hashes to generated text.
#define PROGLIST_VERSION 1
#define PROG_BUCKETS 4096

typedef struct ProgSrc {
    u64 hash;
    char* src;
    bool ondisk;
    struct ProgSrc* next;
} ProgSrc;

typedef struct ProgEnt {
    ProgSrc* v;
    ProgSrc* f;
    _Atomic int state; // 0 queued, 1 building, 2 ready
    bool inlist, hot;
    GLuint prog;
    GLint depthloc[3];
    struct ProgEnt* next;
} ProgEnt;

void (*gpu_gl_helper_bind)(void);
u64 g_ns_progwait;
int g_prog_ready, g_prog_total, g_prog_cold;

static struct {
    ProgSrc* vsrc[PROG_BUCKETS];
    ProgSrc* fsrc[PROG_BUCKETS];
    ProgEnt* prog[PROG_BUCKETS];
    pthread_t thread;
    pthread_mutex_t lock;
    pthread_cond_t wake, done;
    ProgEnt** hot;
    int nhot, caphot;
    ProgEnt** cold;
    int ncold, coldpos;
    bool on;
    FILE* list;
    ProgSrc* sw_vs;
    ProgSrc* uber_fs;
    ProgSrc* cur_hw_vs;
    ProgEnt* cur;
    bool bound;
} g_pb = {.lock = PTHREAD_MUTEX_INITIALIZER,
          .wake = PTHREAD_COND_INITIALIZER,
          .done = PTHREAD_COND_INITIALIZER};

static ProgSrc* src_find(ProgSrc** tab, u64 hash) {
    for (ProgSrc* p = tab[hash % PROG_BUCKETS]; p; p = p->next) {
        if (p->hash == hash) return p;
    }
    return nullptr;
}

static ProgSrc* src_add(ProgSrc** tab, u64 hash, char* src, bool ondisk) {
    ProgSrc* p = calloc(1, sizeof *p);
    p->hash = hash;
    p->src = src;
    p->ondisk = ondisk;
    p->next = tab[hash % PROG_BUCKETS];
    tab[hash % PROG_BUCKETS] = p;
    return p;
}

static ProgEnt* prog_find(ProgSrc* v, ProgSrc* f) {
    u64 h = (v->hash ^ (f->hash * 31)) % PROG_BUCKETS;
    for (ProgEnt* e = g_pb.prog[h]; e; e = e->next) {
        if (e->v == v && e->f == f) return e;
    }
    return nullptr;
}

static ProgEnt* prog_add(ProgSrc* v, ProgSrc* f, bool inlist) {
    u64 h = (v->hash ^ (f->hash * 31)) % PROG_BUCKETS;
    ProgEnt* e = calloc(1, sizeof *e);
    e->v = v;
    e->f = f;
    e->inlist = inlist;
    e->next = g_pb.prog[h];
    g_pb.prog[h] = e;
    g_prog_total++;
    return e;
}

static void list_put_src(char type, ProgSrc* p) {
    if (p->ondisk) return;
    p->ondisk = true;
    u32 len = strlen(p->src);
    fputc(type, g_pb.list);
    fwrite(&p->hash, sizeof p->hash, 1, g_pb.list);
    fwrite(&len, sizeof len, 1, g_pb.list);
    fwrite(p->src, 1, len, g_pb.list);
}

static void prog_build(ProgEnt* e, bool helper) {
    atomic_store(&e->state, 1);
    u64 t0 = diag_now();
    GLuint vs = compile_shader(GL_VERTEX_SHADER, e->v->src);
    GLuint fs = compile_shader(GL_FRAGMENT_SHADER, e->f->src);
    GLuint prog = link_program(vs, fs, e->v == g_pb.sw_vs, e->f == g_pb.uber_fs);
    glDetachShader(prog, vs);
    glDetachShader(prog, fs);
    glDeleteShader(vs);
    glDeleteShader(fs);
    e->depthloc[0] = glGetUniformLocation(prog, "depthOffset");
    e->depthloc[1] = glGetUniformLocation(prog, "depthScale");
    e->depthloc[2] = glGetUniformLocation(prog, "depthWBuffer");
    if (helper) {
        glUseProgram(0);
        // the other context may use it as soon as it is published
        glFinish();
    } else {
        g_pb.bound = false;
    }
    e->prog = prog;
    g_links++;
    g_ns_shaders += diag_now() - t0;

    if (g_pb.list && !e->inlist) {
        list_put_src('V', e->v);
        list_put_src('F', e->f);
        fputc('P', g_pb.list);
        fwrite(&e->v->hash, sizeof(u64), 1, g_pb.list);
        fwrite(&e->f->hash, sizeof(u64), 1, g_pb.list);
        fflush(g_pb.list);
    }

    pthread_mutex_lock(&g_pb.lock);
    atomic_store(&e->state, 2);
    g_prog_ready++;
    pthread_cond_broadcast(&g_pb.done);
    pthread_mutex_unlock(&g_pb.lock);
}

static void* prog_helper(void*) {
    if (getenv("SPIKE_PIN")) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(1, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    setpriority(PRIO_PROCESS, syscall(SYS_gettid), 5);
    pthread_setname_np(pthread_self(), "ctremu:prog");
    gpu_gl_helper_bind();
    while (true) {
        pthread_mutex_lock(&g_pb.lock);
        ProgEnt* e = nullptr;
        while (!e) {
            if (g_pb.nhot) {
                e = g_pb.hot[0];
                memmove(g_pb.hot, g_pb.hot + 1,
                        --g_pb.nhot * sizeof g_pb.hot[0]);
            } else if (g_pb.coldpos < g_pb.ncold) {
                e = g_pb.cold[g_pb.coldpos++];
            } else {
                pthread_cond_wait(&g_pb.wake, &g_pb.lock);
                continue;
            }
            if (atomic_load(&e->state) != 0) e = nullptr;
        }
        atomic_store(&e->state, 1);
        pthread_mutex_unlock(&g_pb.lock);
        prog_build(e, true);
    }
    return nullptr;
}

// asks for a program, moving it ahead of the list when it is not built yet
static ProgEnt* prog_request(ProgSrc* v, ProgSrc* f) {
    ProgEnt* e = prog_find(v, f);
    if (!e) e = prog_add(v, f, false);
    if (atomic_load_explicit(&e->state, memory_order_acquire) == 2) return e;
    if (!g_pb.on) {
        prog_build(e, false);
        return e;
    }
    pthread_mutex_lock(&g_pb.lock);
    if (!e->hot && atomic_load(&e->state) == 0) {
        e->hot = true;
        if (g_pb.nhot == g_pb.caphot) {
            g_pb.caphot = g_pb.caphot ? g_pb.caphot * 2 : 64;
            g_pb.hot = realloc(g_pb.hot, g_pb.caphot * sizeof g_pb.hot[0]);
        }
        g_pb.hot[g_pb.nhot++] = e;
        pthread_cond_signal(&g_pb.wake);
    }
    pthread_mutex_unlock(&g_pb.lock);
    return e;
}

static void prog_wait(ProgEnt* e) {
    u64 t0 = diag_now();
    pthread_mutex_lock(&g_pb.lock);
    while (atomic_load(&e->state) != 2) {
        pthread_cond_wait(&g_pb.done, &g_pb.lock);
    }
    pthread_mutex_unlock(&g_pb.lock);
    g_ns_progwait += diag_now() - t0;
}

static void prog_list_load(const char* path) {
    FILE* fp = fopen(path, "rb");
    long good = 0;
    bool valid = false;
    if (fp) {
        u32 head[2];
        if (fread(head, sizeof head, 1, fp) == 1 && head[0] == 0x314c5054 &&
            head[1] == PROGLIST_VERSION) {
            valid = true;
            good = ftell(fp);
            while (true) {
                int type = fgetc(fp);
                if (type == 'V' || type == 'F') {
                    u64 hash;
                    u32 len;
                    if (fread(&hash, sizeof hash, 1, fp) != 1 ||
                        fread(&len, sizeof len, 1, fp) != 1 || len > 1 << 22) {
                        break;
                    }
                    char* src = malloc(len + 1);
                    if (fread(src, 1, len, fp) != len) {
                        free(src);
                        break;
                    }
                    src[len] = 0;
                    ProgSrc** tab = type == 'V' ? g_pb.vsrc : g_pb.fsrc;
                    if (src_find(tab, hash)) free(src);
                    else src_add(tab, hash, src, true);
                } else if (type == 'P') {
                    u64 h[2];
                    if (fread(h, sizeof h, 1, fp) != 1) break;
                    ProgSrc* v = src_find(g_pb.vsrc, h[0]);
                    ProgSrc* f = src_find(g_pb.fsrc, h[1]);
                    if (v && f && !prog_find(v, f)) {
                        ProgEnt* e = prog_add(v, f, true);
                        g_pb.cold = realloc(g_pb.cold, (g_pb.ncold + 1) *
                                                           sizeof g_pb.cold[0]);
                        g_pb.cold[g_pb.ncold++] = e;
                    }
                } else {
                    break;
                }
                good = ftell(fp);
            }
        }
        fclose(fp);
    }
    if (valid) {
        // a power cut can leave half a record behind
        truncate(path, good);
        g_pb.list = fopen(path, "ab");
    } else {
        g_pb.list = fopen(path, "wb");
        if (g_pb.list) {
            u32 head[2] = {0x314c5054, PROGLIST_VERSION};
            fwrite(head, sizeof head, 1, g_pb.list);
            fflush(g_pb.list);
        }
    }
    g_prog_cold = g_pb.ncold;
    fprintf(stderr, "spike program list %s: %d programs\n", path, g_pb.ncold);
}

static void prog_builder_start() {
    static bool started;
    if (started) return;
    started = true;
    g_pb.sw_vs = src_add(g_pb.vsrc, 1, (char*) gpuvertsource, true);
    g_pb.uber_fs = src_add(g_pb.fsrc, 1, (char*) gpufragsource, true);
    if (ctremu.romfilenoext && !getenv("SPIKE_NOPROGLIST")) {
        mkdir("shaders", 0777);
        char* path;
        asprintf(&path, "shaders/%s.list", ctremu.romfilenoext);
        prog_list_load(path);
        free(path);
    }
    if (gpu_gl_helper_bind && !getenv("SPIKE_NOASYNC")) {
        g_pb.on = true;
        pthread_create(&g_pb.thread, nullptr, prog_helper, nullptr);
    }
}

static void prog_unbind() { g_pb.bound = false; }

static void update_cur_fb(GPU* gpu) {
    u32 w = gpu->regs.fb.dim.width;
    u32 h = gpu->regs.fb.dim.height + 1;
    // using the same fb (height does not matter if its big enough)
    if (gpu->curfb->color_paddr == (gpu->regs.fb.colorbuf_loc << 3) &&
        gpu->curfb->width == w && h <= gpu->curfb->height)
        return;

    if (gpu->regs.fb.colorbuf_loc == 0) {
        // uhh (if we ignored this it would corrupt the cache)
        lwarn("null framebuffer");
        return;
    }

    // little hack to make arisoturas sm64 port work
    // it clears the depthbuffer by binding it as the colorbuffer
    // and drawing on it
    for (int i = 0; i < FB_MAX; i++) {
        if (gpu->fbs.d[i].depth_paddr == gpu->regs.fb.colorbuf_loc << 3) {
            LRU_use(gpu->fbs, &gpu->fbs.d[i]);
            glBindFramebuffer(GL_FRAMEBUFFER, gpu->fbs.d[i].fbo);
            glClearDepth(0);
            glDepthMask(true);
            glClear(GL_DEPTH_BUFFER_BIT);
            linfo("lmao");
        }
    }

    auto curfb = LRU_load(gpu->fbs, gpu->regs.fb.colorbuf_loc << 3);

    curfb->color_paddr = gpu->regs.fb.colorbuf_loc << 3;
    curfb->depth_paddr = gpu->regs.fb.depthbuf_loc << 3;
    curfb->color_fmt = gpu->regs.fb.colorbuf_fmt.fmt;
    curfb->color_Bpp = gpu->regs.fb.colorbuf_fmt.size + 2;
    curfb->depth_fmt = gpu->regs.fb.depthbuf_fmt & 3;

    linfo("drawing on fb %d at %x with depth buffer at %x", curfb - gpu->fbs.d,
          curfb->color_paddr, curfb->depth_paddr);

    glBindFramebuffer(GL_FRAMEBUFFER, curfb->fbo);

    if (w != curfb->width || h > curfb->height) {
        curfb->width = w;
        curfb->height = h;

        linfo("creating new fb at %08x with w=%d, h=%d", curfb->color_paddr,
              curfb->width, curfb->height);

        glBindTexture(GL_TEXTURE_2D, curfb->color_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                     curfb->width * ctremu.videoscale,
                     curfb->height * ctremu.videoscale, 0, GL_RGBA,
                     GL_UNSIGNED_BYTE, nullptr);
        // framebuffers have no mipmaps
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        glBindTexture(GL_TEXTURE_2D, curfb->depth_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8,
                     curfb->width * ctremu.videoscale,
                     curfb->height * ctremu.videoscale, 0, GL_DEPTH_STENCIL,
                     GL_UNSIGNED_INT_24_8, nullptr);
        // needed for the texture viewer
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);

        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                               GL_TEXTURE_2D, curfb->color_tex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT,
                               GL_TEXTURE_2D, curfb->depth_tex, 0);
    }

    gpu->curfb = curfb;
}

// the framebuffers' colours in a save state; depth starts over
void savestate_put(void* ss, const void* p, size_t n);
bool savestate_get(void* ss, void* p, size_t n);

// with the renderer stopped; the rest of a state is then written on its own thread
// The pictures for a save state: fields one by one, as the rest of the state.
#define FB_FIELDS 42 // bytes each framebuffer's fields take
#define BLOB_PUT(v)                                                            \
    ({                                                                         \
        auto v_ = (v);                                                         \
        memcpy(p, &v_, sizeof v_);                                             \
        p += sizeof v_;                                                        \
    })
#define BLOB_GET(T)                                                            \
    ({                                                                         \
        T v_;                                                                  \
        memcpy(&v_, blob, sizeof v_);                                          \
        blob += sizeof v_;                                                     \
        v_;                                                                    \
    })

void* gpu_gl_fbs_capture(GPU* gpu, u64* size) {
    u32 count = 0;
    u64 total = sizeof count + sizeof(u64) + sizeof(u32);
    for (int i = 0; i < FB_MAX; i++) {
        FBInfo* fb = &gpu->fbs.d[i];
        if (!fb->color_paddr || !fb->width) continue;
        count++;
        total += FB_FIELDS + (u64) fb->width * fb->height * ctremu.videoscale *
                                 ctremu.videoscale * 4;
    }
    u8* blob = calloc(1, total); // the unused bytes after a framebuffer's fields stay zero
    u8* p = blob;
    BLOB_PUT(count);
    BLOB_PUT((u64) (gpu->curfb ? gpu->curfb->color_paddr : 0));
    BLOB_PUT((u32) ctremu.videoscale);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    for (int i = 0; i < FB_MAX; i++) {
        FBInfo* fb = &gpu->fbs.d[i];
        if (!fb->color_paddr || !fb->width) continue;
        u8* fields = p;
        BLOB_PUT((u64) fb->color_paddr);
        BLOB_PUT((u32) fb->depth_paddr);
        BLOB_PUT((u32) fb->width);
        BLOB_PUT((u32) fb->height);
        BLOB_PUT((u32) fb->color_fmt);
        BLOB_PUT((u32) fb->color_Bpp);
        BLOB_PUT((u32) fb->depth_fmt);
        BLOB_PUT((u8) fb->shadowMap);
        BLOB_PUT((u8) fb->dirty);
        p = fields + FB_FIELDS;
        u32 w = fb->width * ctremu.videoscale, h = fb->height * ctremu.videoscale;
        glBindFramebuffer(GL_FRAMEBUFFER, fb->fbo);
        glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, p);
        p += w * h * 4;
    }
    glBindFramebuffer(GL_FRAMEBUFFER, gpu->curfb && gpu->curfb->fbo ? gpu->curfb->fbo : 0);
    *size = total;
    return blob;
}

// gpu is the running renderer, from the guest state as it was saved
void gpu_gl_state_loaded(GPU* gpu, GPU* from, const u8* blob) {
    gpu->gsh = from->gsh;
    gpu->vsh = from->vsh;
    gpu->gsh.code_dirty = gpu->vsh.code_dirty = true;
    gpu->vsh_uniform_dirty = true;
    memcpy(gpu->fixattrs, from->fixattrs, sizeof gpu->fixattrs);
    gpu->curfixattr = from->curfixattr;
    gpu->curfixi = from->curfixi;
    gpu->immattrs.size = 0;
    memcpy(gpu->lightLuts, from->lightLuts, sizeof gpu->lightLuts);
    memcpy(gpu->fogLut, from->fogLut, sizeof gpu->fogLut);
    memcpy(gpu->proctexMapLut, from->proctexMapLut, sizeof gpu->proctexMapLut);
    memcpy(gpu->proctexNoiseLut, from->proctexNoiseLut, sizeof gpu->proctexNoiseLut);
    memcpy(gpu->proctexLut, from->proctexLut, sizeof gpu->proctexLut);
    gpu->lightLutDirty = gpu->fogLutDirty = gpu->proctexMapLutDirty = true;
    gpu->proctexNoiseLutDirty = gpu->proctexLutDirty = true;
    gpu->proctex.width = 0;
    gpu->proctex.offset = 0;
    gpu->lastFragUboHash = 0;
    gpu->regs = from->regs;
    g_pb.cur_hw_vs = nullptr;
    g_pb.cur = nullptr;
    g_pb.bound = false;

    // every texture comes from guest memory again
    gpu_gl_shadows_drop(gpu);
    for (int i = 0; i < TEX_MAX; i++) {
        TexInfo* t = &gpu->textures.d[i];
        t->width = t->height = 0;
        t->hash = 0;
        t->sampler_valid = false;
        t->etcnative = false;
    }
    for (int i = 0; i < TEX_MAX; i++) {
        TexInfo* t = &gpu->textures.d[i];
        if (t->next) LRU_remove(gpu->textures, t);
        t->paddr = 0;
    }
    for (int i = 0; i < FB_MAX; i++) {
        FBInfo* fb = &gpu->fbs.d[i];
        if (fb->next) LRU_remove(gpu->fbs, fb);
        fb->color_paddr = 0;
        fb->depth_paddr = 0;
        fb->width = fb->height = 0;
    }
    gpu->curfb = &gpu->fbs.root;

    u32 count = BLOB_GET(u32);
    u64 cur = BLOB_GET(u64);
    u32 scale = BLOB_GET(u32);
    for (u32 n = 0; n < count && n < FB_MAX; n++) {
        const u8* fields = blob;
        u64 color_paddr = BLOB_GET(u64);
        FBInfo* fb = LRU_load(gpu->fbs, color_paddr);
        fb->color_paddr = color_paddr;
        fb->depth_paddr = BLOB_GET(u32);
        fb->width = BLOB_GET(u32);
        fb->height = BLOB_GET(u32);
        fb->color_fmt = BLOB_GET(u32);
        fb->color_Bpp = BLOB_GET(u32);
        fb->depth_fmt = BLOB_GET(u32);
        fb->shadowMap = BLOB_GET(u8);
        fb->dirty = BLOB_GET(u8);
        blob = fields + FB_FIELDS;
        u32 w = fb->width * ctremu.videoscale, h = fb->height * ctremu.videoscale;
        // saved at another resolution: the game draws the picture again on its next frame
        bool same = scale == (u32) ctremu.videoscale;
        glBindFramebuffer(GL_FRAMEBUFFER, fb->fbo);
        glBindTexture(GL_TEXTURE_2D, fb->color_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, same ? blob : nullptr);
        blob += (u64) fb->width * fb->height * scale * scale * 4;
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glBindTexture(GL_TEXTURE_2D, fb->depth_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH24_STENCIL8, w, h, 0, GL_DEPTH_STENCIL,
                     GL_UNSIGNED_INT_24_8, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fb->color_tex, 0);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_TEXTURE_2D, fb->depth_tex, 0);
    }
    for (int i = 0; i < FB_MAX; i++) {
        if (cur && gpu->fbs.d[i].color_paddr == cur) gpu->curfb = &gpu->fbs.d[i];
    }
    if (gpu->curfb->fbo) glBindFramebuffer(GL_FRAMEBUFFER, gpu->curfb->fbo);
}

#define COPYRGBA(dst, src)                                                     \
    ({                                                                         \
        dst[0] = (float) src.r / 255;                                          \
        dst[1] = (float) src.g / 255;                                          \
        dst[2] = (float) src.b / 255;                                          \
        dst[3] = (float) src.a / 255;                                          \
    })

#define COPYRGB(dst, src)                                                      \
    ({                                                                         \
        dst[0] = (float) src.r / 255;                                          \
        dst[1] = (float) src.g / 255;                                          \
        dst[2] = (float) src.b / 255;                                          \
    })

void gpu_gl_display_transfer(GPU* gpu, u32 paddr, int yoffdst, bool scalex,
                             bool scaley, bool vflip, int screenid) {

    FBInfo* fb = gpu_fbcache_find_within(gpu, paddr);
    if (!fb) {
        lwarnonce("could not find source fb at %08x", paddr);
        return;
    }
    int yoffsrc = (paddr - fb->color_paddr) / (fb->color_Bpp * fb->width);

    linfo("display transfer fb at %x to %s", paddr,
          screenid == SCREEN_TOP ? "top" : "bot");

    // scissor test and color mask affects blit framebuffer
    gls_invalidate();
    glDisable(GL_SCISSOR_TEST);
    glColorMask(true, true, true, true);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, fb->fbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, screen_target_fbo(gpu, screenid));

    int yoff = yoffsrc + yoffdst;

    int srcX0 = 0;
    int srcY0 = (fb->height - (SCREEN_WIDTH(screenid) << scaley) - yoff) *
                ctremu.videoscale;
    int srcX1 = (SCREEN_HEIGHT << scalex) * ctremu.videoscale;
    int srcY1 = (fb->height - yoff) * ctremu.videoscale;
    int dstX0 = 0;
    int dstY0 = vflip ? SCREEN_WIDTH(screenid) * ctremu.videoscale : 0;
    int dstX1 = SCREEN_HEIGHT * ctremu.videoscale;
    int dstY1 = vflip ? 0 : SCREEN_WIDTH(screenid) * ctremu.videoscale;

    glBlitFramebuffer(srcX0, srcY0, srcX1, srcY1, dstX0, dstY0, dstX1, dstY1,
                      GL_COLOR_BUFFER_BIT, GL_LINEAR);

    // make sure this gets rebound before next draw
    glBindFramebuffer(GL_FRAMEBUFFER, gpu->curfb->fbo);
}

void gpu_gl_render_lcd_fb(GPU* gpu, u32 paddr, u32 fmt, int screenid) {
    linfo("directly rendering lcd fb at %08x to screen %d", paddr, screenid);

    // ngl this whole function is extremely hacky and ignores many things

    void* data = PTR(paddr);

    int colorfmt = fmt & 7;
    GLuint glfmt = (GLuint[8]) {
        GL_RGBA, GL_BGR, GL_RGB, GL_RGBA, GL_RGBA, GL_RGBA, GL_RGBA, GL_RGBA,
    }[colorfmt];
    GLuint gltype = (GLuint[8]) {
        GL_UNSIGNED_INT_8_8_8_8,   GL_UNSIGNED_BYTE,
        GL_UNSIGNED_SHORT_5_6_5,   GL_UNSIGNED_SHORT_5_5_5_1,
        GL_UNSIGNED_SHORT_4_4_4_4, GL_UNSIGNED_INT_8_8_8_8,
        GL_UNSIGNED_INT_8_8_8_8,   GL_UNSIGNED_INT_8_8_8_8,
    }[colorfmt];

    glBindTexture(GL_TEXTURE_2D, gpu->gl.swrendertex);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, SCREEN_HEIGHT,
                 SCREEN_WIDTH(screenid), 0, glfmt, gltype, data);

    // scissor test and color mask affects blit framebuffer
    gls_invalidate();
    glDisable(GL_SCISSOR_TEST);
    glColorMask(true, true, true, true);

    glBindFramebuffer(GL_READ_FRAMEBUFFER, gpu->gl.swrenderfbo);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, screen_target_fbo(gpu, screenid));

    glBlitFramebuffer(0, 0, SCREEN_HEIGHT, SCREEN_WIDTH(screenid), 0,
                      SCREEN_WIDTH(screenid) * ctremu.videoscale,
                      SCREEN_HEIGHT * ctremu.videoscale, 0, GL_COLOR_BUFFER_BIT,
                      GL_NEAREST);

    glBindFramebuffer(GL_FRAMEBUFFER, gpu->curfb->fbo);
}

// spike: games animate a few tiles of a large ETC1 atlas with texture copies.
// Dropping the texture for that decodes the whole atlas again on the next draw,
// so decode just the copied rectangle. Only for copies that are whole tiles, row
// by row of the atlas.
static bool g_native_etc, g_etc_this_tex;
int g_etc_clamped, g_etc_blocks, g_etc_refused;

static void gpu_gl_etc_start() { g_native_etc = !getenv("SPIKE_NOETC"); }

// spike: shadow copies. Only ETC textures (the ones decoded on the CPU) get one, within a budget.
static u64 g_shadow_bytes;
#define SHADOW_MAX_ONE (8u << 20)
#define SHADOW_MAX_ALL (96ull << 20)
int g_tex_diffs, g_tex_difftiles, g_tex_new, g_tex_stalefull, g_tex_rehashfull, g_partial_fail[4],
    g_tex_difffull;
u64 g_ns_texdiff;
static int g_tex_diffs_all, g_tex_difftiles_all; // whole run, for the exit summary
static u64 g_ns_texdiff_all;
static int g_create_reason; // 0 new or changed attributes, 1 stale bytes, 2 hash changed, 3 diff gave up

static void tex_shadow_drop(TexInfo* tex) {
    if (!tex->shadow) return;
    g_shadow_bytes -= tex->shadow_size;
    free(tex->shadow);
    tex->shadow = nullptr;
    tex->shadow_size = 0;
}

void gpu_gl_shadows_drop(GPU* gpu) {
    for (int i = 0; i < TEX_MAX; i++) tex_shadow_drop(&gpu->textures.d[i]);
}

static void tex_shadow_refresh(GPU* gpu, TexInfo* tex) {
    if ((tex->fmt != 12 && tex->fmt != 13) || getenv("SPIKE_NOSHADOW")) {
        tex_shadow_drop(tex);
        return;
    }
    if (tex->shadow && tex->shadow_size != tex->size) tex_shadow_drop(tex);
    if (!tex->shadow) {
        if (tex->size > SHADOW_MAX_ONE || g_shadow_bytes + tex->size > SHADOW_MAX_ALL) return;
        tex->shadow = malloc(tex->size);
        if (!tex->shadow) return;
        tex->shadow_size = tex->size;
        g_shadow_bytes += tex->size;
    }
    memcpy(tex->shadow, PTR(tex->paddr), tex->size);
    tex->dirty_lo = tex->dirty_hi = 0;
}

// Decodes th rows of tw tiles (8x8 pixels each) of level l, read from `row` with `pitch` bytes
// per row of tiles and `rowbytes` between rows, and uploads them at tile (tx, ty). The texture
// is bound. False when an ETC-native texture meets a block it cannot flip exactly.
static bool tex_upload_tiles(TexInfo* tex, int l, u32 lh, u32 tx, u32 ty, u32 tw, u32 th,
                             const u8* row, u32 pitch, u32 rowbytes) {
    u32 w = tw * 8, h = th * 8;
    u8* raw = malloc(pitch * th);
    u8* dec = malloc(w * h * 4);
    for (u32 i = 0; i < th; i++) {
        memcpy(raw + i * pitch, row + i * rowbytes, pitch);
    }
    bool ok = true;
    if (tex->etcnative) {
        int clamped = 0;
        if (etc1_to_gpu(w, h, (void*) raw, 1, (void*) dec, &clamped)) {
            g_etc_clamped += clamped;
            glCompressedTexSubImage2D(GL_TEXTURE_2D, l, tx * 8, lh - (ty * 8 + h), w, h,
                                      GL_COMPRESSED_RGB8_ETC2, w * h / 2, dec);
        } else {
            ok = false;
        }
    } else if (tex->fmt == 12) {
        etc1_decompress_texture(w, h, (void*) raw, (void*) dec);
        glTexSubImage2D(GL_TEXTURE_2D, l, tx * 8, lh - (ty * 8 + h), w, h, GL_RGB,
                        GL_UNSIGNED_BYTE, dec);
    } else {
        etc1a4_decompress_texture(w, h, (void*) raw, (void*) dec);
        glTexSubImage2D(GL_TEXTURE_2D, l, tx * 8, lh - (ty * 8 + h), w, h, GL_RGBA,
                        GL_UNSIGNED_BYTE, dec);
    }
    free(raw);
    free(dec);
    return ok;
}

// Uploads the tiles whose bytes differ from the shadow copy within lo..hi. Returns the
// number of tiles, 0 when nothing changed, -1 when the texture has to be rebuilt whole
// (no shadow, a mip level too small for tiles, or a block the native path cannot flip).
static int tex_diff_update(GPU* gpu, TexInfo* tex, u32 lo, u32 hi) {
    if (!tex->shadow || tex->shadow_size != tex->size) return -1;
    if (hi > tex->size) hi = tex->size;
    if (lo >= hi) return 0;
    u64 t0 = diag_now();
    u32 tilebytes = tex->fmt == 12 ? 32 : 64;
    const u8* guest = PTR(tex->paddr);
    u32 off = 0;
    int tiles = 0;
    gls_textures_unknown(); // the uploads bind on the active unit
    for (int l = tex->minlod; l <= tex->maxlod; l++) {
        u32 lw = tex->width >> l, lh = tex->height >> l;
        u32 levelsize = lw * lh * (tex->fmt == 12 ? 4 : 8) / 8;
        if (off < hi && lo < off + levelsize) {
            if (lw < 8 || lh < 8) return -1;
            u32 rowbytes = lw / 8 * tilebytes, ntx = lw / 8;
            u32 ty0 = (lo > off ? lo - off : 0) / rowbytes;
            u32 end = hi < off + levelsize ? hi : off + levelsize;
            u32 ty1 = (end - off + rowbytes - 1) / rowbytes;
            if (ty1 > lh / 8) ty1 = lh / 8;
            // the changed tiles' bounding box, uploaded once: each glTexSubImage costs about
            // 0.13 ms here (Panfrost flushes the batch that reads the texture), the decode of a
            // few extra tiles does not
            u32 bx0 = ntx, bx1 = 0, by0 = lh / 8, by1 = 0;
            for (u32 ty = ty0; ty < ty1; ty++) {
                u32 rowoff = off + ty * rowbytes;
                const u8* g = guest + rowoff;
                const u8* s = tex->shadow + rowoff;
                if (!memcmp(g, s, rowbytes)) continue;
                for (u32 tx = 0; tx < ntx; tx++) {
                    if (memcmp(g + tx * tilebytes, s + tx * tilebytes, tilebytes)) {
                        if (tx < bx0) bx0 = tx;
                        if (tx + 1 > bx1) bx1 = tx + 1;
                        if (ty < by0) by0 = ty;
                        if (ty + 1 > by1) by1 = ty + 1;
                    }
                }
            }
            if (bx1 > bx0) {
                u32 tw = bx1 - bx0, th = by1 - by0;
                const u8* g = guest + off + by0 * rowbytes + bx0 * tilebytes;
                if (!tex_upload_tiles(tex, l, lh, bx0, by0, tw, th, g, tw * tilebytes, rowbytes)) {
                    g_ns_texdiff += diag_now() - t0;
                    return -1;
                }
                for (u32 ty = by0; ty < by1; ty++) {
                    u32 o = off + ty * rowbytes + bx0 * tilebytes;
                    memcpy(tex->shadow + o, guest + o, tw * tilebytes);
                }
                tiles += tw * th;
            }
        }
        off += levelsize;
    }
    u64 dt = diag_now() - t0;
    g_ns_texdiff += dt;
    g_ns_texdiff_all += dt;
    g_tex_diffs_all++;
    g_tex_difftiles_all += tiles;
    static int check = -1;
    if (check < 0) check = getenv("SPIKE_DIFFCHECK") != nullptr;
    if (check && memcmp(tex->shadow, guest, tex->size)) {
        u32 first = 0;
        while (first < tex->size && tex->shadow[first] == guest[first]) first++;
        spike_logf("diffcheck: texture %08x fmt %d %dx%d still differs at byte %u after a diff "
                   "of %u..%u (%d tiles)\n",
                   tex->paddr, tex->fmt, tex->width, tex->height, first, lo, hi, tiles);
        memcpy(tex->shadow, guest, tex->size);
        return -1;
    }
    return tiles;
}

static bool tex_partial_update(GPU* gpu, TexInfo* tex, u32 dstpaddr, u32 size,
                               u32 pitch, u32 gap) {
    if ((tex->fmt != 12 && tex->fmt != 13) || !pitch || size % pitch) {
        g_partial_fail[1]++;
        return false;
    }
    u32 tilebytes = tex->fmt == 12 ? 32 : 64;
    if (pitch % tilebytes || dstpaddr < tex->paddr) {
        g_partial_fail[2]++;
        return false;
    }
    u32 off = dstpaddr - tex->paddr;
    u32 th = size / pitch;
    for (int l = tex->minlod; l <= tex->maxlod; l++) {
        u32 lw = tex->width >> l, lh = tex->height >> l;
        u32 levelsize = lw * lh * (tex->fmt == 12 ? 4 : 8) / 8;
        if (off >= levelsize) {
            off -= levelsize;
            continue;
        }
        u32 rowbytes = lw / 8 * tilebytes;
        if (pitch + gap != rowbytes || off % tilebytes) {
            g_partial_fail[2]++;
            return false;
        }
        u32 tx = (off % rowbytes) / tilebytes, ty = off / rowbytes;
        u32 tw = pitch / tilebytes;
        if (tx + tw > lw / 8 || (ty + th) * 8 > lh) {
            g_partial_fail[2]++;
            return false;
        }
        u8* row = PTR(dstpaddr);
        glBindTexture(GL_TEXTURE_2D, tex->tex);
        gls_textures_unknown();
        if (!tex_upload_tiles(tex, l, lh, tx, ty, tw, th, row, pitch, rowbytes)) {
            g_partial_fail[2]++;
            return false;
        }
        if (tex->shadow && tex->shadow_size == tex->size) {
            u32 soff = dstpaddr - tex->paddr;
            for (u32 i = 0; i < th; i++) {
                memcpy(tex->shadow + soff + i * rowbytes, row + i * rowbytes, pitch);
            }
        }
        tex->hash = gpu_hash_texture(PTR(tex->paddr), tex->size);
        tex->needs_rehash = false;
        tex->content = XXH3_64bits(PTR(tex->paddr), tex->size);
        tex->stale = false;
        g_tex_partial++;
        return true;
    }
    return false;
}

void gpu_gl_texture_copy(GPU* gpu, u32 srcpaddr, u32 dstpaddr, u32 size,
                         u32 srcpitch, u32 srcgap, u32 dstpitch, u32 dstgap) {

    if (!srcpitch || !dstpitch) return; // why

    auto srcfb = gpu_fbcache_find_within(gpu, srcpaddr);
    auto dsttex = gpu_texcache_find_within(gpu, dstpaddr);

    linfo("texture copy from %x to %x size=%d sp=%d sg=%d dp=%d dg=%d",
          srcpaddr, dstpaddr, size, srcpitch, srcgap, dstpitch, dstgap);

    {
        static int logged;
        if (logged < 40) {
            logged++;
            fprintf(stderr, "texcopy %x->%x size %u pitch %u/%u gap %u/%u srcfb %d dsttex %d\n",
                    srcpaddr, dstpaddr, size, srcpitch, dstpitch, srcgap, dstgap, srcfb != nullptr,
                    dsttex != nullptr);
        }
    }
    if (srcfb && dsttex) {
        // do a hardware copy

        linfo("copying from fb at %x to texture at %x", srcfb->color_paddr,
              dsttex->paddr);

        // need to handle more general cases at some point

        if (srcgap == 0 || srcgap == dstgap || srcpitch == dstpitch) {
            int yoff = srcpaddr - srcfb->color_paddr + dsttex->paddr - dstpaddr;
            yoff /= (int) (srcfb->width * srcfb->color_Bpp);

            linfo("hardware texture copy sh=%d sw=%d yof=%d dh=%d dw=%d",
                  srcfb->height, srcfb->width, yoff, dsttex->height,
                  dsttex->width);

            glBindFramebuffer(GL_READ_FRAMEBUFFER, srcfb->fbo);
            glBindTexture(GL_TEXTURE_2D, dsttex->tex);
            gls_textures_unknown();
            dsttex->etcnative = false;
            tex_shadow_drop(dsttex);
            glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0,
                             (srcfb->height - dsttex->height - yoff) *
                                 ctremu.videoscale,
                             dsttex->width * ctremu.videoscale,
                             dsttex->height * ctremu.videoscale, 0);
        } else {
            lwarnonce("unhandled texture copy case sp=%d sg=%d dp=%d dg=%d",
                      srcpitch, srcgap, dstpitch, dstgap);
        }

        return;
    }

    if (srcfb) {
        lwarnonce("reading back fb into memory");
        // this case requires us to read the framebuffer back to ram
        // it is very slow
        return;
    }

    u8* src = PTR(srcpaddr);
    u8* dst = PTR(dstpaddr);
    u8* dst0 = dst;

    int cnt = 0;
    int curline = 0;
    if (srcpitch <= dstpitch) {
        while (cnt < size) {
            memcpy(dst, src, srcpitch);
            cnt += srcpitch;
            curline += srcpitch;
            src += srcpitch + srcgap;
            dst += srcpitch;
            if (curline >= dstpitch) {
                dst += dstgap;
                curline = 0;
            }
        }
    } else {
        while (cnt < size) {
            memcpy(dst, src, dstpitch);
            cnt += dstpitch;
            curline += dstpitch;
            dst += dstpitch + dstgap;
            src += dstpitch;
            if (curline >= srcpitch) {
                src += srcgap;
                curline = 0;
            }
        }
    }

    if (dsttex && !getenv("SPIKE_NOPARTIAL") && srcpitch == dstpitch &&
        tex_partial_update(gpu, dsttex, dstpaddr, size, dstpitch, dstgap)) {
        return;
    }
    if (!dsttex) g_partial_fail[3]++;
    else if (srcpitch != dstpitch) g_partial_fail[0]++;

    // the written span is longer than size when the destination has gaps; the tile diff
    // compares only what is marked, so mark all of it (coins from blocks drew wrong without)
    u32 written = dst - dst0;
    gpu_invalidate_range(gpu, dstpaddr, written > size ? written : size);
}

void gpu_gl_clear_fb(GPU* gpu, u32 paddr, u32 endPaddr, u32 value, u32 datasz) {
    // some of the current gl state can affect gl clear
    // so we need to reset it
    gls_invalidate();
    glDisable(GL_SCISSOR_TEST);
    glColorMask(true, true, true, true);
    glDepthMask(true);
    glStencilMask(0xff);
    bool foundDb = false;
    for (int i = 0; i < FB_MAX; i++) {
        if (gpu->fbs.d[i].color_paddr == paddr) {
            LRU_use(gpu->fbs, &gpu->fbs.d[i]);
            gpu->curfb = &gpu->fbs.d[i];
            gpu->curfb->dirty = true;

            float r = 0, g = 0, b = 0, a = 1;
            switch (gpu->curfb->color_fmt) {
                case 0:
                    a = (value & 0xff) / 255.f;
                    b = (value >> 8 & 0xff) / 255.f;
                    g = (value >> 16 & 0xff) / 255.f;
                    r = (value >> 24 & 0xff) / 255.f;
                    break;
                case 1:
                    b = (value & 0xff) / 255.f;
                    g = (value >> 8 & 0xff) / 255.f;
                    r = (value >> 16 & 0xff) / 255.f;
                    break;
                case 2:
                    b = (value & 0x1f) / 31.f;
                    g = (value >> 5 & 0x3f) / 63.f;
                    r = (value >> 11 & 0x1f) / 31.f;
                    break;
                case 3:
                    a = value & 1;
                    b = (value >> 1 & 0x1f) / 31.f;
                    g = (value >> 6 & 0x1f) / 31.f;
                    r = (value >> 15 & 0x1f) / 31.f;
                    break;
                case 4:
                    a = (value & 0xf) / 15.f;
                    b = (value >> 4 & 0xf) / 15.f;
                    g = (value >> 8 & 0xf) / 15.f;
                    r = (value >> 12 & 0xf) / 15.f;
                    break;
            }

            glBindFramebuffer(GL_FRAMEBUFFER, gpu->fbs.d[i].fbo);
            glClearColor(r, g, b, a);
            glClear(GL_COLOR_BUFFER_BIT);
            if (gpu->curfb->shadowMap) {
                glClearDepth(r);
                glClear(GL_DEPTH_BUFFER_BIT);
            }
            linfo("cleared color buffer at %x of fb %d with value %x", paddr, i,
                  value);
            return;
        }
        if (gpu->fbs.d[i].depth_paddr == paddr) {
            LRU_use(gpu->fbs, &gpu->fbs.d[i]);
            gpu->curfb = &gpu->fbs.d[i];
            glBindFramebuffer(GL_FRAMEBUFFER, gpu->fbs.d[i].fbo);
            // todo handle depth formats
            glClearDepth((value & MASK(24)) / (float) BIT(24));
            glClearStencil(value >> 24);
            glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
            linfo("cleared depth buffer at %x of fb %d with value %x", paddr, i,
                  value);
            // dont return size multiple fbs can have the same db
            foundDb = true;
        }
    }

    if (foundDb) return;

    // fallback to sw memfill if no fbs were filled

    linfo("sw memfill at %x to %x value %x datasz %d", paddr, endPaddr, value,
          datasz);

    void* cur = PTR(paddr);
    void* end = PTR(paddr) + endPaddr - paddr;
    switch (datasz) {
        case 2:
            while (cur < end) {
                *(u16*) cur = value;
                cur += 2;
            }
            break;
        case 3:
            while (cur < end) {
                *(u16*) cur = value;
                *(u8*) (cur + 2) = value >> 16;
                cur += 3;
            }
            break;
        case 4:
            while (cur < end) {
                *(u32*) cur = value;
                cur += 4;
            }
            break;
    }

    gpu_invalidate_range(gpu, paddr, endPaddr - paddr);
}

// where each pixel of an 8x8 tile sits, by row and column
static const u8 tile_order[8][8] = {
    {0, 1, 4, 5, 16, 17, 20, 21},     {2, 3, 6, 7, 18, 19, 22, 23},
    {8, 9, 12, 13, 24, 25, 28, 29},   {10, 11, 14, 15, 26, 27, 30, 31},
    {32, 33, 36, 37, 48, 49, 52, 53}, {34, 35, 38, 39, 50, 51, 54, 55},
    {40, 41, 44, 45, 56, 57, 60, 61}, {42, 43, 46, 47, 58, 59, 62, 63},
};

#define LOAD_TEX(t, glfmt, gltype)                                             \
    ({                                                                         \
        t* data = rawdata;                                                     \
                                                                               \
        t pixels[w * h];                                                       \
                                                                               \
        for (int ty = 0; ty < h / 8; ty++) {                                   \
            for (int tx = 0; tx < w / 8; tx++) {                               \
                t* tile = data + (ty * (w / 8) + tx) * 64;                     \
                for (int py = 0; py < 8; py++) {                               \
                    t* row = pixels + (h - 1 - (ty * 8 + py)) * w + tx * 8;    \
                    const u8* order = tile_order[py];                          \
                    for (int px = 0; px < 8; px++) row[px] = tile[order[px]];  \
                }                                                              \
            }                                                                  \
        }                                                                      \
                                                                               \
        glTexImage2D(GL_TEXTURE_2D, level, glfmt, w, h, 0, glfmt, gltype,      \
                     pixels);                                                  \
    })

static void* expand_nibbles(u8* src, u32 count, u8* dst) {
    for (int i = 0; i < count; i++) {
        u8 b = src[i / 2];
        if (i & 1) b >>= 4;
        else b &= 0xf;
        b *= 0x11;
        dst[i] = b;
    }
    return dst;
}

typedef struct {
    u8 d[3];
} u24;

static const GLint texswizzle_default[4] = {GL_RED, GL_GREEN, GL_BLUE,
                                            GL_ALPHA};
static const GLint texswizzle_bgr[4] = {GL_BLUE, GL_GREEN, GL_RED, GL_ALPHA};
static const GLint texswizzle_lum_alpha[4] = {GL_GREEN, GL_GREEN, GL_GREEN,
                                              GL_RED};
static const GLint texswizzle_luminance[4] = {GL_RED, GL_RED, GL_RED, GL_ONE};
static const GLint texswizzle_alpha[4] = {GL_ZERO, GL_ZERO, GL_ZERO, GL_RED};
static const GLint texswizzle_d24s8[4] = {GL_RED, GL_ALPHA, GL_BLUE, GL_GREEN};

[[maybe_unused]] static const GLint texswizzle_dbg_red[4] = {GL_ONE, GL_ZERO,
                                                             GL_ZERO, GL_ALPHA};
[[maybe_unused]] static const GLint texswizzle_dbg_green[4] = {
    GL_ZERO, GL_ONE, GL_ZERO, GL_ALPHA};
[[maybe_unused]] static const GLint texswizzle_dbg_blue[4] = {GL_ZERO, GL_ZERO,
                                                              GL_ONE, GL_ALPHA};
[[maybe_unused]] static const GLint texswizzle_zero[4] = {GL_ZERO, GL_ZERO,
                                                          GL_ZERO, GL_ZERO};

static const int texfmtbpp[16] = {
    32, 24, 16, 16, 16, 16, 16, 8, 8, 8, 4, 4, 4, 8, 0, 0,
};
static const GLint* texfmtswizzle[16] = {
    texswizzle_default,   texswizzle_bgr,       texswizzle_default,
    texswizzle_default,   texswizzle_default,   texswizzle_lum_alpha,
    texswizzle_default,   texswizzle_luminance, texswizzle_alpha,
    texswizzle_lum_alpha, texswizzle_luminance, texswizzle_alpha,
    texswizzle_default,   texswizzle_default,   texswizzle_default,
    texswizzle_default,
};
static const GLenum texminfilter[4] = {
    GL_NEAREST_MIPMAP_NEAREST, GL_LINEAR_MIPMAP_NEAREST,
    GL_NEAREST_MIPMAP_LINEAR, GL_LINEAR_MIPMAP_LINEAR};
static const GLenum texmagfilter[2] = {GL_NEAREST, GL_LINEAR};

static const GLenum proctexfilter[8] = {
    GL_NEAREST,
    GL_LINEAR,
    GL_NEAREST_MIPMAP_NEAREST,
    GL_LINEAR_MIPMAP_NEAREST,
    GL_NEAREST_MIPMAP_LINEAR,
    GL_LINEAR_MIPMAP_LINEAR,
    GL_NEAREST,
    GL_NEAREST,
};

static const GLenum texwrap[4] = {
    GL_CLAMP_TO_EDGE,
    GL_CLAMP_TO_BORDER,
    GL_REPEAT,
    GL_MIRRORED_REPEAT,
};

static const GLenum blend_eq[8] = {
    GL_FUNC_ADD, GL_FUNC_SUBTRACT, GL_FUNC_REVERSE_SUBTRACT,
    GL_MIN,      GL_MAX,           GL_FUNC_ADD,
    GL_FUNC_ADD, GL_FUNC_ADD,
};
static const GLenum blend_func[16] = {
    GL_ZERO,
    GL_ONE,
    GL_SRC_COLOR,
    GL_ONE_MINUS_SRC_COLOR,
    GL_DST_COLOR,
    GL_ONE_MINUS_DST_COLOR,
    GL_SRC_ALPHA,
    GL_ONE_MINUS_SRC_ALPHA,
    GL_DST_ALPHA,
    GL_ONE_MINUS_DST_ALPHA,
    GL_CONSTANT_COLOR,
    GL_ONE_MINUS_CONSTANT_COLOR,
    GL_CONSTANT_ALPHA,
    GL_ONE_MINUS_CONSTANT_ALPHA,
    GL_SRC_ALPHA_SATURATE,
    GL_ZERO,
};
static const GLenum logic_ops[16] = {
    GL_CLEAR,         GL_AND,  GL_AND_REVERSE, GL_COPY,         GL_SET,
    GL_COPY_INVERTED, GL_NOOP, GL_INVERT,      GL_NAND,         GL_OR,
    GL_NOR,           GL_XOR,  GL_EQUIV,       GL_AND_INVERTED, GL_OR_REVERSE,
    GL_OR_INVERTED,
};
static const GLenum compare_func[8] = {
    GL_NEVER, GL_ALWAYS, GL_EQUAL,   GL_NOTEQUAL,
    GL_LESS,  GL_LEQUAL, GL_GREATER, GL_GEQUAL,
};
static const GLenum stencil_op[8] = {
    GL_KEEP, GL_ZERO,   GL_REPLACE,   GL_INCR,
    GL_DECR, GL_INVERT, GL_INCR_WRAP, GL_DECR_WRAP,
};

#define TEXSIZE(w, h, fmt, level)                                              \
    (((w) >> (level)) * ((h) >> (level)) * texfmtbpp[fmt] / 8)

// including all mip levels
static inline u32 texsize_total(TexUnitRegs* regs, u32 fmt) {
    u32 size = 0;
    for (int i = regs->lod.min; i <= regs->lod.max; i++) {
        size += TEXSIZE(regs->width, regs->height, fmt, i);
    }
    return size;
}


void load_tex_image(void* rawdata, int w, int h, int level, int fmt) {
    w >>= level;
    h >>= level;
    switch (fmt) {
        case 0: // rgba8888
            LOAD_TEX(u32, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8);
            break;
        case 1: // rgb888
            LOAD_TEX(u24, GL_RGB, GL_UNSIGNED_BYTE);
            break;
        case 2: // rgba5551
            LOAD_TEX(u16, GL_RGBA, GL_UNSIGNED_SHORT_5_5_5_1);
            break;
        case 3: // rgb565
            LOAD_TEX(u16, GL_RGB, GL_UNSIGNED_SHORT_5_6_5);
            break;
        case 4: // rgba4444
            LOAD_TEX(u16, GL_RGBA, GL_UNSIGNED_SHORT_4_4_4_4);
            break;
        case 5: // ia88
            LOAD_TEX(u16, GL_RG, GL_UNSIGNED_BYTE);
            break;
        case 6: // hilo8 (rg88)
            LOAD_TEX(u16, GL_RG, GL_UNSIGNED_BYTE);
            break;
        case 7: // i8
            LOAD_TEX(u8, GL_RED, GL_UNSIGNED_BYTE);
            break;
        case 8: // a8
            LOAD_TEX(u8, GL_RED, GL_UNSIGNED_BYTE);
            break;
        case 9: { // ia44
            u8 dec[2 * w * h];
            rawdata = expand_nibbles(rawdata, 2 * w * h, dec);
            LOAD_TEX(u16, GL_RG, GL_UNSIGNED_BYTE);
            break;
        }
        case 10: { // i4
            u8 dec[w * h];
            rawdata = expand_nibbles(rawdata, w * h, dec);
            LOAD_TEX(u8, GL_RED, GL_UNSIGNED_BYTE);
            break;
        }
        case 11: { // a4
            u8 dec[w * h];
            rawdata = expand_nibbles(rawdata, w * h, dec);
            LOAD_TEX(u8, GL_RED, GL_UNSIGNED_BYTE);
            break;
        }
        case 12: { // etc1
            if (g_etc_this_tex) {
                u64* blocks = malloc(w * h / 2);
                int clamped = 0;
                etc1_to_gpu(w, h, rawdata, 1, blocks, &clamped);
                g_etc_clamped += clamped;
                g_etc_blocks += w * h / 16;
                glCompressedTexImage2D(GL_TEXTURE_2D, level, GL_COMPRESSED_RGB8_ETC2, w, h, 0,
                                       w * h / 2, blocks);
                free(blocks);
                break;
            }
            u8 dec[h * w * 3];
            etc1_decompress_texture(w, h, rawdata, (void*) dec);
            glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA, w, h, 0, GL_RGB,
                         GL_UNSIGNED_BYTE, dec);
            break;
        }
        case 13: { // etc1a4
            u8 dec[h * w * 4];
            etc1a4_decompress_texture(w, h, rawdata, (void*) dec);
            glTexImage2D(GL_TEXTURE_2D, level, GL_RGBA, w, h, 0, GL_RGBA,
                         GL_UNSIGNED_BYTE, dec);
            break;
        }
        default:
            lerror("unknown texture format %d", fmt);
    }
}

int g_tex_creates, g_tex_partial, g_draws, g_tex_same, g_tex_changed, g_tex_again, g_draws_lit;
// spike: where texture time goes, printed at exit
struct { int n; u64 pixels, ns, ns_sync, ns_max; } g_texstat[16];
void gpu_gl_print_texstat() {
    extern int g_etc_clamped, g_etc_blocks, g_etc_refused;
    fprintf(stderr, "spike etc blocks %d clamped %d refused %d\n", g_etc_blocks, g_etc_clamped,
            g_etc_refused);
    spike_logf("spike stale textures: same %d changed %d of which seen before %d\n", g_tex_same,
               g_tex_changed, g_tex_again);
    spike_logf("spike tex diffs %d tiles %d ms %.0f gave up %d; copies not partial: pitch %d "
               "format %d geometry %d no texture %d; shadows %.1f MB\n",
               g_tex_diffs_all, g_tex_difftiles_all, g_ns_texdiff_all / 1e6, g_tex_difffull,
               g_partial_fail[0], g_partial_fail[1], g_partial_fail[2], g_partial_fail[3],
               g_shadow_bytes / 1048576.0);
    for (int i = 0; i < 16; i++) {
        if (!g_texstat[i].n) continue;
        spike_logf("spike tex fmt %d: n %d pixels %llu ms %.0f sync %.0f max %.1f\n", i,
                g_texstat[i].n, (unsigned long long) g_texstat[i].pixels, g_texstat[i].ns / 1e6,
                g_texstat[i].ns_sync / 1e6, g_texstat[i].ns_max / 1e6);
    }
}

static void create_texture_real(GPU* gpu, TexInfo* tex, TexUnitRegs* regs);
static void create_texture(GPU* gpu, TexInfo* tex, TexUnitRegs* regs) {
    g_tex_creates++;
    static int syncstat = -1;
    if (syncstat < 0) syncstat = getenv("SPIKE_TEXSYNC") != nullptr;
    u64 t0 = diag_now();
    if (syncstat) {
        glGetError();
        g_texstat[tex->fmt & 15].ns_sync += diag_now() - t0;
    }
    create_texture_real(gpu, tex, regs);
    tex_shadow_refresh(gpu, tex);
    u64 dt = diag_now() - t0;
    g_ns_textures += dt;
    static int texlog = -1;
    if (texlog < 0) texlog = getenv("SPIKE_TEXLOG") != nullptr;
    if (texlog) {
        static const char* why[] = {"new", "stale", "rehash", "diff-gave-up"};
        spike_logf("tex create %08x fmt %d %dx%d lod %d-%d %s %.1f ms\n", tex->paddr, tex->fmt,
                   tex->width, tex->height, tex->minlod, tex->maxlod, why[g_create_reason & 3],
                   dt / 1e6);
    }
    g_create_reason = 0;
    typeof(&g_texstat[0]) st = &g_texstat[tex->fmt & 15];
    st->n++;
    st->pixels += tex->width * tex->height;
    st->ns += dt;
    if (dt > st->ns_max) st->ns_max = dt;
}
static void create_texture_real(GPU* gpu, TexInfo* tex, TexUnitRegs* regs) {
    linfo("creating texture from %x with dims %dx%d and fmt=%d", tex->paddr,
          tex->width, tex->height, tex->fmt);

    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, regs->lod.max);

    glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA,
                     texfmtswizzle[tex->fmt]);

    // mipmap images are stored adjacent in memory and each image is
    // half the width and height of the previous one
    void* rawdata = PTR(tex->paddr);

    // all levels one way or the other
    g_etc_this_tex = false;
    if (tex->fmt == 12 && g_native_etc) {
        g_etc_this_tex = true;
        void* p = rawdata;
        for (int l = regs->lod.min; l <= regs->lod.max && g_etc_this_tex; l++) {
            u32 blocks = (tex->width >> l) * (tex->height >> l) / 16;
            u64* b = p;
            for (u32 i = 0; i < blocks; i++) {
                if (!etc1_block_is_plain(b[i])) {
                    g_etc_this_tex = false;
                    g_etc_refused++;
                    break;
                }
            }
            p += TEXSIZE(tex->width, tex->height, tex->fmt, l);
        }
    }
    tex->etcnative = g_etc_this_tex;

    for (int l = regs->lod.min; l <= regs->lod.max; l++) {
        load_tex_image(rawdata, tex->width, tex->height, l, tex->fmt);
        rawdata += TEXSIZE(tex->width, tex->height, tex->fmt, l);
    }

    // the driver is asked once whether it takes these at all
    static bool etc_checked;
    if (tex->etcnative && !etc_checked) {
        etc_checked = true;
        GLenum err = glGetError();
        if (err != GL_NO_ERROR) {
            fprintf(stderr, "spike native etc refused: %x\n", err);
            g_native_etc = false;
            tex->etcnative = false;
            g_etc_this_tex = false;
            rawdata = PTR(tex->paddr);
            for (int l = regs->lod.min; l <= regs->lod.max; l++) {
                load_tex_image(rawdata, tex->width, tex->height, l, tex->fmt);
                rawdata += TEXSIZE(tex->width, tex->height, tex->fmt, l);
            }
        }
    }
}


// ---- spike: GL state cache for the draw path ----
// Every GL call is marshalled to Mesa's driver thread and unmarshalled there, about 40 a
// draw, most of them setting what is already set. The draw path goes through these setters,
// which skip a call when the value is the one last sent. Anything else that touches the same
// state on this context invalidates the cache (frame start, transfers, clears, uploads).
#define GLS_CAPS 8
static const GLenum gls_capv[GLS_CAPS] = {GL_CULL_FACE,    GL_SCISSOR_TEST, GL_DEPTH_TEST,
                                          GL_BLEND,        GL_COLOR_LOGIC_OP, GL_STENCIL_TEST,
                                          GL_POLYGON_OFFSET_FILL, GL_DITHER};
static struct {
    u8 cap[GLS_CAPS]; // 0 off, 1 on, 2 unknown
    bool cull_k, vp_k, sc_k, cmask_k, dmask_k, dfunc_k, blendf_k, blende_k, blendc_k, logic_k,
        sfunc_k, sop_k, smask_k, vao_k, abuf_k, ebuf_k, active_k;
    GLenum cull, dfunc, logic, bsrc_rgb, bdst_rgb, bsrc_a, bdst_a, beq_rgb, beq_a, sfunc, sop[3];
    GLint vp[4], sc[4], sref;
    GLuint smaskv, swmask, vao, abuf, ebuf;
    GLboolean cmask[4], dmask;
    float bcol[4];
    int active;
    struct { GLenum target; GLuint name; bool k; } unit[8];
    struct { GLint size; GLenum type; GLsizei stride; const void* ptr; GLuint buf; bool k; u8 en; } attr[16];
    struct { GLuint buf; GLintptr off; GLsizeiptr size; bool k; } ubo[4];
    bool on;
} g_gls;
int g_gls_skipped, g_gls_calls;

static void vab_invalidate(void);
static void gls_invalidate(void) {
    memset(&g_gls, 0, sizeof g_gls);
    vab_invalidate();
    for (int i = 0; i < GLS_CAPS; i++) g_gls.cap[i] = 2;
    for (int i = 0; i < 16; i++) g_gls.attr[i].en = 2;
    static int on = -1;
    if (on < 0) on = !getenv("SPIKE_NOGLCACHE");
    g_gls.on = on;
}
static void gls_textures_unknown(void) {
    for (int i = 0; i < 8; i++) g_gls.unit[i].k = false;
    g_gls.active_k = false;
}
static inline int gls_capi(GLenum cap) {
    for (int i = 0; i < GLS_CAPS; i++) if (gls_capv[i] == cap) return i;
    return -1;
}
static inline void gls_Enable(GLenum cap) {
    int i = gls_capi(cap);
    g_gls_calls++;
    if (g_gls.on && i >= 0 && g_gls.cap[i] == 1) { g_gls_skipped++; return; }
    glEnable(cap);
    if (i >= 0) g_gls.cap[i] = 1;
}
static inline void gls_Disable(GLenum cap) {
    int i = gls_capi(cap);
    g_gls_calls++;
    if (g_gls.on && i >= 0 && g_gls.cap[i] == 0) { g_gls_skipped++; return; }
    glDisable(cap);
    if (i >= 0) g_gls.cap[i] = 0;
}
#define GLS_SET1(name, field, kfield, call)                                            \
    g_gls_calls++;                                                                      \
    if (g_gls.on && g_gls.kfield && g_gls.field == (name)) { g_gls_skipped++; return; } \
    call;                                                                               \
    g_gls.field = (name);                                                               \
    g_gls.kfield = true;
static inline void gls_CullFace(GLenum m) { GLS_SET1(m, cull, cull_k, glCullFace(m)) }
static inline void gls_DepthFunc(GLenum f) { GLS_SET1(f, dfunc, dfunc_k, glDepthFunc(f)) }
static inline void gls_LogicOp(GLenum f) { GLS_SET1(f, logic, logic_k, glLogicOp(f)) }
static inline void gls_DepthMask(GLboolean b) { GLS_SET1(b, dmask, dmask_k, glDepthMask(b)) }
static inline void gls_StencilMask(GLuint m) { GLS_SET1(m, swmask, smask_k, glStencilMask(m)) }
static inline void gls_BindVertexArray(GLuint v) {
    g_gls_calls++;
    if (g_gls.on && g_gls.vao_k && g_gls.vao == v) { g_gls_skipped++; return; }
    glBindVertexArray(v);
    g_gls.vao = v;
    g_gls.vao_k = true;
    // the element buffer binding and the attribute arrays belong to the vertex array object
    g_gls.ebuf_k = false;
    for (int i = 0; i < 16; i++) { g_gls.attr[i].k = false; g_gls.attr[i].en = 2; }
    vab_invalidate();
}
static inline void gls_Viewport(GLint x, GLint y, GLint w, GLint h) {
    GLint v[4] = {x, y, w, h};
    g_gls_calls++;
    if (g_gls.on && g_gls.vp_k && !memcmp(v, g_gls.vp, sizeof v)) { g_gls_skipped++; return; }
    glViewport(x, y, w, h);
    memcpy(g_gls.vp, v, sizeof v);
    g_gls.vp_k = true;
}
static inline void gls_Scissor(GLint x, GLint y, GLint w, GLint h) {
    GLint v[4] = {x, y, w, h};
    g_gls_calls++;
    if (g_gls.on && g_gls.sc_k && !memcmp(v, g_gls.sc, sizeof v)) { g_gls_skipped++; return; }
    glScissor(x, y, w, h);
    memcpy(g_gls.sc, v, sizeof v);
    g_gls.sc_k = true;
}
static inline void gls_ColorMask(GLboolean r, GLboolean g, GLboolean b, GLboolean a) {
    GLboolean v[4] = {r, g, b, a};
    g_gls_calls++;
    if (g_gls.on && g_gls.cmask_k && !memcmp(v, g_gls.cmask, sizeof v)) { g_gls_skipped++; return; }
    glColorMask(r, g, b, a);
    memcpy(g_gls.cmask, v, sizeof v);
    g_gls.cmask_k = true;
}
static inline void gls_BlendFuncSeparate(GLenum a, GLenum b, GLenum c, GLenum d) {
    g_gls_calls++;
    if (g_gls.on && g_gls.blendf_k && g_gls.bsrc_rgb == a && g_gls.bdst_rgb == b && g_gls.bsrc_a == c &&
        g_gls.bdst_a == d) { g_gls_skipped++; return; }
    glBlendFuncSeparate(a, b, c, d);
    g_gls.bsrc_rgb = a; g_gls.bdst_rgb = b; g_gls.bsrc_a = c; g_gls.bdst_a = d;
    g_gls.blendf_k = true;
}
static inline void gls_BlendEquationSeparate(GLenum a, GLenum b) {
    g_gls_calls++;
    if (g_gls.on && g_gls.blende_k && g_gls.beq_rgb == a && g_gls.beq_a == b) { g_gls_skipped++; return; }
    glBlendEquationSeparate(a, b);
    g_gls.beq_rgb = a; g_gls.beq_a = b;
    g_gls.blende_k = true;
}
static inline void gls_BlendColor(float r, float g, float b, float a) {
    float v[4] = {r, g, b, a};
    g_gls_calls++;
    if (g_gls.on && g_gls.blendc_k && !memcmp(v, g_gls.bcol, sizeof v)) { g_gls_skipped++; return; }
    glBlendColor(r, g, b, a);
    memcpy(g_gls.bcol, v, sizeof v);
    g_gls.blendc_k = true;
}
static inline void gls_StencilFunc(GLenum f, GLint ref, GLuint mask) {
    g_gls_calls++;
    if (g_gls.on && g_gls.sfunc_k && g_gls.sfunc == f && g_gls.sref == ref && g_gls.smaskv == mask) { g_gls_skipped++; return; }
    glStencilFunc(f, ref, mask);
    g_gls.sfunc = f; g_gls.sref = ref; g_gls.smaskv = mask;
    g_gls.sfunc_k = true;
}
static inline void gls_StencilOp(GLenum a, GLenum b, GLenum c) {
    g_gls_calls++;
    if (g_gls.on && g_gls.sop_k && g_gls.sop[0] == a && g_gls.sop[1] == b && g_gls.sop[2] == c) { g_gls_skipped++; return; }
    glStencilOp(a, b, c);
    g_gls.sop[0] = a; g_gls.sop[1] = b; g_gls.sop[2] = c;
    g_gls.sop_k = true;
}
static inline void gls_ActiveTexture(GLenum u) {
    int i = u - GL_TEXTURE0;
    g_gls_calls++;
    if (g_gls.on && g_gls.active_k && g_gls.active == i) { g_gls_skipped++; return; }
    glActiveTexture(u);
    g_gls.active = i;
    g_gls.active_k = true;
}
static inline void gls_BindTexture(GLenum target, GLuint name) {
    int i = g_gls.active_k ? g_gls.active : -1;
    g_gls_calls++;
    if (g_gls.on && i >= 0 && i < 8 && g_gls.unit[i].k && g_gls.unit[i].target == target &&
        g_gls.unit[i].name == name) { g_gls_skipped++; return; }
    glBindTexture(target, name);
    if (i >= 0 && i < 8) {
        g_gls.unit[i].target = target;
        g_gls.unit[i].name = name;
        g_gls.unit[i].k = true;
    }
}
static inline void gls_BindBuffer(GLenum target, GLuint b) {
    if (target == GL_ARRAY_BUFFER) { GLS_SET1(b, abuf, abuf_k, glBindBuffer(target, b)) }
    else if (target == GL_ELEMENT_ARRAY_BUFFER) { GLS_SET1(b, ebuf, ebuf_k, glBindBuffer(target, b)) }
    else glBindBuffer(target, b);
}
static inline void gls_BindBufferRange(GLenum target, GLuint idx, GLuint b, GLintptr off, GLsizeiptr size) {
    g_gls_calls++;
    if (target == GL_UNIFORM_BUFFER && idx < 4) {
        if (g_gls.on && g_gls.ubo[idx].k && g_gls.ubo[idx].buf == b && g_gls.ubo[idx].off == off &&
            g_gls.ubo[idx].size == size) { g_gls_skipped++; return; }
        g_gls.ubo[idx].buf = b; g_gls.ubo[idx].off = off; g_gls.ubo[idx].size = size;
        g_gls.ubo[idx].k = true;
    }
    glBindBufferRange(target, idx, b, off, size);
}
static inline void gls_EnableVertexAttribArray(GLuint a) {
    g_gls_calls++;
    if (g_gls.on && a < 16 && g_gls.attr[a].en == 1) { g_gls_skipped++; return; }
    glEnableVertexAttribArray(a);
    if (a < 16) g_gls.attr[a].en = 1;
}
static inline void gls_DisableVertexAttribArray(GLuint a) {
    g_gls_calls++;
    if (g_gls.on && a < 16 && g_gls.attr[a].en == 0) { g_gls_skipped++; return; }
    glDisableVertexAttribArray(a);
    if (a < 16) g_gls.attr[a].en = 0;
}
static inline void gls_VertexAttribPointer(GLuint a, GLint size, GLenum type, GLboolean norm, GLsizei stride, const void* ptr) {
    g_gls_calls++;
    GLuint buf = g_gls.abuf_k ? g_gls.abuf : 0xffffffffu;
    if (g_gls.on && a < 16 && g_gls.attr[a].k && g_gls.attr[a].size == size && g_gls.attr[a].type == type &&
        g_gls.attr[a].stride == stride && g_gls.attr[a].ptr == ptr && g_gls.attr[a].buf == buf) { g_gls_skipped++; return; }
    glVertexAttribPointer(a, size, type, norm, stride, ptr);
    if (a < 16) {
        g_gls.attr[a].size = size; g_gls.attr[a].type = type; g_gls.attr[a].stride = stride;
        g_gls.attr[a].ptr = ptr; g_gls.attr[a].buf = buf; g_gls.attr[a].k = true;
    }
}
// ---- end of the state cache ----

static void load_texture(GPU* gpu, FragConfig* fcfg, int id, TexUnitRegs* regs,
                         u32 fmt) {
    // make sure we are binding to the correct texture
    gls_ActiveTexture(GL_TEXTURE0 + id);

    // since null is empty for the caches we need to handle
    // null address before searching any cache
    if (regs->addr == 0) {
        lwarnonce("null texture");
        gls_BindTexture(GL_TEXTURE_2D, gpu->gl.blanktex);
        return;
    }

    u32 texsize = texsize_total(regs, fmt);

    // spike: the cache is a list searched from the front, 256 entries long, and every draw looks
    // its textures up (a twelfth of this thread in Mario Kart 7): a hint by address first
    static u16 hint[4096];
    u64 key = regs->addr << 3;
    u32 h = (u32) (key >> 3) * 2654435761u >> 20;
    TexInfo* tex;
    if (hint[h] && gpu->textures.d[hint[h] - 1].key == key) {
        tex = &gpu->textures.d[hint[h] - 1];
        LRU_use(gpu->textures, tex);
    } else {
        tex = LRU_load(gpu->textures, key);
        hint[h] = tex - gpu->textures.d + 1;
    }
    gls_BindTexture(GL_TEXTURE_2D, tex->tex);

    // textures that are partially out of bounds can still be used with rtt ..?
    if (is_valid_physmem(regs->addr << 3) &&
        is_valid_physmem((regs->addr << 3) + texsize - 1)) {
        // if the attributes are different we obviously need to recreate the
        // texture
        // if they are the same we check if the hash needs to be updated
        // and if it does we get the hash and check if that is equal and
        // recreate when it is not
        // Fewer mip levels than are built is no new texture, only a narrower sampler range below:
        // 3D Land binds the same textures both ways, many times a frame.
        if (tex->paddr != (regs->addr << 3) || tex->width != regs->width ||
            tex->height != regs->height || tex->fmt != fmt ||
            tex->minlod != regs->lod.min || tex->maxlod < regs->lod.max) {
            tex->paddr = regs->addr << 3;
            tex->width = regs->width;
            tex->height = regs->height;
            tex->fmt = fmt;
            tex->minlod = regs->lod.min;
            tex->maxlod = regs->lod.max;
            tex->size = texsize;
            tex->sampler_valid = false;

            void* data = PTR(tex->paddr);
            tex->hash = gpu_hash_texture(data, tex->size);
            tex->needs_rehash = false;
            tex->stale = false;
            tex->content = XXH3_64bits(data, tex->size);

            g_tex_new++;
            g_create_reason = 0;
            create_texture(gpu, tex, regs);
        } else if (tex->stale) {
            tex->stale = false;
            u32 lo = tex->dirty_lo, hi = tex->dirty_hi;
            tex->dirty_lo = tex->dirty_hi = 0;
            // the game also writes textures with no flush at all (coins from blocks in NSMB2), and
            // HashTextures is off, so once a frame the whole texture is compared, not just the range
            if (tex->shadow && tex->checked_frame != g_tex_frame) {
                tex->checked_frame = g_tex_frame;
                lo = 0;
                hi = tex->size;
            }
            int r = tex_diff_update(gpu, tex, lo, hi);
            if (r > 0) {
                g_tex_diffs++;
                g_tex_difftiles += r;
                g_tex_changed++;
                tex->content = XXH3_64bits(PTR(tex->paddr), tex->size);
                tex->hash = gpu_hash_texture(PTR(tex->paddr), tex->size);
                tex->needs_rehash = false;
            } else if (r == 0) {
                g_tex_same++;
            } else {
                u64 content = XXH3_64bits(PTR(tex->paddr), tex->size);
                if (content == tex->content) {
                    g_tex_same++;
                } else {
                    tex->content = content;
                    tex->hash = gpu_hash_texture(PTR(tex->paddr), tex->size);
                    tex->needs_rehash = false;
                    g_tex_changed++;
                    // has this picture been here before?
                    static u64 seen[1 << 14];
                    u32 at = content % (1 << 14);
                    if (seen[at] == content) g_tex_again++;
                    seen[at] = content;
                    if (tex->shadow) g_tex_difffull++;
                    g_tex_stalefull++;
                    g_create_reason = tex->shadow ? 3 : 1;
                    create_texture(gpu, tex, regs);
                }
            }
        } else if (tex->needs_rehash) {
            void* data = PTR(tex->paddr);
            u64 hash = gpu_hash_texture(data, tex->size);
            tex->needs_rehash = false;
            if (hash != tex->hash) {
                tex->hash = hash;
                int r = tex_diff_update(gpu, tex, 0, tex->size);
                if (r >= 0) {
                    if (r > 0) {
                        g_tex_diffs++;
                        g_tex_difftiles += r;
                        tex->content = XXH3_64bits(data, tex->size);
                    }
                } else {
                    g_tex_rehashfull++;
                    g_create_reason = 2;
                    create_texture(gpu, tex, regs);
                }
            }
        }
    } else {
        // just add this to the cache but dont actually send it image data ig
        // we only care because you can still rtt to this .......
        lwarnonce("out of bounds texture");
        tex->paddr = regs->addr << 3;
        tex->width = regs->width;
        tex->height = regs->height;
        tex->fmt = fmt;
        tex->minlod = regs->lod.min;
        tex->maxlod = regs->lod.max;
        tex->size = texsize;
    }

    // handle simple render to texture cases, but better ...?
    FBInfo* fb = LRU_find(gpu->fbs, tex->paddr);
    // look for cases of reading a depth buffer as a texture
    bool copyDepth = false;
    bool copyDepthStencil = false;
    if (!fb) {
        for (int i = 0; i < FB_MAX; i++) {
            auto t = &gpu->fbs.d[i];
            if (t->depth_paddr == tex->paddr) {
                // reinterpret d24s8 to rgba8888 or d24 as rgb888
                if (tex->fmt == 1 && t->depth_fmt == 2) {
                    fb = t;
                    copyDepth = true;
                    break;
                } else if (tex->fmt == 0 && t->depth_fmt == 3) {
                    // reading back both depth and stencil is a bit more work
                    fb = t;
                    copyDepthStencil = true;
                    break;
                } else {
                    lwarnonce("unknown reinterpret depth %d to color %d",
                              t->depth_fmt, tex->fmt);
                }
            }
        }
    }
    if (fb && fb->dirty) {
        // only read from a framebuffer if its been modified since the last time
        fb->dirty = false;
        linfo("rtt at %08x", fb->color_paddr);
        glBindFramebuffer(GL_READ_FRAMEBUFFER, fb->fbo);
        gls_BindTexture(GL_TEXTURE_2D, tex->tex);
        // no swizzling or mipmaps for rtt textures
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA,
                         texswizzle_default);
        if (fb->shadowMap && regs->param.shadow) {
            // the real data we need from a shadow map is the depth, we dont
            // care about color
            // so we copy the depth buffer into the texture so we can then
            // bind it to a sampler2DShadow to sample from
            // notably we need the full 24 bits of depth information
            // so we need to use the depth format
            // previously we copied the depth into the red color in the fs
            // but that drops precision from 24 to 8 bit
            glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 0,
                             (fb->height - tex->height) * ctremu.videoscale,
                             tex->width * ctremu.videoscale,
                             tex->height * ctremu.videoscale, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE,
                            GL_COMPARE_REF_TO_TEXTURE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_FUNC, GL_GREATER);
        } else if (copyDepth) {
            // the below only works for d24 -> rgb8
            // reading a depth texture puts depth in r component
            // games also seems to only read r component when reading back from
            // depth texture, since r gives the top 8 bits of the 24bit depth
            // so this can be emulated by just providing the original depth
            // buffer
            glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT24, 0,
                             (fb->height - tex->height) * ctremu.videoscale,
                             tex->width * ctremu.videoscale,
                             tex->height * ctremu.videoscale, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_COMPARE_MODE, GL_NONE);
        } else if (copyDepthStencil) {
            // on macos at least there is no way to read the stencil
            // part of a texture except for reading back the raw data
            // this is pretty slow so its a setting
            if (ctremu.reinterpretTexture) {
                gls_BindBuffer(GL_PIXEL_PACK_BUFFER, gpu->gl.pbo);
                gls_BindBuffer(GL_PIXEL_UNPACK_BUFFER, gpu->gl.pbo);

                gls_BindTexture(GL_TEXTURE_2D, fb->depth_tex);
                glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_STENCIL,
                              GL_UNSIGNED_INT_24_8, nullptr);
                gls_BindTexture(GL_TEXTURE_2D, tex->tex);
                glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
                             tex->width * ctremu.videoscale,
                             tex->height * ctremu.videoscale, 0, GL_RGBA,
                             GL_UNSIGNED_INT_8_8_8_8_REV, nullptr);

                gls_BindBuffer(GL_PIXEL_PACK_BUFFER, 0);
                gls_BindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
                glTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_SWIZZLE_RGBA,
                                 texswizzle_d24s8);
            }
        } else {
            glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, 0,
                             (fb->height - tex->height) * ctremu.videoscale,
                             tex->width * ctremu.videoscale,
                             tex->height * ctremu.videoscale, 0);
        }
    }

    // spike: every one of these dirties the sampler state, per draw and per unit
    u32 sampler_border, sampler_param, sampler_lod;
    memcpy(&sampler_border, &regs->border, sizeof sampler_border);
    memcpy(&sampler_param, &regs->param, sizeof sampler_param);
    memcpy(&sampler_lod, &regs->lod, sizeof sampler_lod);
    if (tex->sampler_valid && tex->sampler_param == sampler_param &&
        tex->sampler_border == sampler_border) {
        if (tex->sampler_lod == sampler_lod) return;
        tex->sampler_lod = sampler_lod;
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS,
                        (float) ((int) (regs->lod.bias << 19) >> 19) / 256);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, regs->lod.min);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LOD, regs->lod.max);
        return;
    }
    tex->sampler_valid = true;
    tex->sampler_param = sampler_param;
    tex->sampler_border = sampler_border;
    tex->sampler_lod = sampler_lod;

    glTexParameteri(
        GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,
        texminfilter[regs->param.min_filter | regs->param.mipmapfilter << 1]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER,
                    texmagfilter[regs->param.mag_filter]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S,
                    texwrap[regs->param.wrap_s]);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T,
                    texwrap[regs->param.wrap_t]);
    float bordercolor[4];
    COPYRGBA(bordercolor, regs->border);
    glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, bordercolor);
    glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS,
                    (float) ((int) (regs->lod.bias << 19) >> 19) / 256);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, regs->lod.min);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LOD, regs->lod.max);
}

static void load_texenv(FragConfig* fcfg, FragUniforms* fbuf, int i,
                        TexEnvRegs* regs) {
    fcfg->tev[i].rgb.src0 = regs->source.rgb0;
    fcfg->tev[i].rgb.src1 = regs->source.rgb1;
    fcfg->tev[i].rgb.src2 = regs->source.rgb2;
    fcfg->tev[i].a.src0 = regs->source.a0;
    fcfg->tev[i].a.src1 = regs->source.a1;
    fcfg->tev[i].a.src2 = regs->source.a2;
    fcfg->tev[i].rgb.op0 = regs->operand.rgb0;
    fcfg->tev[i].rgb.op1 = regs->operand.rgb1;
    fcfg->tev[i].rgb.op2 = regs->operand.rgb2;
    fcfg->tev[i].a.op0 = regs->operand.a0;
    fcfg->tev[i].a.op1 = regs->operand.a1;
    fcfg->tev[i].a.op2 = regs->operand.a2;
    fcfg->tev[i].rgb.combiner = regs->combiner.rgb;
    fcfg->tev[i].a.combiner = regs->combiner.a;
    COPYRGBA(fbuf->tev_color[i], regs->color);
    fcfg->tev[i].rgb.scale = 1 << (regs->scale.rgb);
    fcfg->tev[i].a.scale = 1 << (regs->scale.a);
}

static const GLuint attrtypes[] = {
    GL_BYTE,
    GL_UNSIGNED_BYTE,
    GL_SHORT,
    GL_FLOAT,
};

static void setup_fixattrs_hw(GPU* gpu) {
    for (int i = 0; i < 12; i++) {
        if (gpu->regs.geom.fixed_attr_mask & BIT(i)) {
            int attr = (gpu->regs.vsh.permutation >> 4 * i) & 0xf;
            // spike: each of these flushes Mesa's immediate-mode vertex state
            static float current[16][4];
            if (memcmp(current[attr], gpu->fixattrs[i], sizeof current[0])) {
                memcpy(current[attr], gpu->fixattrs[i], sizeof current[0]);
                glVertexAttrib4fv(attr, gpu->fixattrs[i]);
            }
            gls_DisableVertexAttribArray(attr);
        }
    }
}

// spike: with ARB_vertex_attrib_binding the vertex data's place in the ring is one
// glBindVertexBuffer per buffer a draw, and the attribute formats (set per attribute, and the
// same from draw to draw) are sent only when they change. Measured 2026-11-03 with the
// extension forced on (MESA_EXTENSION_OVERRIDE=+GL_ARB_vertex_attrib_binding): fewer calls
// (7.3 against 7.8 a draw) but the driver thread busier (60% against 45%) and more frames
// lost, so it is opt-in: SPIKE_ATTRIBBINDING=1. Without it, glVertexAttribPointer per attribute.
static int g_vab = -1;
static struct { GLint size; GLenum type; GLuint reloff, binding; bool k; } g_vab_fmt[16];
static struct { GLintptr off; GLsizei stride; GLuint buf; bool k; } g_vab_buf[12];
static void vab_invalidate(void) {
    memset(g_vab_fmt, 0, sizeof g_vab_fmt);
    memset(g_vab_buf, 0, sizeof g_vab_buf);
}
// draws whose vertex buffers the main thread laid out for the driver (VboLayout in
// renderer_gl.h): `relaid` in the stats line
int g_draws_relaid;

static void setup_hw_vao(GPU* gpu, DrawSnap* snap, int nbufverts) {
    setup_fixattrs_hw(gpu);
    if (g_vab < 0) {
        // the loader only knows it as part of GL 4.3, which the 3.3 override hides: ask by name
        GLint n = 0;
        glGetIntegerv(GL_NUM_EXTENSIONS, &n);
        bool ext = false;
        for (GLint i = 0; i < n && !ext; i++) {
            const char* e = (const char*) glGetStringi(GL_EXTENSIONS, i);
            ext = e && !strcmp(e, "GL_ARB_vertex_attrib_binding");
        }
        g_vab = ext && glad_glBindVertexBuffer && glad_glVertexAttribFormat &&
                glad_glVertexAttribBinding && getenv("SPIKE_ATTRIBBINDING");
        fprintf(stderr, "spike vertex attrib binding: %s\n", g_vab ? "on" : "off");
    }
    if (g_vab) {
        for (int vbo = 0; vbo < 12; vbo++) {
            if (gpu->regs.geom.attrbuf[vbo].count == 0) continue;
            VboLayout lay;
            gpu_vbo_layout(&gpu->regs, vbo, &lay);
            for (int i = 0; i < lay.n; i++) {
                int permattr = (gpu->regs.vsh.permutation >> 4 * lay.a[i].attr) & 0xf;
                typeof(&g_vab_fmt[0]) f = &g_vab_fmt[permattr];
                g_gls_calls += 2;
                if (!f->k || f->size != lay.a[i].size || f->type != attrtypes[lay.a[i].type] ||
                    f->reloff != lay.a[i].off || f->binding != (GLuint) vbo) {
                    glVertexAttribFormat(permattr, lay.a[i].size, attrtypes[lay.a[i].type], GL_FALSE,
                                         lay.a[i].off);
                    glVertexAttribBinding(permattr, vbo);
                    f->size = lay.a[i].size;
                    f->type = attrtypes[lay.a[i].type];
                    f->reloff = lay.a[i].off;
                    f->binding = vbo;
                    f->k = true;
                } else {
                    g_gls_skipped += 2;
                }
                gls_EnableVertexAttribArray(permattr);
            }
            typeof(&g_vab_buf[0]) b = &g_vab_buf[vbo];
            g_gls_calls++;
            if (!b->k || b->off != (GLintptr) snap->vbo[vbo] || b->stride != (GLsizei) lay.stride ||
                b->buf != g_vstream) {
                glBindVertexBuffer(vbo, g_vstream, snap->vbo[vbo], lay.stride);
                b->off = snap->vbo[vbo];
                b->stride = lay.stride;
                b->buf = g_vstream;
                b->k = true;
            } else {
                g_gls_skipped++;
            }
        }
        return;
    }
    gls_BindBuffer(GL_ARRAY_BUFFER, g_vstream);

    for (int vbo = 0; vbo < 12; vbo++) {
        // skip unused vbos
        if (gpu->regs.geom.attrbuf[vbo].count == 0) continue;
        // the layout the main thread copied this buffer in (VboLayout)
        VboLayout lay;
        gpu_vbo_layout(&gpu->regs, vbo, &lay);
        for (int i = 0; i < lay.n; i++) {
            int permattr = (gpu->regs.vsh.permutation >> 4 * lay.a[i].attr) & 0xf;
            gls_VertexAttribPointer(permattr, lay.a[i].size, attrtypes[lay.a[i].type], GL_FALSE,
                                  lay.stride, (void*) (size_t) (snap->vbo[vbo] + lay.a[i].off));
            gls_EnableVertexAttribArray(permattr);
        }
    }
}

static void setup_hw_vao_imm(GPU* gpu) {
    setup_fixattrs_hw(gpu);

    int nattrs = gpu->regs.geom.vsh_num_attr + 1;
    // only need to use one vbo
    size_t base = stream_put(GL_ARRAY_BUFFER, gpu->immattrs.d,
                             gpu->immattrs.size * sizeof(fvec4), 16);
    for (int i = 0; i < nattrs; i++) {
        int attr = (gpu->regs.vsh.permutation >> 4 * i) & 0xf;
        gls_VertexAttribPointer(attr, 4, GL_FLOAT, GL_FALSE,
                              nattrs * sizeof(fvec4),
                              (void*) (base + i * sizeof(fvec4)));
        gls_EnableVertexAttribArray(attr);
    }
}

static const GLenum prim_mode[4] = {
    GL_TRIANGLES,
    GL_TRIANGLE_STRIP,
    GL_TRIANGLE_FAN,
    GL_TRIANGLES,
};

static const GLuint indextypes[2] = {GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT};

void gpu_gl_draw(GPU* gpu, bool elements, bool immediate, DrawSnap* snap) {
    g_draws++;
    int nattrs = gpu->regs.geom.vsh_num_attr + 1;
    int nverts =
        immediate ? gpu->immattrs.size / nattrs : gpu->regs.geom.nverts;
    int primMode = gpu->regs.geom.prim_config.mode;

    linfo("drawing %s %s nverts=%d, prim mode=%d",
          elements ? "elements" : "arrays", immediate ? "immediate mode" : "",
          nverts, primMode);

    update_cur_fb(gpu);

    // ensure unused entries are 0 so the hashing is consistent
    FragConfig fcfg = {};
    FragUniforms fbuf;

    // cull mode
    switch (gpu->regs.raster.cullmode) {
        case 0:
            gls_Disable(GL_CULL_FACE);
            break;
        case 1:
            gls_Enable(GL_CULL_FACE);
            gls_CullFace(GL_FRONT);
            break;
        case 2:
            gls_Enable(GL_CULL_FACE);
            gls_CullFace(GL_BACK);
            break;
        case 3:
            gls_Enable(GL_CULL_FACE);
            gls_CullFace(GL_FRONT_AND_BACK);
            break;
    }

    // viewport and scissor
    // since the framebuffer texture may be taller than the current framebuffer,
    // we need to offset the Y value
    int vyoff = gpu->curfb->height - (gpu->regs.fb.dim.height + 1);
    gls_Viewport(gpu->regs.raster.view_x * ctremu.videoscale,
               (gpu->regs.raster.view_y + vyoff) * ctremu.videoscale,
               2 * cvtf24(gpu->regs.raster.view_w) * ctremu.videoscale,
               2 * cvtf24(gpu->regs.raster.view_h) * ctremu.videoscale);
    if (gpu->regs.raster.scissortest.enable) {
        gls_Enable(GL_SCISSOR_TEST);
        gls_Scissor(gpu->regs.raster.scissortest.x1 * ctremu.videoscale,
                  (gpu->regs.raster.scissortest.y1 + vyoff) * ctremu.videoscale,
                  (gpu->regs.raster.scissortest.x2 + 1 -
                   gpu->regs.raster.scissortest.x1) *
                      ctremu.videoscale,
                  (gpu->regs.raster.scissortest.y2 + 1 -
                   gpu->regs.raster.scissortest.y1) *
                      ctremu.videoscale);
    } else {
        gls_Disable(GL_SCISSOR_TEST);
    }

    // depth map
    float dmOffset = cvtf24(gpu->regs.raster.depthmap_offset);
    float dmScale = cvtf24(gpu->regs.raster.depthmap_scale);
    bool wbuffer = !gpu->regs.raster.depthmap_enable;
    // previously we used glDepthRange here, but this cannot properly
    // emulate negative depth offset (probably glPolygonOffset)
    // so we do in vertex shader instead

    // textures
    fcfg.texconfig.w = gpu->regs.tex.config.raw;
    if (gpu->regs.tex.config.tex0enable) {
        load_texture(gpu, &fcfg, 0, &gpu->regs.tex.tex0,
                     gpu->regs.tex.tex0_fmt);
        fcfg.tex0type = gpu->regs.tex.tex0.param.type;
        fcfg.shadowPerspective = !gpu->regs.tex.tex0_shadow.perspective;
        fbuf.shadowBias = (float) gpu->regs.tex.tex0_shadow.bias / BIT(23);
        fbuf.shadowMax = cvtf16(gpu->regs.fb.shadow.max);
        fbuf.shadowRamp = cvtf16(gpu->regs.fb.shadow.ramp);
    }
    if (gpu->regs.tex.config.tex1enable) {
        load_texture(gpu, &fcfg, 1, &gpu->regs.tex.tex1,
                     gpu->regs.tex.tex1_fmt);
    }
    if (gpu->regs.tex.config.tex2enable) {
        load_texture(gpu, &fcfg, 2, &gpu->regs.tex.tex2,
                     gpu->regs.tex.tex2_fmt);
    }

    if (gpu->regs.tex.config.tex3enable) {
        gls_ActiveTexture(GL_TEXTURE3);
        gls_BindTexture(GL_TEXTURE_1D, gpu->proctex.tex);
        if (gpu->proctexLutDirty ||
            gpu->proctex.width != gpu->regs.tex.tex3.paramH.width ||
            gpu->proctex.offset != gpu->regs.tex.tex3.offset[0]) {
            gpu->proctexLutDirty = false;
            gpu->proctex.width = gpu->regs.tex.tex3.paramH.width;
            gpu->proctex.offset = gpu->regs.tex.tex3.offset[0];
            glTexImage1D(GL_TEXTURE_1D, 0, GL_RGBA, gpu->proctex.width, 0,
                         GL_RGBA, GL_UNSIGNED_BYTE,
                         &gpu->proctexLut[gpu->proctex.offset]);
        }
        // todo proctex mipmaps
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_1D, GL_TEXTURE_MIN_FILTER,
                        proctexfilter[gpu->regs.tex.tex3.paramH.minFilter]);
        fcfg.proctex.w = gpu->regs.tex.tex3.paramL.w;

        gls_ActiveTexture(GL_TEXTURE6);
        gls_BindTexture(GL_TEXTURE_1D, gpu->gl.proctexmaptex);
        if (gpu->proctexMapLutDirty) {
            gpu->proctexMapLutDirty = false;
            glTexSubImage1D(GL_TEXTURE_1D, 0, 0, countof(gpu->proctexMapLut),
                            GL_RG, GL_UNSIGNED_SHORT, gpu->proctexMapLut);
        }
        gls_ActiveTexture(GL_TEXTURE7);
        gls_BindTexture(GL_TEXTURE_1D, gpu->gl.proctexnoisetex);
        if (gpu->proctexNoiseLutDirty) {
            gpu->proctexNoiseLutDirty = false;
            glTexSubImage1D(GL_TEXTURE_1D, 0, 0, countof(gpu->proctexNoiseLut),
                            GL_RED, GL_UNSIGNED_SHORT, gpu->proctexNoiseLut);
        }
        fbuf.ptNoiseU.ampl =
            (float) (s16) gpu->regs.tex.tex3.noise.ampU / BIT(12);
        fbuf.ptNoiseU.phase = cvtf16(gpu->regs.tex.tex3.noise.phaseU);
        fbuf.ptNoiseU.freq = cvtf16(gpu->regs.tex.tex3.noise.freqU);
        fbuf.ptNoiseV.ampl =
            (float) (s16) gpu->regs.tex.tex3.noise.ampV / BIT(12);
        fbuf.ptNoiseV.phase = cvtf16(gpu->regs.tex.tex3.noise.phaseV);
        fbuf.ptNoiseV.freq = cvtf16(gpu->regs.tex.tex3.noise.freqV);
    }

    // texenvs
    load_texenv(&fcfg, &fbuf, 0, &gpu->regs.tex.tev0);
    load_texenv(&fcfg, &fbuf, 1, &gpu->regs.tex.tev1);
    load_texenv(&fcfg, &fbuf, 2, &gpu->regs.tex.tev2);
    load_texenv(&fcfg, &fbuf, 3, &gpu->regs.tex.tev3);
    load_texenv(&fcfg, &fbuf, 4, &gpu->regs.tex.tev4);
    load_texenv(&fcfg, &fbuf, 5, &gpu->regs.tex.tev5);
    fcfg.tev_buffer.w = gpu->regs.tex.tev_buffer;
    COPYRGBA(fbuf.tev_buffer_color, gpu->regs.tex.tev5.buffer_color);
    COPYRGB(fbuf.fog_color, gpu->regs.tex.fogColor);

    // blending/logic ops
    fcfg.fragOp = gpu->regs.fb.color_op.frag_mode;
    if (gpu->regs.fb.color_op.blend_mode) {
        gls_Disable(GL_COLOR_LOGIC_OP);
        gls_Enable(GL_BLEND);
        gls_BlendEquationSeparate(blend_eq[gpu->regs.fb.blend_func.rgb_eq],
                                blend_eq[gpu->regs.fb.blend_func.a_eq]);
        gls_BlendFuncSeparate(blend_func[gpu->regs.fb.blend_func.rgb_src],
                            blend_func[gpu->regs.fb.blend_func.rgb_dst],
                            blend_func[gpu->regs.fb.blend_func.a_src],
                            blend_func[gpu->regs.fb.blend_func.a_dst]);
        gls_BlendColor(gpu->regs.fb.blend_color.r / 255.f,
                     gpu->regs.fb.blend_color.g / 255.f,
                     gpu->regs.fb.blend_color.b / 255.f,
                     gpu->regs.fb.blend_color.a / 255.f);
    } else {
        gls_Disable(GL_BLEND);
        gls_Enable(GL_COLOR_LOGIC_OP);
        gls_LogicOp(logic_ops[gpu->regs.fb.logic_op]);
    }

    // alpha test
    fcfg.alphatest = gpu->regs.fb.alpha_test.enable;
    if (gpu->regs.fb.alpha_test.enable) {
        fcfg.alphafunc = gpu->regs.fb.alpha_test.func;
        fbuf.alpharef = (float) gpu->regs.fb.alpha_test.ref / 255;
    }

    // stencil test
    if (gpu->regs.fb.stencil_test.enable) {
        gls_Enable(GL_STENCIL_TEST);
        if (gpu->regs.fb.perms.depthbuf.write) {
            gls_StencilMask(gpu->regs.fb.stencil_test.bufmask);
        } else {
            gls_StencilMask(0);
        }
        gls_StencilFunc(compare_func[gpu->regs.fb.stencil_test.func],
                      gpu->regs.fb.stencil_test.ref,
                      gpu->regs.fb.stencil_test.mask);
        gls_StencilOp(stencil_op[gpu->regs.fb.stencil_op.fail],
                    stencil_op[gpu->regs.fb.stencil_op.zfail],
                    stencil_op[gpu->regs.fb.stencil_op.zpass]);
    } else {
        gls_Disable(GL_STENCIL_TEST);
    }

    // color mask and depth mask
    if (gpu->regs.fb.perms.colorbuf.write) {
        gls_ColorMask(gpu->regs.fb.color_mask.red, gpu->regs.fb.color_mask.green,
                    gpu->regs.fb.color_mask.blue,
                    gpu->regs.fb.color_mask.alpha);
    } else {
        gls_ColorMask(false, false, false, false);
    }
    // you can disable writing to the depth buffer with this register
    // instead of using the depth mask
    if (gpu->regs.fb.perms.depthbuf.write) {
        gls_DepthMask(gpu->regs.fb.color_mask.depth);
    } else {
        gls_DepthMask(false);
    }

    // depth test
    // we need to always enable the depth test, since the pica can still
    // write the depth buffer even if depth testing is disabled
    gls_Enable(GL_DEPTH_TEST);
    if (gpu->regs.fb.color_mask.depthtest) {
        gls_DepthFunc(compare_func[gpu->regs.fb.color_mask.depthfunc]);
    } else {
        gls_DepthFunc(GL_ALWAYS);
    }

    // shadow map generation mode
    // in this mode, the red component of color is set to depth
    // and depth test is performed on the color buffer
    // to make this easier for a modern shader
    // we use our own depth buffer to do depth comparisons
    // thus we must enable depth mask here and when this
    // framebuffer is cleared the depth is cleared as well
    if (gpu->regs.fb.color_op.frag_mode == 3) {
        gpu->curfb->shadowMap = true;
        gls_DepthMask(true);
    } else {
        gpu->curfb->shadowMap = false;
    }

    // lighting params
    fcfg.lightDisable = gpu->regs.lighting.disable;
    if (!fcfg.lightDisable) {
        fcfg.numlights = gpu->regs.lighting.numlights + 1;
        for (int i = 0; i < 8; i++) {
            COPYRGB(fbuf.light[i].specular0,
                    gpu->regs.lighting.light[i].specular0);
            COPYRGB(fbuf.light[i].specular1,
                    gpu->regs.lighting.light[i].specular1);
            COPYRGB(fbuf.light[i].diffuse, gpu->regs.lighting.light[i].diffuse);
            COPYRGB(fbuf.light[i].ambient, gpu->regs.lighting.light[i].ambient);
            fbuf.light[i].vec[0] = cvtf16(gpu->regs.lighting.light[i].vec.x);
            fbuf.light[i].vec[1] = cvtf16(gpu->regs.lighting.light[i].vec.y);
            fbuf.light[i].vec[2] = cvtf16(gpu->regs.lighting.light[i].vec.z);
            fcfg.light[i].config.w = gpu->regs.lighting.light[i].config;
            fbuf.light[i].spotdir[0] =
                (float) gpu->regs.lighting.light[i].spotdir.x / BIT(11);
            fbuf.light[i].spotdir[1] =
                (float) gpu->regs.lighting.light[i].spotdir.y / BIT(11);
            fbuf.light[i].spotdir[2] =
                (float) gpu->regs.lighting.light[i].spotdir.z / BIT(11);
            fbuf.light[i].attn_bias =
                cvtf20(gpu->regs.lighting.light[i].attn_bias);
            fbuf.light[i].attn_scale =
                cvtf20(gpu->regs.lighting.light[i].attn_scale);
        }
        COPYRGB(fbuf.ambient_color, gpu->regs.lighting.ambient);
        fcfg.lconfig0.w = gpu->regs.lighting.config0;
        fcfg.lconfig1.w = gpu->regs.lighting.config1;
        fcfg.llutAbs = gpu->regs.lighting.lutinputAbs;
        fcfg.llutSel = gpu->regs.lighting.lutinputSel;
        fcfg.llutScale = gpu->regs.lighting.lutinputScale;
        fcfg.lightPerm = gpu->regs.lighting.permutation;

        // light luts, use slot 4
        gls_ActiveTexture(GL_TEXTURE4);
        if (gpu->lightLutDirty) {
            gpu->lightLutDirty = false;
            // spike: writing a texture the current batch uses makes Panfrost copy or
            // flush, and games go back and forth between a few sets of tables
            g_lut_tick++;
            u64 hash = XXH3_64bits(gpu->lightLuts, sizeof gpu->lightLuts);
            int hit = -1, old = 0;
            for (int i = 0; i < LUT_CACHE; i++) {
                if (g_lut[i].tex && g_lut[i].hash == hash) hit = i;
                if (g_lut[i].used < g_lut[old].used) old = i;
            }
            if (hit < 0) {
                hit = old;
                g_lut_misses++;
                if (!g_lut[hit].tex) {
                    glGenTextures(1, &g_lut[hit].tex);
                    gls_BindTexture(GL_TEXTURE_1D_ARRAY, g_lut[hit].tex);
                    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
                    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_MAX_LEVEL, 0);
                    glTexParameteri(GL_TEXTURE_1D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
                    glTexImage2D(GL_TEXTURE_1D_ARRAY, 0, GL_R16, countof(gpu->lightLuts[0]),
                                 countof(gpu->lightLuts), 0, GL_RED, GL_UNSIGNED_SHORT,
                                 gpu->lightLuts);
                } else {
                    gls_BindTexture(GL_TEXTURE_1D_ARRAY, g_lut[hit].tex);
                    glTexSubImage2D(GL_TEXTURE_1D_ARRAY, 0, 0, 0,
                                    countof(gpu->lightLuts[0]), countof(gpu->lightLuts),
                                    GL_RED, GL_UNSIGNED_SHORT, gpu->lightLuts);
                }
                g_lut[hit].hash = hash;
            }
            g_lut[hit].used = g_lut_tick;
            g_lut_cur = g_lut[hit].tex;
        }
        gls_BindTexture(GL_TEXTURE_1D_ARRAY, g_lut_cur ? g_lut_cur : gpu->gl.lightluttex);
    }

    // fog lut, use slot 5
    gls_ActiveTexture(GL_TEXTURE5);
    gls_BindTexture(GL_TEXTURE_1D, gpu->gl.fogluttex);
    if (gpu->fogLutDirty) {
        gpu->fogLutDirty = false;
        glTexSubImage1D(GL_TEXTURE_1D, 0, 0, countof(gpu->fogLut), GL_RED,
                        GL_UNSIGNED_SHORT, gpu->fogLut);
    }

    // vertex shaders
    bool swshaders = !ctremu.hwvshaders || gpu->regs.geom.config.use_gsh;
    ProgSrc* vsrc;
    if (swshaders) {
        vsrc = g_pb.sw_vs;
        gls_BindVertexArray(gpu->gl.gpu_vao_sw);
        gls_BindBuffer(GL_ARRAY_BUFFER, gpu->gl.gpu_vbos[0]);
    } else {
        if (gpu->vsh_uniform_dirty) {
            gpu->vsh_uniform_dirty = false;
            VertUniforms vubuf;
            memcpy(vubuf.c, gpu->vsh.floatuniform, sizeof vubuf.c);
            // expand intuniform from bytes to ints
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < 4; j++) {
                    vubuf.i[i][j] = gpu->regs.vsh.intuniform[i][j];
                }
            }
            vubuf.b_raw = gpu->regs.vsh.booluniform;
            size_t uoff = stream_put(GL_UNIFORM_BUFFER, &vubuf, sizeof vubuf, g_ubo_align);
            gls_BindBufferRange(GL_UNIFORM_BUFFER, 0, g_stream, uoff, sizeof vubuf);
        }
        if (gpu->vsh.code_dirty || !g_pb.cur_hw_vs) {
            u64 hash = gpu_hash_hw_shader(gpu);
            ProgSrc* p = src_find(g_pb.vsrc, hash);
            if (!p) p = src_add(g_pb.vsrc, hash, shader_dec_vs(gpu), false);
            g_pb.cur_hw_vs = p;
        }
        vsrc = g_pb.cur_hw_vs;
        gls_BindVertexArray(gpu->gl.gpu_vao_hw);
    }

    // fragment shaders
    u64 fbufHash = gpu_hash_fs_data(&fbuf);
    if (fbufHash != gpu->lastFragUboHash) {
        gpu->lastFragUboHash = fbufHash;
        size_t uoff = stream_put(GL_UNIFORM_BUFFER, &fbuf, sizeof fbuf, g_ubo_align);
        gls_BindBufferRange(GL_UNIFORM_BUFFER, 2, g_stream, uoff, sizeof fbuf);
    }

    ProgSrc* fsrc;
    if (ctremu.ubershader) {
        gls_BindBuffer(GL_UNIFORM_BUFFER, gpu->gl.uber_ubo);
        glBufferData(GL_UNIFORM_BUFFER, sizeof fcfg, &fcfg, GL_STREAM_DRAW);
        fsrc = g_pb.uber_fs;
    } else {
        u64 hash = gpu_hash_fs(&fcfg);
        fsrc = src_find(g_pb.fsrc, hash);
        if (!fsrc) fsrc = src_add(g_pb.fsrc, hash, shader_gen_fs(&fcfg), false);
    }

    // finally get the program. One that is still building is waited for: a draw
    // left out could be one the game makes once and then keeps using. The game
    // itself runs on meanwhile, and the list makes this rare on later runs.
    ProgEnt* pe = g_pb.cur;
    if (!pe || pe->v != vsrc || pe->f != fsrc) {
        pe = prog_request(vsrc, fsrc);
        g_pb.cur = pe;
        g_pb.bound = false;
    }
    if (atomic_load_explicit(&pe->state, memory_order_acquire) != 2) prog_wait(pe);
    GLuint prog = pe->prog;
    if (!g_pb.bound) {
        glUseProgram(prog);
        g_pb.bound = true;
    }
    {
        typeof(&g_depth_uni[0]) u = &g_depth_uni[prog & 8191];
        if (!u->known) {
            memcpy(u->loc, pe->depthloc, sizeof u->loc);
            u->known = true;
            u->sent = false;
        }
        if (!u->sent || u->offset != dmOffset || u->scale != dmScale ||
            u->wbuffer != wbuffer) {
            glUniform1f(u->loc[0], dmOffset);
            glUniform1f(u->loc[1], dmScale);
            glUniform1i(u->loc[2], wbuffer);
            u->offset = dmOffset;
            u->scale = dmScale;
            u->wbuffer = wbuffer;
            u->sent = true;
        }
    }

    // starting  index
    int basevert = immediate ? 0 : gpu->regs.geom.vtx_off;
    // how many verts are sent to the VS (can be more than nverts for
    // drawelements)
    int nbufverts = nverts;
    u32 rangemin = 0, rangemax = 0;
    size_t indexoff = 0;

    // setup index buffer and find min/max index
    void* indexbuf = nullptr;
    bool indexsize = gpu->regs.geom.indexfmt;
    if (elements) {
        u32 minind = snap->minind, maxind = snap->maxind;
        indexbuf =
            PTR(gpu->regs.geom.attr_base * 8 + gpu->regs.geom.indexbufoff);
        gls_BindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_vstream);
        indexoff = snap->index;
        rangemin = minind;
        rangemax = maxind;
        // update these since we are drawing elements
        basevert = minind;
        nbufverts = maxind + 1 - minind;
    }

    if (swshaders) {
        fvec4 vshout[nbufverts][16];
        // run the vertex shader
        gpu_run_vsh(gpu, immediate, basevert, nbufverts, vshout);

        // fshin is either vshout or gshout
        fvec4(*fshin)[16] = vshout;

        // run the geometry shader if enabled
        ShaderUnit gsh;
        gpu_init_gsh(gpu, &gsh);
        if (gpu->regs.geom.config.use_gsh) {
            gpu_run_gsh(gpu, &gsh, elements, basevert, nverts, vshout,
                        indexsize, indexbuf);

            // since we are drawing geometry shader output, these are changed
            // appriopriately
            fshin = gsh.gsh.outvtx.d;
            basevert = 0;
            nverts = gsh.gsh.outvtx.size;
            nbufverts = nverts;
            elements = false;
        }

        // use the outmap config to setup the final vertex buffer sent to gpu
        // for the fragment shader
        Vertex vbuf[nbufverts];
        for (int i = 0; i < nbufverts; i++) {
            gpu_write_outmap_vtx(gpu, &vbuf[i], fshin[i]);
        }
        Vec_free(gsh.gsh.outvtx);

        glBufferData(GL_ARRAY_BUFFER, sizeof vbuf, vbuf, GL_STREAM_DRAW);
    } else {
        if (immediate) {
            setup_hw_vao_imm(gpu);
        } else {
            setup_hw_vao(gpu, snap, nbufverts);
        }
    }
    Vec_free(gpu->immattrs);

    if (g_draw_dump) {
        static int dump_n;
        fprintf(stderr,
                "D %d el %d imm %d n %d min %u max %u sw %d gsh %d fb %x w %d h %d v %08x i %08x u %08x "
                "base %x vtxoff %d attrs %x tex %x %x %x prim %d prog %u\n",
                dump_n++, elements, immediate, nverts, rangemin, rangemax, swshaders,
                gpu->regs.geom.config.use_gsh, gpu->curfb->color_paddr, gpu->curfb->width,
                gpu->curfb->height, snap->vhash, snap->ihash,
                (u32) XXH3_64bits(gpu->vsh.floatuniform, sizeof gpu->vsh.floatuniform),
                gpu->regs.geom.attr_base, gpu->regs.geom.vtx_off, (u32) gpu->regs.geom.attr_format,
                gpu->regs.tex.tex0.addr << 3, gpu->regs.tex.tex1.addr << 3, gpu->regs.tex.tex2.addr << 3,
                primMode, prog);
    }

    // spike: SPIKE_SKIPDRAW bit 0 leaves every draw out, bit 1 the lit ones, bit 2 the unlit
    // ones; to see what the GPU spends its time on. The stats line counts lit draws.
    // SPIKE_TINYSCISSOR=1 keeps every draw but clips it to one pixel: the CPU and vertex work
    // stay, the fragment work goes, so the difference is what fragments cost.
    static int skipdraw = -1, tiny = -1;
    if (skipdraw < 0) skipdraw = getenv("SPIKE_SKIPDRAW") ? atoi(getenv("SPIKE_SKIPDRAW")) : 0;
    if (tiny < 0) tiny = getenv("SPIKE_TINYSCISSOR") != nullptr;
    if (tiny) {
        gls_Enable(GL_SCISSOR_TEST);
        gls_Scissor(0, 0, 1, 1);
    }
    if (!fcfg.lightDisable) g_draws_lit++;
    bool skip = (skipdraw & 1) || ((skipdraw & 2) && !fcfg.lightDisable) ||
                ((skipdraw & 4) && fcfg.lightDisable);

    // finally do the draw call
    if (skip) {
    } else if (elements) {
        glDrawRangeElementsBaseVertex(prim_mode[primMode], rangemin, rangemax, nverts,
                                      indextypes[indexsize], (void*) indexoff, -basevert);
    } else {
        glDrawArrays(prim_mode[primMode], 0, nverts);
    }

    gpu->curfb->dirty = true;
}
