// RG DS Core's shared in-game menu (runtime/game_menu.py), for an engine that is
// not on the SDL2 bridge: the same files beside RGDS_CONTROL_FILE, the same frame
// in RGDS_OVERLAY_FILE (src/platform/overlay_frame.hpp).
#include "rgds_bridge.h"

#include <SDL3/SDL.h>
#include <execinfo.h>
#include <fcntl.h>
#include <pthread.h>
#include <time.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "emulator.h"
#include "savestate.h"
#include "video/gpu.h"
#include "video/renderer_gl.h"

extern void gpu_worker_stop();
extern bool gpu_worker_on();
extern void rgds_worker_restart();
extern void rgds_audio_hold(bool hold);

static bool screen_pass(int screen);
static bool panel_ground();
static bool set_frame(const char* frame);
extern const char* rgds_shader_file(int id);
extern bool rgds_bezel_known(const char* slug);
extern void rgds_bezel_draw(unsigned char* pixels, const char* slug, int sx, int sy, int sw, int sh);

static struct {
    bool on, held, autosave;
    char control[512], block[512], result[512], capture[512], speed_state[512], speed_request[512];
    char autostate[512];
    char last_token[128], speed_token[128];
    Uint64 poll_at;
    atomic_bool show, capture_wanted;
    atomic_int fit, second, bar, shader;
    // the lid around the top screen ("" none): written between frames, read on the renderer thread
    char lid[32];
    atomic_int lid_change;
    GLuint lid_tex;
    u8* lid_pixels;
    int lid_drawn, lid_fit;
    // a state being written or read on its own thread
    int job;
    atomic_bool job_done;
    bool job_ok, job_wrecked, job_quit;
    char job_token[128], job_pending[600];
    void* job_blob;
    u64 job_blob_size;
    SSLoaded* job_loaded;
    Uint64 job_at;
    // screen looks
    struct { GLint prog; GLint size, target; } pass[64][2][2];
    GLuint pass_vao, pass_vbo;
    const char* shader_dir;
    char volume_ini[512], volume_hud[512], volume_token[128];
    int watched_volume;
    atomic_int volume_level;
    _Atomic Uint64 volume_until;
    atomic_bool fps_on;
    int poll_count;
    GLuint hud_tex;
    u8* hud_pixels;
    int hud_volume, hud_fast;
    double hud_fps, fps_shown;
    int fps_frames;
    Uint64 fps_at;
    const char* overlay_path;
    u8* overlay;
    u32 seen;
    u8* staging;
    GLuint tex[2], prog;
    GLint use_tex, tint;
    bool have_picture;
} g;

#define PANEL_BYTES (640 * 480 * 4)
#define OVERLAY_HEAD 64
#define OVERLAY_BYTES (OVERLAY_HEAD + PANEL_BYTES * 4)

void rgds_bridge_start() {
    const char* control = getenv("RGDS_CONTROL_FILE");
    if (!control || !*control) return;
    g.on = true;
    gpu_gl_screen_pass = screen_pass;
    snprintf(g.control, sizeof g.control, "%s", control);
    snprintf(g.block, sizeof g.block, "%s.blocked", control);
    snprintf(g.result, sizeof g.result, "%s.result", control);
    snprintf(g.capture, sizeof g.capture, "%s.capture", control);
    snprintf(g.speed_state, sizeof g.speed_state, "%s.speed-state", control);
    snprintf(g.speed_request, sizeof g.speed_request, "%s.speed-request", control);
    const char* v = getenv("RGDS_AUTOSTATE_FILE");
    if (v) snprintf(g.autostate, sizeof g.autostate, "%s", v);
    v = getenv("RGDS_AUTOSAVE");
    g.autosave = !v || strcmp(v, "0");
    g.overlay_path = getenv("RGDS_OVERLAY_FILE");
    v = getenv("RGDS_VOLUME_DATA");
    if (v) {
        snprintf(g.volume_ini, sizeof g.volume_ini, "%s/volume.ini", v);
        snprintf(g.volume_hud, sizeof g.volume_hud, "%s/session/volume-hud.ini", v);
    }
    g.watched_volume = -1;
    atomic_store(&g.volume_level, -1);
    g.hud_volume = g.hud_fast = -2;
    g.hud_fps = -2;
    g.fps_shown = -1;
    v = getenv("RGDS_FPS");
    atomic_store(&g.fps_on, v && !strcmp(v, "1"));
    // what the game's Graphics page was left at: fit second shader frame
    v = getenv("RGDS_3DS_VIEW");
    int fit = 0, second = 0, shader = 0;
    char frame[64] = "off";
    if (v && sscanf(v, "%d %d %d %63s", &fit, &second, &shader, frame) == 4) {
        atomic_store(&g.fit, fit);
        atomic_store(&g.second, second);
        atomic_store(&g.shader, shader);
        set_frame(frame);
    }
    g.lid_drawn = g.lid_fit = -1;
    gpu_gl_panel_ground = panel_ground;
}

static bool set_frame(const char* frame) {
    bool bar = !strcmp(frame, "bar"), lid = rgds_bezel_known(frame);
    if (!bar && !lid && strcmp(frame, "off")) return false;
    atomic_store(&g.bar, bar);
    snprintf(g.lid, sizeof g.lid, "%s", lid ? frame : "");
    atomic_fetch_add(&g.lid_change, 1);
    return true;
}

int rgds_bridge_filter() {
    if (!g.on) return ctremu.outputfilter;
    int fit = atomic_load(&g.fit);
    if (fit == 1) return FILTER_NEAREST;
    return fit == 0 || atomic_load(&g.shader) == 1 ? FILTER_SHARP_LINEAR : FILTER_LINEAR;
}

// ---- screen looks: RG DS Core's preset files (GLES2) over a 3DS screen
extern const char* rgds_shader_file(int id);
extern const char* rgds_shader_defines(int id);
extern bool rgds_shader_uses_target(int id);

static char* section_of(const char* text, const char* tag) {
    char open[32], close[32];
    snprintf(open, sizeof open, "<%s>", tag);
    snprintf(close, sizeof close, "</%s>", tag);
    const char* a = strstr(text, open);
    if (!a) return nullptr;
    a += strlen(open);
    const char* b = strstr(a, close);
    return b ? strndup(a, b - a) : nullptr;
}

static char* replace_all(const char* text, const char* from, const char* to) {
    size_t n = 0, lf = strlen(from), lt = strlen(to);
    for (const char* p = text; (p = strstr(p, from)); p += lf) n++;
    char* out = malloc(strlen(text) + n * (lt > lf ? lt - lf : 0) + 1);
    char* w = out;
    for (const char* p = text;;) {
        const char* q = strstr(p, from);
        if (!q) {
            strcpy(w, p);
            break;
        }
        memcpy(w, p, q - p);
        w += q - p;
        memcpy(w, to, lt);
        w += lt;
        p = q + lf;
    }
    return out;
}

static GLuint pass_compile(GLenum type, const char* src, int id) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    int okay = 0;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &okay);
    if (!okay) {
        char log[1024] = {};
        glGetShaderInfoLog(sh, sizeof log - 1, nullptr, log);
        fprintf(stderr, "rgds bridge: shader %d: %s\n", id, log);
        glDeleteShader(sh);
        return 0;
    }
    return sh;
}

static void pass_build(int id, int screen, bool sharp) {
    typeof(&g.pass[0][0][0]) p = &g.pass[id][screen][sharp];
    p->prog = -1;
    if (!g.shader_dir) g.shader_dir = getenv("RGDS_SHADER_DIR");
    const char* file = rgds_shader_file(id);
    if (!g.shader_dir || !file) return;
    char path[700];
    snprintf(path, sizeof path, "%s/%s.dsd", g.shader_dir, file);
    FILE* f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "rgds bridge: no shader file %s\n", path);
        return;
    }
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* text = calloc(1, size + 1);
    fread(text, 1, size, f);
    fclose(f);
    char* vert = section_of(text, "vertex");
    char* frag = section_of(text, "fragment");
    free(text);
    if (!vert || !frag) {
        free(vert);
        free(frag);
        return;
    }
    int width = SCREEN_WIDTH(screen);
    // the 3DS screens lie on their side in their textures: the presets see them upright
    const char* sample_sharp =
        "vec4 rgds_sample(vec2 uv){vec2 t=vec2(1.0-uv.y,1.0-uv.x)*vec2(textureSize(u_texture,0));"
        "vec2 scale=ceil(1.0/fwidth(t));vec2 f=fract(t);"
        "f=clamp(f*scale,0.0,0.5)+clamp((f-1.0)*scale+0.5,0.0,0.5);"
        "return texture(u_texture,(floor(t)+f)/vec2(textureSize(u_texture,0)));}\n";
    const char* sample_plain = "vec4 rgds_sample(vec2 uv){return texture(u_texture,vec2(1.0-uv.y,1.0-uv.x));}\n";
    char grid[64];
    snprintf(grid, sizeof grid, "vec2(%d.0,240.0)", width);
    char* a = replace_all(frag, "texture2D(u_texture,", "rgds_sample(");
    char* b = replace_all(a, "vec2(256.0,192.0)", grid);
    char sampler[800];
    snprintf(sampler, sizeof sampler, "uniform vec4 u_texture_size;\n%s",
             sharp && !rgds_shader_uses_target(id) ? sample_sharp : sample_plain);
    char* c = replace_all(b, "uniform vec4 u_texture_size;", sampler);
    free(a);
    free(b);
    bool declared = strstr(c, "rgds_sample(vec2 uv)") != nullptr;
    const char* vhead = "#version 330 core\n#define attribute in\n#define varying out\n";
    const char* fhead_tpl =
        "#version 330 core\n%s#define varying in\n#define texture2D texture\n"
        "out vec4 rgds_color;\n#define gl_FragColor rgds_color\n%s%s";
    char* vs_text;
    char* fs_text;
    asprintf(&vs_text, "%s%s", vhead, vert);
    asprintf(&fs_text, fhead_tpl, rgds_shader_defines(id),
             declared ? "" : "uniform sampler2D u_texture;\n", declared ? "" : sample_plain);
    char* whole;
    asprintf(&whole, "%s%s", fs_text, c);
    GLuint vs = pass_compile(GL_VERTEX_SHADER, vs_text, id);
    GLuint fs = pass_compile(GL_FRAGMENT_SHADER, whole, id);
    free(vs_text);
    free(fs_text);
    free(whole);
    free(c);
    free(vert);
    free(frag);
    if (!vs || !fs) return;
    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glBindAttribLocation(prog, 0, "a_vertex_coordinate");
    glBindAttribLocation(prog, 1, "a_texture_coordinate");
    glLinkProgram(prog);
    glDeleteShader(vs);
    glDeleteShader(fs);
    int linked = 0;
    glGetProgramiv(prog, GL_LINK_STATUS, &linked);
    if (!linked) {
        char log[1024] = {};
        glGetProgramInfoLog(prog, sizeof log - 1, nullptr, log);
        fprintf(stderr, "rgds bridge: shader %d link: %s\n", id, log);
        glDeleteProgram(prog);
        return;
    }
    glUseProgram(prog);
    glUniform1i(glGetUniformLocation(prog, "u_texture"), 0);
    p->size = glGetUniformLocation(prog, "u_texture_size");
    p->target = glGetUniformLocation(prog, "u_target_size");
    p->prog = prog;
}

// One screen in the current viewport, its texture bound; false: draw it the plain way.
static bool screen_pass(int screen) {
    int id = atomic_load(&g.shader);
    if (id < 2 || id >= 64) return false;
    bool sharp = atomic_load(&g.fit) == 0;
    typeof(&g.pass[0][0][0]) p = &g.pass[id][screen][sharp];
    if (!p->prog) pass_build(id, screen, sharp);
    if (p->prog <= 0) return false;
    if (!g.pass_vao) {
        // position, then the picture's coordinate with its first row at the top
        static const float quad[] = {-1, 1, 0, 0, -1, -1, 0, 1, 1, 1, 1, 0, 1, -1, 1, 1};
        glGenVertexArrays(1, &g.pass_vao);
        glGenBuffers(1, &g.pass_vbo);
        glBindVertexArray(g.pass_vao);
        glBindBuffer(GL_ARRAY_BUFFER, g.pass_vbo);
        glBufferData(GL_ARRAY_BUFFER, sizeof quad, quad, GL_STATIC_DRAW);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) 0);
        glEnableVertexAttribArray(1);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*) (2 * sizeof(float)));
    }
    glUseProgram(p->prog);
    glBindVertexArray(g.pass_vao);
    float w = SCREEN_WIDTH(screen), h = SCREEN_HEIGHT;
    if (p->size >= 0) glUniform4f(p->size, 1.f / w, 1.f / h, w, h);
    if (p->target >= 0) glUniform2f(p->target, ctremu.screens[screen].w, ctremu.screens[screen].h);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
    return true;
}

// One 3DS screen per panel, in the panel's pixels (y down)
void rgds_bridge_layout(int panel, Rect* top, Rect* bot) {
    int fit = g.on ? atomic_load(&g.fit) : 0, second = g.on ? atomic_load(&g.second) : 0;
    bool bar = g.on && atomic_load(&g.bar);
    Rect game = fit == 1 ? (Rect) {120, bar ? 0 : 120, 400, 240} : (Rect) {0, bar ? 0 : 48, 640, 384};
    Rect full = {0, 0, 640, 480};
    *top = (Rect) {};
    *bot = (Rect) {};
    // Off: the lower panel stays dark; Swap: each screen on the other panel
    if (second == 2) {
        if (panel == 0) *bot = full;
        else *top = game;
    } else if (panel == 0) {
        *top = game;
    } else if (second != 1) {
        *bot = full;
    }
}

bool rgds_bridge_on() {
    return g.on;
}

static void answer(const char* token, const char* state) {
    char pending[600];
    snprintf(pending, sizeof pending, "%s.tmp", g.result);
    FILE* f = fopen(pending, "w");
    if (f) {
        fprintf(f, "token=%s\nstate=%s\n", token, state);
        fclose(f);
        rename(pending, g.result);
    }
    fprintf(stderr, "rgds bridge: %s %s\n", token, state);
}

static void set_speed(bool on) {
    ctremu.fastforward = on;
    char pending[600];
    snprintf(pending, sizeof pending, "%s.pending", g.speed_state);
    FILE* f = fopen(pending, "w");
    if (f) {
        fprintf(f, "active=%d\n", on);
        fclose(f);
        rename(pending, g.speed_state);
    }
}

static void* job_thread(void* arg) {
    (void) arg;
    if (g.job == 1) {
        g.job_ok = savestate_write(g.job_pending, g.job_blob, g.job_blob_size);
        free(g.job_blob);
    } else {
        g.job_loaded = savestate_read(g.autostate, &g.job_wrecked);
        g.job_ok = g.job_loaded != nullptr;
    }
    atomic_store(&g.job_done, true);
    return nullptr;
}

// A note that outlives the next launch (the launcher's log is written anew each time). Only what
// is safe in a signal handler, apart from the game's name.
void rgds_crash_note(const char* what, void* addr, void* pc, void* lr) {
    int fd = open("crashes.log", O_WRONLY | O_CREAT | O_APPEND, 0666);
    if (fd < 0) return;
    char comm[32] = "";
    pthread_getname_np(pthread_self(), comm, sizeof comm);
    dprintf(fd, "=== %s, thread %s, addr %p pc %p lr %p, game %s, unix time %lld, fast-forward %d, held %d, job %d\n",
            what, comm, addr, pc, lr, ctremu.romfilenoext ? ctremu.romfilenoext : "?",
            (long long) time(nullptr), ctremu.fastforward, g.held, g.job);
    void* frames[32];
    backtrace_symbols_fd(frames, backtrace(frames, 32), fd);
    int maps = open("/proc/self/maps", O_RDONLY);
    if (maps >= 0) {
        char buf[4096];
        ssize_t n;
        // where the code lay, to read the addresses above with
        while ((n = read(maps, buf, sizeof buf)) > 0) write(fd, buf, n);
        close(maps);
    }
    close(fd);
}

// The menu and the HUD change on their own, whatever the game does.
bool rgds_bridge_wants_present() {
    if (!g.on) return true;
    if (atomic_load(&g.show) || atomic_load(&g.capture_wanted)) return true;
    // the bar until it has gone; the counter and the badge go along with the game's own pictures,
    // or they would take the time those need
    Uint64 now = SDL_GetTicksNS();
    if (now < atomic_load(&g.volume_until) + 100 * SDL_NS_PER_MS) return true;
    // ... unless a game has stopped showing any, as it does while loading
    return atomic_load(&g.fps_on) && now - g.fps_at >= 500 * SDL_NS_PER_MS;
}

bool rgds_bridge_busy() {
    return g.job != 0;
}

static void job_start(int job, const char* token, bool quit) {
    g.job = job;
    g.job_quit = quit;
    g.job_at = SDL_GetTicksNS();
    snprintf(g.job_token, sizeof g.job_token, "%s", token);
    atomic_store(&g.job_done, false);
    pthread_t thread;
    pthread_create(&thread, nullptr, job_thread, nullptr);
    pthread_detach(thread);
}

// The menu's spinner goes on: only the pictures are read with the renderer stopped.
static void start_save(const char* token, bool quit) {
    if (!g.autostate[0]) {
        answer(token, "save-failed");
        return;
    }
    char dir[512];
    snprintf(dir, sizeof dir, "%s", g.autostate);
    for (char* p = dir + 1; *p; p++) {
        if (*p == '/') {
            *p = 0;
            mkdir(dir, 0777);
            *p = '/';
        }
    }
    snprintf(g.job_pending, sizeof g.job_pending, "%s.pending", g.autostate);
    gpu_worker_stop();
    g.job_blob = savestate_capture(&g.job_blob_size);
    rgds_worker_restart();
    job_start(1, token, quit);
}

static void job_finish() {
    double ms = (SDL_GetTicksNS() - g.job_at) / 1e6;
    int job = g.job;
    if (job == 1) {
        bool ok = g.job_ok;
        if (ok) ok = !rename(g.job_pending, g.autostate);
        else unlink(g.job_pending);
        fprintf(stderr, "rgds bridge: state written in %.0f ms: %d\n", ms, ok);
        answer(g.job_token, ok ? "saved" : "save-failed");
        if (ok && g.job_quit) ctremu.running = false;
    } else {
        fprintf(stderr, "rgds bridge: state read in %.0f ms: %d\n", ms, g.job_ok);
        if (g.job_ok) {
            gpu_worker_stop();
            savestate_read_finish(g.job_loaded);
            rgds_worker_restart();
        }
        answer(g.job_token, g.job_ok ? "loaded" : "load-failed");
        if (g.job_wrecked) {
            fprintf(stderr, "rgds bridge: the state broke while loading, the game cannot go on\n");
            ctremu.running = false;
        }
    }
    g.job = 0;
}

void rgds_bridge_autoload() {
    const char* v = getenv("RGDS_NATIVE_AUTOLOAD");
    if (!g.on || !v || strcmp(v, "1") || !g.autostate[0]) return;
    bool wrecked = false;
    Uint64 t0 = SDL_GetTicksNS();
    SSLoaded* loaded = savestate_read(g.autostate, &wrecked);
    if (loaded) savestate_read_finish(loaded);
    fprintf(stderr, "rgds bridge: state at start read in %.0f ms: %d\n",
            (SDL_GetTicksNS() - t0) / 1e6, loaded != nullptr);
    // tools/tanuki3ds/state-check.sh: what was just read, written again, must be the same bytes
    const char* again = getenv("RGDS_STATE_AGAIN");
    if (loaded && again) {
        u64 size = 0;
        void* blob = savestate_capture(&size);
        bool same = savestate_write(again, blob, size);
        free(blob);
        fprintf(stderr, "rgds bridge: state written again: %d\n", same);
    }
    if (wrecked) ctremu.running = false;
}

static void command(const char* token, const char* what) {
    fprintf(stderr, "rgds bridge: command %s %s\n", token, what);
    if (!strcmp(what, "save")) {
        start_save(token, false);
    } else if (!strcmp(what, "save-quit") || (!strcmp(what, "quit") && g.autosave)) {
        start_save(token, true);
    } else if (!strcmp(what, "quit")) {
        ctremu.running = false;
    } else if (!strcmp(what, "load")) {
        job_start(2, token, false);
    } else if (!strcmp(what, "fast-forward")) {
        set_speed(!ctremu.fastforward);
        answer(token, ctremu.fastforward ? "fast-forward-on" : "fast-forward-off");
    } else if (!strcmp(what, "fps:0")) {
        atomic_store(&g.fps_on, false);
        answer(token, "fps-off");
    } else if (!strcmp(what, "fps:1")) {
        atomic_store(&g.fps_on, true);
        answer(token, "fps-on");
    } else if (!strncmp(what, "ds:", 3)) {
        answer(token, "ds-unavailable");
    } else if (!strncmp(what, "play-graphics:", 14)) {
        int fit, second, used = 0;
        const char* frame = "";
        if (sscanf(what + 14, "%d:%d:%n", &fit, &second, &used) == 2 && used) frame = what + 14 + used;
        if (used && fit >= 0 && fit <= 2 && second >= 0 && second <= 3 && set_frame(frame)) {
            atomic_store(&g.fit, fit);
            atomic_store(&g.second, second == 3 ? 0 : second);
            answer(token, "play-graphics-applied");
        } else {
            answer(token, "play-graphics-failed");
        }
    } else if (!strncmp(what, "shader:", 7)) {
        int id = atoi(what + 7);
        if (id >= 0 && id < 64 && (id < 2 || rgds_shader_file(id))) {
            atomic_store(&g.shader, id);
            answer(token, "shader-applied");
        } else {
            answer(token, "shader-failed");
        }
    } else if (!strncmp(what, "play-color:", 11)) {
        answer(token, "play-color-failed");
    } else if (!strncmp(what, "cheat:", 6)) {
        answer(token, "cheat-failed");
    } else {
        answer(token, "shader-failed");
    }
}

// volume.ini and the volume keys' request, as the bridge's volume_feedback
static void read_volume(const char* path, int* level, char* event, size_t size) {
    FILE* f = fopen(path, "r");
    if (!f) return;
    char line[160];
    while (fgets(line, sizeof line, f)) {
        line[strcspn(line, "\r\n")] = 0;
        if (!strncmp(line, "volume=", 7)) {
            char* end;
            long v = strtol(line + 7, &end, 10);
            if (!*end && v >= 0 && v <= 100) *level = v;
        } else if (event && !strncmp(line, "token=", 6)) {
            snprintf(event, size, "%s", line + 6);
        }
    }
    fclose(f);
}

static void poll_volume() {
    if (!g.volume_ini[0]) return;
    int level = -1;
    char event[128] = {};
    read_volume(g.volume_hud, &level, event, sizeof event);
    read_volume(g.volume_ini, &level, nullptr, 0);
    bool requested = *event && strcmp(event, g.volume_token);
    if (level >= 0 && (requested || (g.watched_volume >= 0 && level != g.watched_volume))) {
        atomic_store(&g.volume_level, level);
        atomic_store(&g.volume_until, SDL_GetTicksNS() + 1800 * SDL_NS_PER_MS);
    }
    if (level >= 0) g.watched_volume = level;
    snprintf(g.volume_token, sizeof g.volume_token, "%s", event);
}

static void poll_files() {
    if (++g.poll_count >= 3) {
        g.poll_count = 0;
        poll_volume();
    }
    char buf[600] = {};
    FILE* f = fopen(g.control, "r");
    if (f) {
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        fclose(f);
        buf[n] = 0;
        char token[128] = {};
        int used = 0;
        if (sscanf(buf, "%127s %n", token, &used) == 1 && used > 0 && strcmp(token, g.last_token)) {
            char* what = buf + used;
            what[strcspn(what, "\r\n")] = 0;
            if (*what) {
                snprintf(g.last_token, sizeof g.last_token, "%s", token);
                command(token, what);
            }
        }
    }
    f = fopen(g.speed_request, "r");
    if (f) {
        char line[200], id[128] = {}, action[64] = {};
        while (fgets(line, sizeof line, f)) {
            line[strcspn(line, "\r\n")] = 0;
            if (!strncmp(line, "token=", 6)) snprintf(id, sizeof id, "%s", line + 6);
            if (!strncmp(line, "action=", 7)) snprintf(action, sizeof action, "%s", line + 7);
        }
        fclose(f);
        if (*id && strcmp(id, g.speed_token)) {
            snprintf(g.speed_token, sizeof g.speed_token, "%s", id);
            if (!strcmp(action, "fast-forward-off")) set_speed(false);
            else if (!g.held && !strcmp(action, "fast-forward-on")) set_speed(true);
            else if (!g.held && !strcmp(action, "fast-forward-toggle")) set_speed(!ctremu.fastforward);
        }
    }
    if (!access(g.capture, F_OK)) atomic_store(&g.capture_wanted, true);
}

bool rgds_bridge_tick() {
    if (!g.on || !ctremu.initialized || !gpu_worker_on()) return false;
    if (g.job) {
        // keeps the menu up, whatever else happens, until the state is done
        if (atomic_load(&g.job_done)) job_finish();
        if (g.job) return true;
    }
    bool blocked = !access(g.block, F_OK);
    if (blocked != g.held) {
        g.held = blocked;
        atomic_store(&g.show, blocked);
        rgds_audio_hold(blocked);
    }
    Uint64 now = SDL_GetTicksNS();
    if (now - g.poll_at >= 25 * SDL_NS_PER_MS) {
        g.poll_at = now;
        poll_files();
    }
    return g.held || g.job;
}

static void write_capture() {
    // the save gallery's picture: 32-bit BMP, bottom row first as GL reads it
    static u8 px[PANEL_BYTES];
    glReadPixels(0, 0, 640, 480, GL_BGRA, GL_UNSIGNED_BYTE, px);
    for (int i = 3; i < PANEL_BYTES; i += 4) px[i] = 255;
    u8 head[54] = {'B', 'M'};
    u32 size = 54 + PANEL_BYTES;
    memcpy(head + 2, &size, 4);
    head[10] = 54;
    head[14] = 40;
    u32 w = 640, h = 480;
    memcpy(head + 18, &w, 4);
    memcpy(head + 22, &h, 4);
    head[26] = 1;
    head[28] = 32;
    char path[600], pending[640];
    snprintf(path, sizeof path, "%s-top.bmp", g.capture);
    snprintf(pending, sizeof pending, "%s.pending", path);
    FILE* f = fopen(pending, "wb");
    if (f) {
        fwrite(head, 1, 54, f);
        fwrite(px, 1, PANEL_BYTES, f);
        fclose(f);
        rename(pending, path);
    }
    unlink(g.capture);
}

static GLuint compile_part(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    return sh;
}

static bool overlay_ready() {
    if (g.overlay) return true;
    if (!g.overlay_path) return false;
    int fd = open(g.overlay_path, O_RDWR | O_CLOEXEC);
    if (fd < 0) return false;
    struct stat st;
    if (fstat(fd, &st) || st.st_size != OVERLAY_BYTES) {
        close(fd);
        return false;
    }
    void* m = mmap(nullptr, OVERLAY_BYTES, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    if (m == MAP_FAILED) return false;
    g.overlay = m;
    g.staging = malloc(PANEL_BYTES * 2);
    return true;
}

static bool overlay_refresh() {
    if (!overlay_ready()) return false;
    u32* h = (u32*) g.overlay;
    if (h[0] != 0x52474453 || h[1] != 640 || h[2] != 480) return false;
    u32 seq = __atomic_load_n(&h[3], __ATOMIC_ACQUIRE);
    if (!__atomic_load_n(&h[4], __ATOMIC_ACQUIRE)) return false;
    if (!seq || (seq & 1)) return g.have_picture;
    if (seq != g.seen || !g.have_picture) {
        u32 slot = __atomic_load_n(&h[5], __ATOMIC_ACQUIRE);
        if (slot > 1) return false;
        memcpy(g.staging, g.overlay + OVERLAY_HEAD + slot * PANEL_BYTES * 2, PANEL_BYTES * 2);
        if (__atomic_load_n(&h[3], __ATOMIC_ACQUIRE) != seq) return g.have_picture;
        for (int i = 0; i < 2; i++) {
            glBindTexture(GL_TEXTURE_2D, g.tex[i]);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE,
                            g.staging + i * PANEL_BYTES);
        }
        g.seen = seq;
        g.have_picture = true;
    }
    return true;
}

static void overlay_program() {
    if (!g.prog) {
        GLuint vs = compile_part(GL_VERTEX_SHADER,
            "#version 330 core\n"
            "out vec2 uv;\n"
            "const vec2 p[3] = vec2[](vec2(-1, -1), vec2(3, -1), vec2(-1, 3));\n"
            "void main() {\n"
            "  gl_Position = vec4(p[gl_VertexID], 0, 1);\n"
            "  uv = vec2(p[gl_VertexID].x * 0.5 + 0.5, 0.5 - p[gl_VertexID].y * 0.5);\n"
            "}\n");
        GLuint fs = compile_part(GL_FRAGMENT_SHADER,
            "#version 330 core\n"
            "in vec2 uv;\n"
            "out vec4 color;\n"
            "uniform sampler2D menu;\n"
            "uniform bool useTex;\n"
            "uniform vec4 tint;\n"
            "void main() { color = useTex ? texture(menu, uv) : tint; }\n");
        g.prog = glCreateProgram();
        glAttachShader(g.prog, vs);
        glAttachShader(g.prog, fs);
        glLinkProgram(g.prog);
        glDeleteShader(vs);
        glDeleteShader(fs);
        g.use_tex = glGetUniformLocation(g.prog, "useTex");
        g.tint = glGetUniformLocation(g.prog, "tint");
        glUseProgram(g.prog);
        glUniform1i(glGetUniformLocation(g.prog, "menu"), 0);
        glGenTextures(2, g.tex);
        for (int i = 0; i < 2; i++) {
            glBindTexture(GL_TEXTURE_2D, g.tex[i]);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
    }
}

// The lid under the 3DS top screen, on the panel that shows it.
static bool panel_ground() {
    if (!g.on || !g.lid[0] || !ctremu.screens[SCREEN_TOP].w) return false;
    overlay_program();
    if (!g.lid_tex) {
        g.lid_pixels = malloc(PANEL_BYTES);
        glGenTextures(1, &g.lid_tex);
        glBindTexture(GL_TEXTURE_2D, g.lid_tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    }
    glBindTexture(GL_TEXTURE_2D, g.lid_tex);
    int change = atomic_load(&g.lid_change), fit = atomic_load(&g.fit);
    if (change != g.lid_drawn || fit != g.lid_fit) {
        g.lid_drawn = change;
        g.lid_fit = fit;
        Rect r = ctremu.screens[SCREEN_TOP];
        rgds_bezel_draw(g.lid_pixels, g.lid, r.x, r.y, r.w, r.h);
        glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, g.lid_pixels);
    }
    glUseProgram(g.prog);
    glViewport(0, 0, 640, 480);
    glUniform1i(g.use_tex, 1);
    glDrawArrays(GL_TRIANGLES, 0, 3);
    return true;
}

extern void rgds_hud_draw(unsigned char* pixels, int volume, bool fast_forward, double fps);

void rgds_bridge_panel(int panel) {
    if (!g.on) return;
    if (panel == 0 && atomic_exchange(&g.capture_wanted, false)) write_capture();
    bool show = atomic_load(&g.show);
    if (!show) g.have_picture = false;

    // frames shown per second, for the counter
    Uint64 now = SDL_GetTicksNS();
    if (panel == 0) {
        g.fps_frames++;
        if (now - g.fps_at >= 500 * SDL_NS_PER_MS) {
            if (g.fps_at) g.fps_shown = g.fps_frames * 1e9 / (now - g.fps_at);
            g.fps_at = now;
            g.fps_frames = 0;
        }
    }
    int hud_volume = now < atomic_load(&g.volume_until) ? atomic_load(&g.volume_level) : -1;
    int hud_fast = ctremu.fastforward && !show;
    double hud_fps = atomic_load(&g.fps_on) && !show ? g.fps_shown : -1;
    bool hud = panel == 0 && (hud_volume >= 0 || hud_fast || hud_fps >= 0);
    if (!show && !hud) return;

    overlay_program();
    glActiveTexture(GL_TEXTURE0);
    bool picture = show && (panel == 0 ? overlay_refresh() : g.have_picture);
    glUseProgram(g.prog);
    glViewport(0, 0, 640, 480);
    glEnable(GL_BLEND);
    glBlendEquation(GL_FUNC_ADD);
    glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    // the same darkening of the held game as the bridge's
    if (show) {
        glUniform1i(g.use_tex, 0);
        glUniform4f(g.tint, 0, 0, 0, 150.f / 255);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    if (picture) {
        glBindTexture(GL_TEXTURE_2D, g.tex[panel]);
        glUniform1i(g.use_tex, 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
        if (panel == 1) __atomic_store_n(&((u32*) g.overlay)[6], g.seen, __ATOMIC_RELEASE);
    }
    if (hud) {
        if (!g.hud_tex) {
            g.hud_pixels = malloc(PANEL_BYTES);
            glGenTextures(1, &g.hud_tex);
            glBindTexture(GL_TEXTURE_2D, g.hud_tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 640, 480, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
        }
        glBindTexture(GL_TEXTURE_2D, g.hud_tex);
        if (hud_volume != g.hud_volume || hud_fast != g.hud_fast || hud_fps != g.hud_fps) {
            g.hud_volume = hud_volume;
            g.hud_fast = hud_fast;
            g.hud_fps = hud_fps;
            rgds_hud_draw(g.hud_pixels, hud_volume, hud_fast, hud_fps);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 640, 480, GL_RGBA, GL_UNSIGNED_BYTE, g.hud_pixels);
        }
        glUniform1i(g.use_tex, 1);
        glDrawArrays(GL_TRIANGLES, 0, 3);
    }
    glDisable(GL_BLEND);

    // test hook: the panels as shown, menu included
    static const char* shot;
    if (!shot) shot = getenv("SPIKE_MENUSHOT") ? getenv("SPIKE_MENUSHOT") : "";
    if (*shot && !access(shot, F_OK)) {
        char path[600];
        snprintf(path, sizeof path, "%s-%d.bmp", shot, panel);
        static u8 px[PANEL_BYTES];
        glReadPixels(0, 0, 640, 480, GL_BGRA, GL_UNSIGNED_BYTE, px);
        u8 head[54] = {'B', 'M'};
        u32 size = 54 + PANEL_BYTES, w = 640, h = 480;
        memcpy(head + 2, &size, 4);
        head[10] = 54;
        head[14] = 40;
        memcpy(head + 18, &w, 4);
        memcpy(head + 22, &h, 4);
        head[26] = 1;
        head[28] = 32;
        FILE* f = fopen(path, "wb");
        if (f) {
            fwrite(head, 1, 54, f);
            fwrite(px, 1, PANEL_BYTES, f);
            fclose(f);
        }
        if (panel == 1) unlink(shot);
    }
}
