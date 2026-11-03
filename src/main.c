#include <SDL3/SDL.h>
#include "rgds_bridge.h"
#include <dirent.h>
#include <sched.h>
#include <sys/resource.h>
#include <SDL3/SDL_main.h>
#include <getopt.h>
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#include <imgui/dcimgui.h>
#include <imgui/dcimgui_impl_opengl3.h>
#include <imgui/dcimgui_impl_sdl3.h>

#include "3ds.h"
#include "emulator.h"
#include "gui.h"
#include "video/renderer_gl.h"

#ifdef _WIN32
#define realpath(a, b) _fullpath(b, a, 4096)
#endif

SDL_Window* g_window;

SDL_JoystickID g_gamepad_id;
SDL_Gamepad* g_gamepad;

SDL_AudioStream* g_audio;
SDL_AudioStream* g_audio_input;

SDL_Camera* g_camera;

char* romfile_arg;

#define FREECAM_SPEED 5.0
#define FREECAM_ROTATE_SPEED 0.02

#ifdef GLDEBUGCTX
void glDebugOutput(GLenum source, GLenum type, unsigned int id, GLenum severity,
                   GLsizei length, const char* message, const void* userParam) {
    ldebug("[GLDEBUG]%d %d %d %d %s", source, type, id, severity, message);
}
#endif

// spike: wall-clock profiler with stacks for the main thread, from SPIKE_PROF_FROM, dumped at quit
#include <execinfo.h>
#include <signal.h>
extern int g_clipped;
static void spike_crash(int sig) {
    rgds_crash_note(sig == SIGABRT ? "abort" : "signal", nullptr, nullptr, nullptr);
    void* frames[32];
    int n = backtrace(frames, 32);
    fprintf(stderr, "spike crash signal %d\n", sig);
    backtrace_symbols_fd(frames, n, 2);
    FILE* m = fopen("/proc/self/maps", "r");
    char line[256];
    while (m && fgets(line, sizeof line, m)) {
        if (strstr(line, "r-xp")) fputs(line, stderr);
    }
    signal(sig, SIG_DFL);
    raise(sig);
}
#include <pthread.h>
#include <stdatomic.h>
#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <time.h>
#include <ucontext.h>
#define SPIKE_MAX 60000
static void* spike_stacks[SPIKE_MAX][20];
static unsigned char spike_depth[SPIKE_MAX];
static unsigned long spike_x1[SPIKE_MAX];
static volatile int spike_n = 0;
static void spike_prof_sig(int sig, siginfo_t* si, void* uc) {
    int i = spike_n;
    if (i >= SPIKE_MAX) return;
    spike_x1[i] = ((ucontext_t*) uc)->uc_mcontext.regs[1];
    spike_depth[i] = backtrace(spike_stacks[i], 20);
    spike_n = i + 1;
}
static timer_t spike_timer;
static void spike_prof_start_tid(pid_t tid) {
    void* warm[4];
    backtrace(warm, 4);
    struct sigaction sa = {.sa_sigaction = spike_prof_sig, .sa_flags = SA_SIGINFO | SA_RESTART};
    sigaction(SIGPROF, &sa, nullptr);
    struct sigevent ev = {.sigev_notify = SIGEV_THREAD_ID, .sigev_signo = SIGPROF};
    ev._sigev_un._tid = tid;
    timer_create(CLOCK_MONOTONIC, &ev, &spike_timer);
    long us = getenv("SPIKE_PROF_US") ? atol(getenv("SPIKE_PROF_US")) : 10000;
    struct itimerspec t = {{0, us * 1000}, {0, us * 1000}};
    timer_settime(spike_timer, 0, &t, nullptr);
}
static void spike_prof_start(void) {
    spike_prof_start_tid(syscall(SYS_gettid));
}
static void spike_prof_dump(void) {
    timer_delete(spike_timer);
    FILE* f = fopen("prof.txt", "w");
    if (!f) return;
    FILE* m = fopen("/proc/self/maps", "r");
    char line[512];
    while (m && fgets(line, sizeof line, m)) fprintf(f, "map %s", line);
    if (m) fclose(m);
    for (int i = 0; i < spike_n; i++) {
        fprintf(f, "st %lx", spike_x1[i]);
        for (int j = 0; j < spike_depth[i]; j++) fprintf(f, " %lx", (unsigned long) spike_stacks[i][j]);
        fprintf(f, "\n");
    }
    fclose(f);
}

// spike: scripted buttons, screenshots and a quit frame, for timed runs on the handheld
static struct { int frame; char keys[32]; } spike_script[1024];
static int spike_lines = -1, spike_at = 0, spike_frame = 0;
static const char* spike_keys(void) {
    if (spike_lines < 0) {
        spike_lines = 0;
        const char* path = getenv("SPIKE_INPUT");
        FILE* f = path ? fopen(path, "r") : nullptr;
        if (f) {
            while (spike_lines < 1024 &&
                   fscanf(f, "%d %31s", &spike_script[spike_lines].frame,
                          spike_script[spike_lines].keys) == 2)
                spike_lines++;
            fclose(f);
        }
    }
    while (spike_at < spike_lines && spike_script[spike_at].frame <= spike_frame) spike_at++;
    spike_frame++;
    return spike_at ? spike_script[spike_at - 1].keys : "-";
}
static int spike_slow_frames;
// only samples from long gaps between presents stay in the worker's profile
void spike_gap_hook(double ms) {
    static int kept, gap_ms = -1;
    if (gap_ms < 0) gap_ms = getenv("SPIKE_GAPPROF") ? atoi(getenv("SPIKE_GAPPROF")) : 0;
    if (!gap_ms) return;
    if (ms >= gap_ms) {
        spike_x1[spike_n < SPIKE_MAX ? spike_n : SPIKE_MAX - 1] = (unsigned long) ms;
        fprintf(stderr, "spike gap %.0f ms, %d samples\n", ms, spike_n - kept);
        kept = spike_n;
    } else {
        spike_n = kept;
    }
}
static Uint64 spike_win_main;
static int spike_win_audio = 1 << 30, spike_win_empty;
static const char* spike_tag = "";
static bool spike_noquit;
static SDL_Window* g_window2;
// spike: every run's numbers in logs/<time>-<rom>.log beside the settings, the newest 30 kept,
// because the launcher's log holds one run and the next launch overwrites it
static FILE* g_runlog;
static void runlog_open() {
    static bool tried;
    if (tried || !ctremu.romfilenoext) return;
    tried = true;
    mkdir("logs", 0777);
    struct dirent** names;
    int n = scandir("logs", &names, nullptr, alphasort);
    if (n >= 0) {
        for (int i = 0; i < n; i++) {
            const char* name = names[i]->d_name;
            if (name[0] != '.' && n - i > 29) {
                char path[300];
                snprintf(path, sizeof path, "logs/%s", name);
                unlink(path);
            }
            free(names[i]);
        }
        free(names);
    }
    const char* rom = strrchr(ctremu.romfilenoext, '/');
    rom = rom ? rom + 1 : ctremu.romfilenoext;
    char stamp[32];
    time_t now = time(nullptr);
    strftime(stamp, sizeof stamp, "%Y%m%d-%H%M%S", localtime(&now));
    char path[300];
    snprintf(path, sizeof path, "logs/%s-%.40s.log", stamp, rom);
    g_runlog = fopen(path, "w");
    if (g_runlog) fprintf(g_runlog, "Tanuki3DS %s %s\n", EMUVERSION, ctremu.romfilenoext);
}

void spike_logf(const char* fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    if (!g_runlog) runlog_open();
    if (g_runlog) {
        va_start(ap, fmt);
        vfprintf(g_runlog, fmt, ap);
        va_end(ap);
        fflush(g_runlog);
    }
}

static void spike_frame_done(int w, int h) {
    const char* every = getenv("SPIKE_SHOT_EVERY");
    const char* dir = getenv("SPIKE_SHOTS");
    const char* quit = getenv("SPIKE_QUIT_AT");
    // both panels of one present land in the same pair of files
    static int shot_frame;
    if (!spike_noquit && *spike_tag) {
        // second panel: keep the frame the first one saw
    } else {
        shot_frame = spike_frame;
    }
    if (every && dir && shot_frame % atoi(every) == 0) {
        char name[512];
        snprintf(name, sizeof name, "%s/%d%s.bmp", dir, shot_frame, spike_tag);
        FILE* f = fopen(name, "wb");
        u8* px = malloc(w * h * 4);
        if (f && px) {
            glReadPixels(0, 0, w, h, GL_RGBA, GL_UNSIGNED_BYTE, px);
            for (int i = 0; i < w * h; i++) {
                u8 t = px[i * 4];
                px[i * 4] = px[i * 4 + 2];
                px[i * 4 + 2] = t;
            }
            u32 size = 54 + w * h * 4;
            u8 hdr[54] = {'B', 'M'};
            memcpy(hdr + 2, &size, 4);
            hdr[10] = 54;
            hdr[14] = 40;
            memcpy(hdr + 18, &w, 4);
            memcpy(hdr + 22, &h, 4);
            hdr[26] = 1;
            hdr[28] = 32;
            fwrite(hdr, 1, 54, f);
            fwrite(px, 1, w * h * 4, f);
        }
        if (f) fclose(f);
        free(px);
    }
    if (spike_noquit) return;
    const char* from = getenv("SPIKE_PROF_FROM");
    if (from && spike_frame >= atoi(from) && !getenv("SPIKE_PROF_MAIN") &&
        !getenv("SPIKE_PROF_WORKER")) {
        static bool started;
        if (!started) spike_prof_start();
        started = true;
    }
    if (quit && spike_frame >= atoi(quit)) {
        spike_prof_dump();
        fflush(stdout);
        _exit(0);
    }
}

extern u64 g_jit_ns, g_jit_max;
extern int g_jit_blocks, g_jit_roots;
extern u64 g_jit_pre_ns;
extern int g_jit_pre_roots, g_jit_pre_blocks;
size_t rasArenaUsed();
void jit_list_start(const char* rom);
void jit_list_step(ArmCore* cpu, u64 budget_ns);
extern int gpu_worker_lost();
static SDL_GLContext g_glctx;
// spike: per present, for each panel in turn: drawing it, its swap
static Uint64 g_present_ns[2][2];
static int g_present_n;
static void worker_bind(bool on);
static void worker_present(int w, int h);

void rgds_worker_restart() {
    gpu_worker_start(&ctremu.system.gpu, worker_bind, worker_present);
}

void rgds_audio_hold(bool hold) {
    if (!g_audio) return;
    if (hold) SDL_PauseAudioStreamDevice(g_audio);
    else {
        SDL_ClearAudioStream(g_audio);
        SDL_ResumeAudioStreamDevice(g_audio);
    }
}
static SDL_GLContext g_helperctx;

static int list_tasks(int* out, int max) {
    int n = 0;
    DIR* d = opendir("/proc/self/task");
    if (!d) return 0;
    for (struct dirent* e; (e = readdir(d)) && n < max;) {
        if (e->d_name[0] != '.') out[n++] = atoi(e->d_name);
    }
    closedir(d);
    return n;
}

// a context with glthread brings its own thread, which does the real work
static void place_new_threads(const int* before, int nbefore, int cpu, int nice) {
    int now[256];
    int n = list_tasks(now, 256);
    for (int i = 0; i < n; i++) {
        bool old = false;
        for (int j = 0; j < nbefore; j++) old |= before[j] == now[i];
        if (old) continue;
        char path[64], comm[32] = "";
        snprintf(path, sizeof path, "/proc/self/task/%d/comm", now[i]);
        FILE* f = fopen(path, "r");
        if (f) {
            fgets(comm, sizeof comm, f);
            fclose(f);
        }
        if (strncmp(comm, "ctremu:gl", 9)) continue;
        if (getenv("SPIKE_PIN")) {
            cpu_set_t set;
            CPU_ZERO(&set);
            CPU_SET(cpu, &set);
            sched_setaffinity(now[i], sizeof set, &set);
        }
        if (nice) setpriority(PRIO_PROCESS, now[i], nice);
    }
}
static void helper_bind() {
    if (!SDL_GL_MakeCurrent(nullptr, g_helperctx)) {
        fprintf(stderr, "spike helper context: %s\n", SDL_GetError());
    }
}
// spike: the present thread. SDL's KMS swap waits for the previous flip of that panel, so two
// panels cost the caller up to two vblank waits a frame (11-12 ms of the renderer's 16 in NSMB2).
// The renderer now hands a copy of the screens to this thread and goes on with the next frame.
static struct {
    SDL_GLContext ctx;
    pthread_t thread;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int pending, pending_w, pending_h; // the newest frame waiting, -1 for none
    int pending_sets[2];
    int inuse;                         // the frame being drawn, -1 for none
    int inuse_sets[2];
    bool on;
    GLuint vao;
    _Atomic int presented, dropped;
    _Atomic Uint64 maxgap;
    Uint64 last;
} g_pt = {.pending = -1, .inuse = -1, .mu = PTHREAD_MUTEX_INITIALIZER, .cv = PTHREAD_COND_INITIALIZER};

// renderer thread: the sets a queued or drawing frame still refers to
static int present_busy_sets(int* out) {
    int n = 0;
    pthread_mutex_lock(&g_pt.mu);
    if (g_pt.pending >= 0) {
        out[n++] = g_pt.pending_sets[0];
        out[n++] = g_pt.pending_sets[1];
    }
    if (g_pt.inuse >= 0) {
        out[n++] = g_pt.inuse_sets[0];
        out[n++] = g_pt.inuse_sets[1];
    }
    pthread_mutex_unlock(&g_pt.mu);
    return n;
}

static void present_panels(const int sets[2], int w, int h) {
    renderer_gl_set_filter(&ctremu.system.gpu.gl, rgds_bridge_filter());
    static int order = -1;
    if (order < 0) order = getenv("SPIKE_SWAP_BT") ? 1 : 0;
    for (int step = 0; step < 2; step++) {
        int panel = step ^ order;
        SDL_Window* window = panel ? g_window2 : g_window;
        Uint64 t0 = SDL_GetTicksNS();
        SDL_GL_MakeCurrent(window, g_pt.ctx);
        rgds_bridge_layout(panel, &ctremu.screens[SCREEN_TOP], &ctremu.screens[SCREEN_BOT]);
        gpu_gl_present_draw(&ctremu.system.gpu.gl, sets, g_pt.vao);
        rgds_bridge_panel(panel);
        spike_tag = panel ? "b" : "t";
        spike_noquit = !step;
        if (!ctremu.pause) spike_frame_done(w, h);
        glFlush();
        Uint64 t1 = SDL_GetTicksNS();
        SDL_GL_SwapWindow(window);
        Uint64 t2 = SDL_GetTicksNS();
        g_present_ns[step][0] += t1 - t0;
        g_present_ns[step][1] += t2 - t1;
    }
    g_present_n++;
}

static void* present_main(void* arg) {
    pthread_setname_np(pthread_self(), "ctremu:present");
    if (getenv("SPIKE_PIN")) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(1, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    if (!SDL_GL_MakeCurrent(g_window, g_pt.ctx)) {
        fprintf(stderr, "spike present thread: %s\n", SDL_GetError());
        return nullptr;
    }
    glGenVertexArrays(1, &g_pt.vao);
    for (;;) {
        pthread_mutex_lock(&g_pt.mu);
        while (g_pt.pending < 0) pthread_cond_wait(&g_pt.cv, &g_pt.mu);
        int sets[2] = {g_pt.pending_sets[0], g_pt.pending_sets[1]};
        int w = g_pt.pending_w, h = g_pt.pending_h;
        g_pt.pending = -1;
        g_pt.inuse = 1;
        g_pt.inuse_sets[0] = sets[0];
        g_pt.inuse_sets[1] = sets[1];
        pthread_mutex_unlock(&g_pt.mu);
        present_panels(sets, w, h);
        gpu_gl_present_done(sets);
        pthread_mutex_lock(&g_pt.mu);
        g_pt.inuse = -1;
        pthread_mutex_unlock(&g_pt.mu);
        atomic_fetch_add(&g_pt.presented, 1);
        Uint64 now = SDL_GetTicksNS();
        if (g_pt.last && now - g_pt.last > atomic_load(&g_pt.maxgap)) {
            atomic_store(&g_pt.maxgap, now - g_pt.last);
        }
        g_pt.last = now;
    }
    return nullptr;
}

static void present_thread_start() {
    if (!g_pt.ctx) return;
    gpu_gl_present_direct(true, present_busy_sets);
    pthread_create(&g_pt.thread, nullptr, present_main, nullptr);
    g_pt.on = true;
}

// spike: SPIKE_PROF_GL=1 samples the first driver thread (ctremu:gl0) instead of the renderer
// the lowest task id with that name: the renderer context is created first, so its driver
// thread has the lowest id of the ctremu:gl0 threads
static pid_t find_task(const char* name) {
    DIR* d = opendir("/proc/self/task");
    if (!d) return 0;
    pid_t found = 0;
    for (struct dirent* e; (e = readdir(d));) {
        if (e->d_name[0] == '.') continue;
        char path[64], comm[32] = "";
        snprintf(path, sizeof path, "/proc/self/task/%s/comm", e->d_name);
        FILE* f = fopen(path, "r");
        if (!f) continue;
        if (fgets(comm, sizeof comm, f) && !strncmp(comm, name, strlen(name))) {
            pid_t t = atoi(e->d_name);
            if (!found || t < found) found = t;
        }
        fclose(f);
    }
    closedir(d);
    return found;
}
static void spike_prof_term(int sig) {
    spike_prof_dump();
    fprintf(stderr, "spike profile dumped on signal %d\n", sig);
    _exit(0);
}

static void worker_bind(bool on) {
    // with the present thread the renderer draws into framebuffer objects only, so it holds no surface
    if (g_pt.on) SDL_GL_MakeCurrent(on ? nullptr : g_window, on ? g_glctx : nullptr);
    else SDL_GL_MakeCurrent(g_window, on ? g_glctx : nullptr);
}
static void worker_present(int w, int h) {
    // spike: SPIKE_PROF_WORKER=1 with SPIKE_PROF_FROM samples this (the renderer) thread's stacks
    if ((getenv("SPIKE_PROF_WORKER") || getenv("SPIKE_PROF_GL")) && getenv("SPIKE_PROF_FROM") &&
        spike_frame >= atoi(getenv("SPIKE_PROF_FROM"))) {
        static bool started;
        if (!started) {
            // the dump happens at quit; a SIGTERM (Home, the test scripts) dumps too
            signal(SIGTERM, spike_prof_term);
            if (getenv("SPIKE_PROF_GL")) {
                pid_t tid = find_task("ctremu:gl0");
                fprintf(stderr, "spike profiling task %d\n", tid);
                if (tid) spike_prof_start_tid(tid);
            } else {
                spike_prof_start();
            }
        }
        started = true;
    }
    if (g_pt.on) {
        int sets[2];
        gpu_gl_present_finish(&ctremu.system.gpu.gl, sets);
        pthread_mutex_lock(&g_pt.mu);
        if (g_pt.pending >= 0) atomic_fetch_add(&g_pt.dropped, 1);
        g_pt.pending = 1;
        g_pt.pending_sets[0] = sets[0];
        g_pt.pending_sets[1] = sets[1];
        g_pt.pending_w = w;
        g_pt.pending_h = h;
        pthread_cond_signal(&g_pt.cv);
        pthread_mutex_unlock(&g_pt.mu);
        return;
    }
    if (g_window2) {
        // one 3DS screen per panel: the top one 1.6x, the bottom one 2x, unless
        // the game's Graphics page says otherwise
        renderer_gl_set_filter(&ctremu.system.gpu.gl, rgds_bridge_filter());
        static int order = -1;
        if (order < 0) order = getenv("SPIKE_SWAP_BT") ? 1 : 0;
        for (int step = 0; step < 2; step++) {
            int panel = step ^ order;
            SDL_Window* window = panel ? g_window2 : g_window;
            Uint64 t0 = SDL_GetTicksNS();
            if (step) SDL_GL_MakeCurrent(window, g_glctx);
            else if (panel) SDL_GL_MakeCurrent(window, g_glctx);
            rgds_bridge_layout(panel, &ctremu.screens[SCREEN_TOP], &ctremu.screens[SCREEN_BOT]);
            render_gl_main(&ctremu.system.gpu.gl);
            rgds_bridge_panel(panel);
            spike_tag = panel ? "b" : "t";
            spike_noquit = !step;
            if (!ctremu.pause) spike_frame_done(w, h);
            glFlush();
            Uint64 t1 = SDL_GetTicksNS();
            SDL_GL_SwapWindow(window);
            Uint64 t2 = SDL_GetTicksNS();
            g_present_ns[step][0] += t1 - t0;
            g_present_ns[step][1] += t2 - t1;
        }
        g_present_n++;
        SDL_GL_MakeCurrent(g_window, g_glctx);
        return;
    }
    render_gl_main(&ctremu.system.gpu.gl);
    if (!ctremu.pause) spike_frame_done(w, h);
    SDL_GL_SwapWindow(g_window);
}

void read_args(int argc, char** argv) {
    char c;
    while ((c = getopt(argc, argv, "hl")) != (char) -1) {
        switch (c) {
            case 'l':
                g_infologs = true;
                break;
        }
    }
    argc -= optind;
    argv += optind;
    if (argc >= 1) {
        romfile_arg = realpath(argv[0], nullptr);
    }
}

void hotkey_press(SDL_Keycode key) {
    if (ImGui_GetIO()->WantCaptureKeyboard) return;
    switch (key) {
        case SDLK_F5:
            ctremu.pause = !ctremu.pause;
            break;
        case SDLK_TAB:
            ctremu.fastforward = !ctremu.fastforward;
            break;
        case SDLK_F1:
            ctremu.pending_reset = true;
            break;
        case SDLK_F2:
            load_rom_dialog();
            break;
        case SDLK_F3:
            uistate.settings = true;
            break;
        case SDLK_F7:
            ctremu.freecam_enable = !ctremu.freecam_enable;
            glm_mat4_identity(ctremu.freecam_mtx);
            renderer_gl_update_freecam(&ctremu.system.gpu.gl);
            break;
        case SDLK_F6:
            ctremu.mute = !ctremu.mute;
            break;
        case SDLK_F9:
            ctremu.swapscreens = !ctremu.swapscreens;
            break;
        case SDLK_F10:
            ctremu.viewlayout = (ctremu.viewlayout + 1) % LAYOUT_MAX;
            break;
        case SDLK_F11:
            ctremu.fullscreen = !ctremu.fullscreen;
            SDL_SetWindowFullscreen(g_window, ctremu.fullscreen);
            break;
        case SDLK_ESCAPE:
            uistate.menubar = !uistate.menubar;
            break;
#ifdef AUDIO_DEBUG
        case SDLK_0 ... SDLK_9:
            g_dsp_chn_disable ^= BIT(key - SDLK_0);
            break;
#endif
        default:
            break;
    }
}

// spike: the RG DS's own buttons (RG DS OS device tree: BTN_EAST is A).
// Select + Start held for a second quits, Home quits at once.
#include <dirent.h>
#include <fcntl.h>
#include <linux/input.h>
static int spike_open_input(const char* want) {
    DIR* d = opendir("/dev/input");
    if (!d) return -1;
    int found = -1;
    for (struct dirent* e; found < 0 && (e = readdir(d));) {
        if (strncmp(e->d_name, "event", 5)) continue;
        char path[300], name[128] = {};
        snprintf(path, sizeof path, "/dev/input/%s", e->d_name);
        int fd = open(path, O_RDONLY | O_NONBLOCK);
        if (fd < 0) continue;
        ioctl(fd, EVIOCGNAME(sizeof name - 1), name);
        if (!strcmp(name, want)) found = fd;
        else close(fd);
    }
    closedir(d);
    return found;
}
// the bottom panel's touchscreen sits on i2c-5 (fe5e0000), as in the OS session script
static bool spike_touch(int* tx, int* ty) {
    static int fd = -2;
    static bool down;
    static int x, y;
    static struct input_absinfo ix, iy;
    if (fd == -2) {
        fd = -1;
        for (int i = 0; i < 16 && fd < 0; i++) {
            char link[64], real[300];
            snprintf(link, sizeof link, "/sys/class/input/event%d/device", i);
            if (!realpath(link, real) || !strstr(real, "fe5e0000.i2c")) continue;
            snprintf(link, sizeof link, "/dev/input/event%d", i);
            fd = open(link, O_RDONLY | O_NONBLOCK);
        }
        if (fd >= 0) {
            ioctl(fd, EVIOCGABS(ABS_X), &ix);
            ioctl(fd, EVIOCGABS(ABS_Y), &iy);
        }
        fprintf(stderr, "spike touch fd %d x %d..%d y %d..%d\n", fd, ix.minimum, ix.maximum,
                iy.minimum, iy.maximum);
    }
    struct input_event ev;
    while (fd >= 0 && read(fd, &ev, sizeof ev) == sizeof ev) {
        if (ev.type == EV_KEY && ev.code == BTN_TOUCH) down = ev.value;
        if (ev.type == EV_ABS && ev.code == ABS_X) x = ev.value;
        if (ev.type == EV_ABS && ev.code == ABS_Y) y = ev.value;
    }
    if (!down || ix.maximum <= ix.minimum || iy.maximum <= iy.minimum) return false;
    *tx = (x - ix.minimum) * (SCREEN_WIDTH_BOT - 1) / (ix.maximum - ix.minimum);
    *ty = (y - iy.minimum) * (SCREEN_HEIGHT - 1) / (iy.maximum - iy.minimum);
    return true;
}

static void spike_evdev(PadState* btn, int* cx, int* cy) {
    static int pad = -2, home = -1, held, quit_frames;
    static u32 keys;
    static int ax, ay, hx, hy;
    static struct input_absinfo ix, iy;
    if (pad == -2) {
        pad = spike_open_input("retrogame_joypad");
        home = spike_open_input("adc-keys-home");
        if (pad >= 0) {
            ioctl(pad, EVIOCGABS(ABS_X), &ix);
            ioctl(pad, EVIOCGABS(ABS_Y), &iy);
        }
        fprintf(stderr, "spike pad fd %d home fd %d\n", pad, home);
    }
    struct input_event ev;
    while (pad >= 0 && read(pad, &ev, sizeof ev) == sizeof ev) {
        if (ev.type == EV_KEY) {
            int bit = -1;
            switch (ev.code) {
                case BTN_EAST: bit = 0; break;
                case BTN_SOUTH: bit = 1; break;
                case BTN_NORTH: bit = 2; break;
                case BTN_WEST: bit = 3; break;
                case BTN_TL: bit = 4; break;
                case BTN_TR: bit = 5; break;
                case BTN_SELECT: bit = 6; break;
                case BTN_START: bit = 7; break;
                case BTN_DPAD_UP: bit = 8; break;
                case BTN_DPAD_DOWN: bit = 9; break;
                case BTN_DPAD_LEFT: bit = 10; break;
                case BTN_DPAD_RIGHT: bit = 11; break;
            }
            if (bit >= 0) keys = ev.value ? keys | BIT(bit) : keys & ~BIT(bit);
        } else if (ev.type == EV_ABS) {
            if (ev.code == ABS_X) ax = ev.value;
            if (ev.code == ABS_Y) ay = ev.value;
            if (ev.code == ABS_HAT0X) hx = ev.value;
            if (ev.code == ABS_HAT0Y) hy = ev.value;
        }
    }
    while (home >= 0 && read(home, &ev, sizeof ev) == sizeof ev) {
        // with the shared menu Home is the launcher's
        if (ev.type == EV_KEY && ev.value && !rgds_bridge_on()) ctremu.running = false;
    }
    btn->a |= keys >> 0 & 1;
    btn->b |= keys >> 1 & 1;
    btn->x |= keys >> 2 & 1;
    btn->y |= keys >> 3 & 1;
    btn->l |= keys >> 4 & 1;
    btn->r |= keys >> 5 & 1;
    btn->select |= keys >> 6 & 1;
    btn->start |= keys >> 7 & 1;
    btn->up |= (keys >> 8 & 1) | (hy < 0);
    btn->down |= (keys >> 9 & 1) | (hy > 0);
    btn->left |= (keys >> 10 & 1) | (hx < 0);
    btn->right |= (keys >> 11 & 1) | (hx > 0);
    if (ix.maximum > ix.minimum) {
        float x = (ax - (ix.minimum + ix.maximum) / 2.f) / ((ix.maximum - ix.minimum) / 2.f);
        if (x > -0.2f && x < 0.2f) x = 0;
        if (abs((int) (x * INT16_MAX)) > abs(*cx)) *cx = x * INT16_MAX;
    }
    if (iy.maximum > iy.minimum) {
        float y = (ay - (iy.minimum + iy.maximum) / 2.f) / ((iy.maximum - iy.minimum) / 2.f);
        if (y > -0.2f && y < 0.2f) y = 0;
        if (abs((int) (y * INT16_MAX)) > abs(*cy)) *cy = -y * INT16_MAX;
    }
    if (btn->select && btn->start) {
        if (++quit_frames > 60 && !rgds_bridge_on()) ctremu.running = false;
    } else {
        quit_frames = 0;
    }
    (void) held;
}

void update_input() {
    if (ImGui_GetIO()->WantCaptureKeyboard || ImGui_GetIO()->WantCaptureMouse) return;

    const bool* keys = SDL_GetKeyboardState(nullptr);

    PadState btn = {};
    int cx = 0;
    int cy = 0;

    if (!ctremu.freecam_enable) {
        btn.a = keys[ctremu.inputmap.kb.a];
        btn.b = keys[ctremu.inputmap.kb.b];
        btn.x = keys[ctremu.inputmap.kb.x];
        btn.y = keys[ctremu.inputmap.kb.y];
        btn.l = keys[ctremu.inputmap.kb.l];
        btn.r = keys[ctremu.inputmap.kb.r];
        btn.start = keys[ctremu.inputmap.kb.start];
        btn.select = keys[ctremu.inputmap.kb.select];
        btn.up = keys[ctremu.inputmap.kb.du];
        btn.down = keys[ctremu.inputmap.kb.dd];
        btn.left = keys[ctremu.inputmap.kb.dl];
        btn.right = keys[ctremu.inputmap.kb.dr];

        cx = (keys[ctremu.inputmap.kb.cr] - keys[ctremu.inputmap.kb.cl]) *
             INT16_MAX;
        cy = (keys[ctremu.inputmap.kb.cu] - keys[ctremu.inputmap.kb.cd]) *
             INT16_MAX;
        if (keys[ctremu.inputmap.kb.cmod]) {
            cx *= ctremu.inputmap.kb.cmodscale;
            cy /= ctremu.inputmap.kb.cmodscale;
        }
    } else {
        float speed = FREECAM_SPEED;
        if (keys[ctremu.inputmap.freecam.slow_mod]) speed /= 20;
        if (keys[ctremu.inputmap.freecam.fast_mod]) speed *= 20;

        vec3 t = {};
        if (keys[ctremu.inputmap.freecam.ml]) {
            t[0] = speed;
        }
        if (keys[ctremu.inputmap.freecam.mr]) {
            t[0] = -speed;
        }
        if (keys[ctremu.inputmap.freecam.md]) {
            t[1] = speed;
        }
        if (keys[ctremu.inputmap.freecam.mu]) {
            t[1] = -speed;
        }
        if (keys[ctremu.inputmap.freecam.mf]) {
            t[2] = speed;
        }
        if (keys[ctremu.inputmap.freecam.mb]) {
            t[2] = -speed;
        }

        mat4 r = GLM_MAT4_IDENTITY_INIT;
        if (keys[ctremu.inputmap.freecam.ld]) {
            glm_rotate_make(r, FREECAM_ROTATE_SPEED, GLM_XUP);
        }
        if (keys[ctremu.inputmap.freecam.lu]) {
            glm_rotate_make(r, -FREECAM_ROTATE_SPEED, GLM_XUP);
        }
        if (keys[ctremu.inputmap.freecam.ll]) {
            glm_rotate_make(r, -FREECAM_ROTATE_SPEED, GLM_YUP);
        }
        if (keys[ctremu.inputmap.freecam.lr]) {
            glm_rotate_make(r, FREECAM_ROTATE_SPEED, GLM_YUP);
        }
        if (keys[ctremu.inputmap.freecam.rl]) {
            glm_rotate_make(r, FREECAM_ROTATE_SPEED, GLM_ZUP);
        }
        if (keys[ctremu.inputmap.freecam.rr]) {
            glm_rotate_make(r, -FREECAM_ROTATE_SPEED, GLM_ZUP);
        }

        mat4 m;
        glm_translate_make(m, t);
        glm_mat4_mul(m, ctremu.freecam_mtx, ctremu.freecam_mtx);
        glm_mat4_mul(r, ctremu.freecam_mtx, ctremu.freecam_mtx);

        renderer_gl_update_freecam(&ctremu.system.gpu.gl);
    }

    // on the handheld its own pad is read directly: SDL's guess at it also made R2 an R
    if (g_gamepad && !rgds_bridge_on()) {
        btn.a |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_EAST);
        btn.b |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_SOUTH);
        btn.x |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_NORTH);
        btn.y |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_WEST);
        btn.start |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_START);
        btn.select |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_BACK);
        btn.left |=
            SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_DPAD_LEFT);
        btn.right |=
            SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_DPAD_RIGHT);
        btn.up |= SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_DPAD_UP);
        btn.down |=
            SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_DPAD_DOWN);
        btn.l |=
            SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
        btn.r |=
            SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);

        int x = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTX);
        if (abs(x) > abs(cx)) cx = x;
        int y = -SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFTY);
        if (abs(y) > abs(cy)) cy = y;

        int tl = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_LEFT_TRIGGER);
        if (tl > INT16_MAX / 10) btn.l = 1;
        int tr = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHT_TRIGGER);
        if (tr > INT16_MAX / 10) btn.r = 1;

        int rx = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTX);
        int ry = SDL_GetGamepadAxis(g_gamepad, SDL_GAMEPAD_AXIS_RIGHTY);
        if (rx > INT16_MAX / 2) btn.right = 1;
        if (rx < -INT16_MAX / 2) btn.left = 1;
        if (ry > INT16_MAX / 2) btn.down = 1;
        if (ry < -INT16_MAX / 2) btn.up = 1;
    }

    {
        char keys[32];
        strcpy(keys, spike_keys());
        for (char* k = strtok(keys, "+"); k; k = strtok(nullptr, "+")) {
            if (!strcmp(k, "A")) btn.a = 1;
            if (!strcmp(k, "B")) btn.b = 1;
            if (!strcmp(k, "X")) btn.x = 1;
            if (!strcmp(k, "Y")) btn.y = 1;
            if (!strcmp(k, "L")) btn.l = 1;
            if (!strcmp(k, "R")) btn.r = 1;
            if (!strcmp(k, "START")) btn.start = 1;
            if (!strcmp(k, "SELECT")) btn.select = 1;
            if (!strcmp(k, "LEFT")) btn.left = 1;
            if (!strcmp(k, "RIGHT")) btn.right = 1;
            if (!strcmp(k, "UP")) btn.up = 1;
            if (!strcmp(k, "DOWN")) btn.down = 1;
        }
    }

    spike_evdev(&btn, &cx, &cy);

    btn.cup = cy > INT16_MAX / 2;
    btn.cdown = cy < INT16_MIN / 2;
    btn.cleft = cx < INT16_MIN / 2;
    btn.cright = cx > INT16_MAX / 2;

    float xf, yf;
    bool pressed =
        SDL_GetMouseState(&xf, &yf) & SDL_BUTTON_MASK(SDL_BUTTON_LEFT);

    if (g_gamepad) {
        if (SDL_GetGamepadButton(g_gamepad, SDL_GAMEPAD_BUTTON_RIGHT_STICK)) {
            pressed = true;
        }
    }
    int tx = xf, ty = yf;
    if (g_window2) pressed = false; // read from the panel itself below

    if (pressed) {
        tx -= ctremu.screens[SCREEN_BOT].x;
        tx = tx * SCREEN_WIDTH_BOT / ctremu.screens[SCREEN_BOT].w;
        ty -= ctremu.screens[SCREEN_BOT].y;
        ty = ty * SCREEN_HEIGHT / ctremu.screens[SCREEN_BOT].h;
        if (tx < 0 || tx >= SCREEN_WIDTH_BOT || ty < 0 || ty >= SCREEN_HEIGHT) {
            pressed = false;
            tx = 0;
            ty = 0;
        }
    } else {
        tx = 0;
        ty = 0;
    }

    if (g_window2) pressed = spike_touch(&tx, &ty);

    float accel[3] = {};
    float gyro[3] = {};
    if (g_gamepad) {
        SDL_GetGamepadSensorData(g_gamepad, SDL_SENSOR_ACCEL, accel, 3);
        SDL_GetGamepadSensorData(g_gamepad, SDL_SENSOR_GYRO, gyro, 3);

        // measurements from 3ds
        // gyroscope
        // yaw left: +z
        // pitch down: +x
        // roll left: -y
        // accelerometer
        // resting: around 0, -500, 0
        // left: -x
        // up: -y
        // back: -z
        // clamp at 930

        const float accelClamp = 930;
        const float gravityRead = 500;
        const float accelScale = gravityRead / SDL_STANDARD_GRAVITY;

        accel[0] = glm_clamp(accel[0] * accelScale, -accelClamp, accelClamp);
        accel[1] = glm_clamp(accel[1] * -accelScale, -accelClamp, accelClamp);
        accel[2] = glm_clamp(accel[2] * accelScale, -accelClamp, accelClamp);

        const float gyroScale = 180 / M_PI * HID_GYRO_DPS_COEFF;
        gyro[0] *= -gyroScale;
        gyro[1] *= gyroScale;
        gyro[2] *= -gyroScale;
    }

    // hid updates inputs 4x per frame so we must simulate this
    // by calling each function 4 times
    for (int i = 0; i < 4; i++) {
        hid_update_pad(&ctremu.system, btn.w, cx, cy);
        hid_update_touch(&ctremu.system, tx, ty, pressed);
        hid_update_accel(&ctremu.system, accel[0], accel[1], accel[2]);
        hid_update_gyro(&ctremu.system, gyro[0], gyro[1], gyro[2]);
    }
}

void update_mic() {
    if (!ctremu.system.services.mic.sampling || !ctremu.micEnable) {
        if (!g_audio_input) return;
        SDL_ClearAudioStream(g_audio_input);
        SDL_PauseAudioStreamDevice(g_audio_input);
        return;
    }
    if (!g_audio_input) {
        SDL_AudioSpec as_in = {
            .format = SDL_AUDIO_S16,
            .channels = 1,
            .freq = SAMPLE_RATE,
        };
        g_audio_input = SDL_OpenAudioDeviceStream(
            SDL_AUDIO_DEVICE_DEFAULT_RECORDING, &as_in, nullptr, nullptr);
    }
    if (SDL_AudioStreamDevicePaused(g_audio_input)) {
        SDL_AudioFormat fmts[] = {SDL_AUDIO_U8, SDL_AUDIO_S16, SDL_AUDIO_S8,
                                  SDL_AUDIO_S16};
        SDL_AudioSpec as = {
            .format = fmts[ctremu.system.services.mic.encoding & 3],
            .channels = 1,
            .freq = ctremu.system.services.mic.sampleRate,
        };
        SDL_SetAudioStreamFormat(g_audio_input, &as, &as);
        SDL_SetAudioStreamGain(g_audio_input,
                               (float) ctremu.system.services.mic.gain / 64);
        SDL_ResumeAudioStreamDevice(g_audio_input);
    }
    int buflen = SDL_GetAudioStreamAvailable(g_audio_input);
    u8 buf[buflen];
    SDL_GetAudioStreamData(g_audio_input, buf, buflen);
    mic_send_data(&ctremu.system, buf, buflen);
}

void update_cam() {
    if (!ctremu.system.services.cam.capturing || !ctremu.camEnable) {
        return;
    }
    if (!g_camera) {
        SDL_CameraID* cams = SDL_GetCameras(nullptr);
        if (!cams || !cams[0]) return;
        SDL_CameraSpec cs = {
            .format = SDL_PIXELFORMAT_YUY2,
            .colorspace = SDL_COLORSPACE_YUV_DEFAULT,
            .width = 640,
            .height = 480,
            .framerate_numerator = 30,
            .framerate_denominator = 1,
        };
        g_camera = SDL_OpenCamera(cams[0], &cs);
    }

    auto image = SDL_AcquireCameraFrame(g_camera, nullptr);
    if (!image) return;
    auto scaled = SDL_ScaleSurface(image, ctremu.system.services.cam.width,
                                   ctremu.system.services.cam.height,
                                   SDL_SCALEMODE_LINEAR);
    if (ctremu.system.services.cam.rgb) {
        auto imageRGB = SDL_ConvertSurfaceAndColorspace(
            scaled, SDL_PIXELFORMAT_RGB565, nullptr, SDL_COLORSPACE_RGB_DEFAULT,
            0);
        SDL_DestroySurface(scaled);
        scaled = imageRGB;
    }
    cam_send_data(&ctremu.system, scaled->pixels);
    SDL_DestroySurface(scaled);
    SDL_ReleaseCameraFrame(g_camera, image);
}

static volatile Uint64 spike_post_gap;
static void SDLCALL spike_postmix(void* user, const SDL_AudioSpec* spec, float* buffer, int buflen) {
    static FILE* f;
    static int tried;
    static Uint64 last;
    (void) user; (void) spec;
    if (!tried) {
        tried = 1;
        if (getenv("SPIKE_POSTMIX")) f = fopen(getenv("SPIKE_POSTMIX"), "wb");
    }
    if (f) fwrite(buffer, 1, buflen, f);
    Uint64 now = SDL_GetTicksNS();
    if (last && now - last > spike_post_gap) spike_post_gap = now - last;
    last = now;
}

// spike: a recording of what the DSP hands over, to tell its own clicks from playback's
void audio_callback(s16 (*samples)[2], u32 count) {
    static FILE* wav;
    static int wav_tried;
    if (!wav_tried) {
        wav_tried = 1;
        if (getenv("SPIKE_WAV")) wav = fopen(getenv("SPIKE_WAV"), "wb");
    }
    if (wav) fwrite(samples, 4, count, wav);
    if (ctremu.fastforward || ctremu.mute) return;
    SDL_PutAudioStreamData(g_audio, samples, count * 2 * sizeof(s16));
}

int main(int argc, char** argv) {
    SDL_SetAppMetadataProperty(SDL_PROP_APP_METADATA_NAME_STRING, "Tanuki3DS");

    signal(SIGSEGV, spike_crash);
    signal(SIGBUS, spike_crash);
    signal(SIGABRT, spike_crash);
    read_args(argc, argv);
    {
        // spike: RG DS Core's session starts the launcher under taskset -c 0-2 and games inherit
        // that, so the pin to CPU 3 failed and three threads shared three cores: take every CPU
        // the cpuset allows (SPIKE_KEEPMASK=1 keeps the inherited mask)
        cpu_set_t before, all;
        CPU_ZERO(&before);
        sched_getaffinity(0, sizeof before, &before);
        int n = sysconf(_SC_NPROCESSORS_ONLN);
        CPU_ZERO(&all);
        for (int i = 0; i < n && i < CPU_SETSIZE; i++) CPU_SET(i, &all);
        if (!getenv("SPIKE_KEEPMASK") && !CPU_EQUAL(&before, &all)) {
            int r = sched_setaffinity(0, sizeof all, &all);
            fprintf(stderr, "spike cpus: %d of %d allowed at start, widened to all: %s\n",
                    CPU_COUNT(&before), n, r ? strerror(errno) : "ok");
        }
    }
    rgds_bridge_start();

    const char* basepath = SDL_GetBasePath();

    chdir(basepath);

    FILE* fp;
    if ((fp = fopen("portable.txt", "r"))) {
        fclose(fp);
    } else {
        char* prefpath = SDL_GetPrefPath("", "Tanuki3DS");
        chdir(prefpath);
        SDL_free(prefpath);
    }

#ifdef REDIRECTSTDOUT
    freopen("ctremu.log", "w", stdout);
#endif

    emulator_init();
    // spike: the internal resolution per game, from the launcher (RGDS_3DS_SCALE=1..4); the ini's
    // VideoScale otherwise
    if (getenv("RGDS_3DS_SCALE")) {
        int sc = atoi(getenv("RGDS_3DS_SCALE"));
        if (sc >= 1 && sc <= 4) ctremu.videoscale = sc;
        spike_logf("spike internal scale %d\n", ctremu.videoscale);
    }

    if (romfile_arg) {
        emulator_set_rom(romfile_arg);
        free(romfile_arg);
    }

    SDL_Init(SDL_INIT_VIDEO | SDL_INIT_AUDIO | SDL_INIT_GAMEPAD |
             SDL_INIT_CAMERA);

    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 3);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK,
                        SDL_GL_CONTEXT_PROFILE_CORE);
#ifdef GLDEBUGCTX
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_FLAGS, SDL_GL_CONTEXT_DEBUG_FLAG);
#endif
    int ndisplays = 0;
    SDL_DisplayID* displays = SDL_GetDisplays(&ndisplays);
    for (int i = 0; i < ndisplays; i++) {
        fprintf(stderr, "spike display %d: %s\n", i, SDL_GetDisplayName(displays[i]));
    }
    if (getenv("SPIKE_PANELS") && ndisplays >= 2) {
        // RG DS OS: DSI-2 is the top panel (SDL_KMSDRM_FIRST in its session script)
        int top = getenv("SPIKE_TOP_DISPLAY") ? atoi(getenv("SPIKE_TOP_DISPLAY")) : 1;
        for (int i = 0; i < 2; i++) {
            SDL_PropertiesID p = SDL_CreateProperties();
            SDL_DisplayID d = displays[i == 0 ? top : !top];
            SDL_SetNumberProperty(p, SDL_PROP_WINDOW_CREATE_X_NUMBER,
                                  SDL_WINDOWPOS_UNDEFINED_DISPLAY(d));
            SDL_SetNumberProperty(p, SDL_PROP_WINDOW_CREATE_Y_NUMBER,
                                  SDL_WINDOWPOS_UNDEFINED_DISPLAY(d));
            SDL_SetNumberProperty(p, SDL_PROP_WINDOW_CREATE_WIDTH_NUMBER, 640);
            SDL_SetNumberProperty(p, SDL_PROP_WINDOW_CREATE_HEIGHT_NUMBER, 480);
            SDL_SetBooleanProperty(p, SDL_PROP_WINDOW_CREATE_OPENGL_BOOLEAN, true);
            SDL_SetBooleanProperty(p, SDL_PROP_WINDOW_CREATE_FULLSCREEN_BOOLEAN, true);
            SDL_Window* w = SDL_CreateWindowWithProperties(p);
            SDL_DestroyProperties(p);
            if (i == 0) g_window = w;
            else g_window2 = w;
            fprintf(stderr, "spike window %d on display %u: %p\n", i, d, (void*) w);
        }
    } else {
        g_window = SDL_CreateWindow("Tanuki3DS", ctremu.windowW, ctremu.windowH,
                                    SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
        SDL_SetWindowPosition(g_window, SDL_WINDOWPOS_CENTERED,
                              SDL_WINDOWPOS_CENTERED);
    }

    int tasks[256];
    int ntasks = list_tasks(tasks, 256);
    SDL_GLContext glcontext = SDL_GL_CreateContext(g_window);
    place_new_threads(tasks, ntasks, 0, 0);
    g_glctx = glcontext;
    if (!glcontext) {
        SDL_Quit();
        lerror("could not create gl context");
        return 1;
    }

    gladLoadGLLoader((void*) SDL_GL_GetProcAddress);

    if (getenv("SPIKE_NOGUI") && !getenv("SPIKE_NOASYNC")) {
        extern void (*gpu_gl_helper_bind)(void);
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
        ntasks = list_tasks(tasks, 256);
        g_helperctx = SDL_GL_CreateContext(g_window);
        place_new_threads(tasks, ntasks, 1, 5);
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 0);
        SDL_GL_MakeCurrent(g_window, glcontext);
        if (g_helperctx) gpu_gl_helper_bind = helper_bind;
        else fprintf(stderr, "spike no helper context: %s\n", SDL_GetError());
    }
    if (g_window2 && getenv("SPIKE_NOGUI") && !getenv("SPIKE_NOPRESENTTHREAD")) {
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);
        ntasks = list_tasks(tasks, 256);
        g_pt.ctx = SDL_GL_CreateContext(g_window);
        place_new_threads(tasks, ntasks, 1, 0);
        SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 0);
        SDL_GL_MakeCurrent(g_window, glcontext);
        if (!g_pt.ctx) fprintf(stderr, "spike no present context: %s\n", SDL_GetError());
    }

#ifdef GLDEBUGCTX
    glEnable(GL_DEBUG_OUTPUT);
    glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
    glDebugMessageCallback(glDebugOutput, nullptr);
    glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DONT_CARE, 0, nullptr,
                          GL_TRUE);
#endif

    if (ctremu.vsync) {
        if (!SDL_GL_SetSwapInterval(-1)) SDL_GL_SetSwapInterval(1);
    } else {
        SDL_GL_SetSwapInterval(0);
    }

    glClear(GL_COLOR_BUFFER_BIT);
    SDL_GL_SwapWindow(g_window);

    SDL_AudioSpec as = {
        .format = SDL_AUDIO_S16,
        .channels = 2,
        .freq = SAMPLE_RATE,
    };
    if (getenv("SPIKE_NOGUI")) {
        // spike: the handheld's mixer runs at 48 kHz. Opened any other way the sound gets a second, cruder
        // resampling in ALSA's plug on the way there.
        SDL_AudioSpec devspec = {.format = SDL_AUDIO_S16, .channels = 2, .freq = 48000};
        SDL_AudioDeviceID dev = SDL_OpenAudioDevice(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &devspec);
        g_audio = SDL_CreateAudioStream(&as, &devspec);
        SDL_BindAudioStream(dev, g_audio);
    } else {
        g_audio = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &as,
                                            nullptr, nullptr);
    }

    SDL_ResumeAudioStreamDevice(g_audio);
    ctremu.audio_cb = audio_callback;
    {
        // spike: what the device really gets, and how evenly
        SDL_AudioSpec dev;
        int frames = 0;
        SDL_GetAudioDeviceFormat(SDL_GetAudioStreamDevice(g_audio), &dev, &frames);
        fprintf(stderr, "spike audio device %d Hz %d ch fmt %x, %d frames\n", dev.freq, dev.channels,
                dev.format, frames);
        SDL_SetAudioPostmixCallback(SDL_GetAudioStreamDevice(g_audio), spike_postmix, nullptr);
    }
    fprintf(stderr, "spike audio driver %s device %s stream %p\n", SDL_GetCurrentAudioDriver(),
            SDL_GetAudioDeviceName(SDL_GetAudioStreamDevice(g_audio)), (void*) g_audio);

    ImGui_CreateContext(nullptr);
    ImGui_GetIO()->ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    ImGui_GetIO()->ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;
    ImGui_GetIO()->ConfigViewportsNoDecoration = false;
    cImGui_ImplSDL3_InitForOpenGL(g_window, glcontext);
    cImGui_ImplOpenGL3_Init();

    setup_gui_theme();

    Uint64 prev_frame_time = SDL_GetTicksNS();
    Uint64 prev_fps_update = prev_frame_time;
    Uint64 prev_fps_frame = 0;
    const Uint64 frame_ticks = SDL_NS_PER_SECOND / FPS;
    Uint64 frame = 0;
    double avg_frame_time = 0;
    int avg_frame_time_ct = 0;

    ldebug("Tanuki3DS %s", EMUVERSION);

    ctremu.running = true;
    while (ctremu.running) {

        int exc;
        if ((exc = setjmp(ctremu.exceptionJmp))) {
            // on the handheld there is no window to say it in and nothing to reset into: back home
            if (rgds_bridge_on()) {
                const char* why = exc == EXC_EXIT    ? "the game exited"
                                  : exc == EXC_MEM   ? "invalid memory access"
                                  : exc == EXC_ERRF  ? "the game reported a fatal error"
                                  : "the game terminated";
                rgds_crash_note(why, nullptr, (void*) (uintptr_t) ctremu.system.cpu.pc, nullptr);
                fprintf(stderr, "rgds bridge: %s, returning home\n", why);
                fflush(nullptr);
                _exit(exc == EXC_EXIT ? 0 : 1);
            }
            emulator_set_rom(nullptr);
            static const char* errmess[] = {
                [EXC_MEM] = "Invalid memory access.",
                [EXC_EXIT] = "The application has exited.",
                [EXC_ERRF] = "The application has reported a fatal error.",
                [EXC_BREAK] = "The application has terminated.",
            };
            SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Tanuki3DS",
                                     errmess[exc], g_window);
        }

        if (ctremu.pending_reset) {
            ctremu.pending_reset = false;
            if (emulator_reset()) {
                ctremu.pause = false;
                rgds_bridge_autoload();
                uistate.menubar = !ctremu.initialized;
            } else {
                SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "Tanuki3DS",
                                         "ROM loading failed", g_window);
                ctremu.pause = true;
                uistate.menubar = true;
            }
            SDL_RaiseWindow(g_window);
            SDL_ClearAudioStream(g_audio);
        }

        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            bool forward_imgui = true;
            switch (e.type) {
                case SDL_EVENT_QUIT:
                    ctremu.running = false;
                    break;
                case SDL_EVENT_KEY_DOWN:
                    if (uistate.waiting_key) {
                        *uistate.waiting_key = e.key.scancode;
                        uistate.waiting_key = nullptr;
                        forward_imgui = false;
                    } else {
                        hotkey_press(e.key.key);
                    }
                    break;
                case SDL_EVENT_GAMEPAD_ADDED:
                    if (!g_gamepad) {
                        g_gamepad_id = e.gdevice.which;
                        g_gamepad = SDL_OpenGamepad(g_gamepad_id);
                        SDL_SetGamepadSensorEnabled(g_gamepad, SDL_SENSOR_ACCEL,
                                                    true);
                        SDL_SetGamepadSensorEnabled(g_gamepad, SDL_SENSOR_GYRO,
                                                    true);
                    }
                    break;
                case SDL_EVENT_GAMEPAD_REMOVED:
                    if (g_gamepad && e.gdevice.which == g_gamepad_id) {
                        g_gamepad = nullptr;
                    }
                    break;
                case SDL_EVENT_DROP_FILE:
                    emulator_set_rom(e.drop.data);
                    break;
            }
            if (forward_imgui) cImGui_ImplSDL3_ProcessEvent(&e);
        }

        if (!ctremu.initialized) ctremu.pause = true;

        SDL_GetWindowSizeInPixels(g_window, &ctremu.windowW, &ctremu.windowH);
        // the worker sets the rectangles itself on two panels
        if (!g_window2) emulator_calc_viewports();

        // spike: the worker owns the gl context, so only without the imgui pass
        if (ctremu.initialized && !gpu_worker_on() && getenv("SPIKE_NOGUI") &&
            !getenv("SPIKE_NOWORKER")) {
            if (g_pt.ctx && !g_pt.on) {
                // the present thread takes the window surfaces; the renderer goes surfaceless
                SDL_GL_MakeCurrent(g_window, nullptr);
                present_thread_start();
            }
            gpu_worker_start(&ctremu.system.gpu, worker_bind, worker_present);
            if (getenv("SPIKE_PIN")) {
                // the emulation core, as RG DS Core's session script keeps CPU 3 for it
                cpu_set_t set;
                CPU_ZERO(&set);
                CPU_SET(3, &set);
                pthread_setaffinity_np(pthread_self(), sizeof set, &set);
            }
        }

        if (getenv("SPIKE_PROF_MAIN") && getenv("SPIKE_PROF_FROM") &&
            spike_frame == atoi(getenv("SPIKE_PROF_FROM"))) {
            spike_prof_start();
        }

        bool menu_held = rgds_bridge_tick();
        if (menu_held) gpu_worker_start_frame(&ctremu.system.gpu);

        if (!ctremu.pause && !menu_held) {
            update_input();
            update_mic();
            update_cam();

            gpu_worker_start_frame(&ctremu.system.gpu);
            if (getenv("SPIKE_DUMP_AT")) {
                int at = atoi(getenv("SPIKE_DUMP_AT"));
                g_draw_dump = spike_frame >= at && spike_frame < at + 3;
            }

            Uint64 frame_start = SDL_GetTicksNS();
            // only samples from inside slow frames stay in the slow-frame profile
            static int prof_kept;
            if (getenv("SPIKE_SLOWPROF")) spike_n = prof_kept;
            e3ds_run_frame(&ctremu.system);
            frame++;
            Uint64 frame_time = SDL_GetTicksNS() - frame_start;
            {
                static int slow_ms = -1;
                if (slow_ms < 0) {
                    slow_ms = getenv("SPIKE_SLOWPROF") ? atoi(getenv("SPIKE_SLOWPROF")) : 0;
                }
                if (slow_ms && frame_time >= slow_ms * SDL_NS_PER_MS) {
                    prof_kept = spike_n;
                    spike_slow_frames++;
                }
                if (frame_time > spike_win_main) spike_win_main = frame_time;
                int q = SDL_GetAudioStreamQueued(g_audio) / 4;
                if (q < spike_win_audio) spike_win_audio = q;
                if (q == 0) spike_win_empty++;
            }
            avg_frame_time += (double) frame_time / SDL_NS_PER_MS;
            avg_frame_time_ct++;
        }
        if (gpu_worker_on()) {
            // presented on the worker
        } else if (ctremu.initialized) {
            render_gl_main(&ctremu.system.gpu.gl);
        } else {
            glClear(GL_COLOR_BUFFER_BIT);
        }

        // spike: no menus on the handheld, so no UI pass
        static int nogui = -1;
        if (nogui < 0) nogui = getenv("SPIKE_NOGUI") != nullptr;
        if (!nogui || !ctremu.initialized) {
        cImGui_ImplOpenGL3_NewFrame();
        cImGui_ImplSDL3_NewFrame();
        ImGui_NewFrame();

        draw_menubar();

        if (!ctremu.initialized) draw_gamelist();

        draw_gui();

        ImGui_Render();
        cImGui_ImplOpenGL3_RenderDrawData(ImGui_GetDrawData());

        if (ImGui_GetIO()->ConfigFlags & ImGuiConfigFlags_ViewportsEnable) {
            ImGui_UpdatePlatformWindows();
            ImGui_RenderPlatformWindowsDefault();
            SDL_GL_MakeCurrent(g_window, glcontext);
        }
        }

        if (gpu_worker_on()) {
            gpu_worker_present(ctremu.windowW, ctremu.windowH);
        } else {
            if (!ctremu.pause) spike_frame_done(ctremu.windowW, ctremu.windowH);
            SDL_GL_SwapWindow(g_window);
        }

        Uint64 elapsed = SDL_GetTicksNS() - prev_fps_update;
        if (!ctremu.pause && elapsed >= SDL_NS_PER_SECOND / 2) {
            prev_fps_update = SDL_GetTicksNS();

            double fps =
                (double) SDL_NS_PER_SECOND * (frame - prev_fps_frame) / elapsed;

            char* wintitle;
            asprintf(&wintitle, "Tanuki3DS %s | %s | %.2lf FPS, %.3lf ms",
                     EMUVERSION, ctremu.system.romimage.name, fps,
                     avg_frame_time / avg_frame_time_ct);
            SDL_SetWindowTitle(g_window, wintitle);
            {
                char stamp[16];
                time_t now = time(nullptr);
                strftime(stamp, sizeof stamp, "%H:%M:%S", localtime(&now));
                spike_logf("%s %s\n", stamp, wintitle); // spike: fps on stderr and the run log
            }
            if (gpu_worker_on()) {
                double busy, present, wait;
                int copies, wraps, n = frame - prev_fps_frame;
                gpu_worker_stats(&busy, &present, &wait, &copies, &wraps);
                if (n) {
                    spike_logf(
                            "worker: busy %.1f present %.1f, main wait %.1f ms/frame, "
                            "copies %d wraps %d drawn %d/%d lutmiss %d guest %d "
                            "draws %d creates %d partial %d behind %.2f audio %d "
                            "| maxmain %.0f maxgap %.0f minaudio %d empty %d links %d "
                            "ms: link %.0f compile %.0f tex %.0f clip %d auddev %.0f "
                            "progwait %.0f progs %d/%d jit %d/%d ms %.1f max %.1f "
                            "pre %d/%d ms %.1f arena %.1f swaps %.1f+%.1f %.1f+%.1f lost %d "
                            "texdiff %d tiles %d ms %.0f new/stale/rehash %d/%d/%d "
                            "pt %d/%d gap %.0f lit %d gpu %.1f glcalls %d/%d relaid %d\n",
                            busy / n, present / n, wait / n, copies, wraps,
                            gpu_worker_drawn(), n, g_lut_misses, gpu_worker_transfers(),
                            g_draws / (n ? n : 1), g_tex_creates, g_tex_partial,
                            gpu_worker_behind(), SDL_GetAudioStreamQueued(g_audio) / 4,
                            spike_win_main / 1e6, gpu_worker_maxgap(), spike_win_audio,
                            spike_win_empty, g_links, g_ns_shaders / 1e6, g_ns_compile / 1e6,
                            g_ns_textures / 1e6, g_clipped, spike_post_gap / 1e6,
                            g_ns_progwait / 1e6, g_prog_ready, g_prog_total,
                            g_jit_roots, g_jit_blocks, g_jit_ns / 1e6, g_jit_max / 1e6,
                            g_jit_pre_roots, g_jit_pre_blocks, g_jit_pre_ns / 1e6,
                            rasArenaUsed() / 1048576.0,
                            g_present_ns[0][0] / 1e6 / (g_present_n ? g_present_n : 1),
                            g_present_ns[0][1] / 1e6 / (g_present_n ? g_present_n : 1),
                            g_present_ns[1][0] / 1e6 / (g_present_n ? g_present_n : 1),
                            g_present_ns[1][1] / 1e6 / (g_present_n ? g_present_n : 1),
                            gpu_worker_lost(), g_tex_diffs, g_tex_difftiles, g_ns_texdiff / 1e6,
                            g_tex_new, g_tex_stalefull, g_tex_rehashfull,
                            atomic_exchange(&g_pt.presented, 0), atomic_exchange(&g_pt.dropped, 0),
                            atomic_exchange(&g_pt.maxgap, 0) / 1e6, g_draws_lit / (n ? n : 1),
                            g_gpu_n ? g_gpu_ns / 1e6 / g_gpu_n : 0.0,
                            g_gls_calls - g_gls_skipped, g_gls_calls, g_draws_relaid);
                    g_gls_calls = g_gls_skipped = 0;
                    g_draws_relaid = 0;
                    g_draws_lit = 0;
                    g_gpu_ns = 0;
                    g_gpu_n = 0;
                    g_tex_diffs = g_tex_difftiles = g_tex_new = g_tex_stalefull = g_tex_rehashfull = 0;
                    g_ns_texdiff = 0;
                    memset(g_present_ns, 0, sizeof g_present_ns);
                    g_present_n = 0;
                    g_jit_pre_roots = g_jit_pre_blocks = 0;
                    g_jit_pre_ns = 0;
                    g_jit_roots = g_jit_blocks = 0;
                    g_jit_ns = g_jit_max = 0;
                    g_ns_progwait = 0;
                    spike_post_gap = 0;
                    g_ns_shaders = g_ns_compile = g_ns_textures = 0;
                    g_clipped = 0;
                    spike_win_main = 0;
                    spike_win_audio = 1 << 30;
                    spike_win_empty = 0;
                    g_links = 0;
                    g_lut_misses = 0;
                    g_draws = g_tex_creates = g_tex_partial = 0;
                }
            }
            free(wintitle);
            prev_fps_frame = frame;
            avg_frame_time = 0;
            avg_frame_time_ct = 0;
        } else if (!ctremu.initialized) {
            char* wintitle;
            asprintf(&wintitle, "Tanuki3DS %s", EMUVERSION);
            SDL_SetWindowTitle(g_window, wintitle);
            free(wintitle);
        }

        if (ctremu.fastforward && !ctremu.pause) {
            gpu_worker_start_frame(&ctremu.system.gpu);
            while (SDL_GetTicksNS() - prev_frame_time < frame_ticks) {
                Uint64 frame_start = SDL_GetTicksNS();
                e3ds_run_frame(&ctremu.system);
                frame++;
                Uint64 frame_time = SDL_GetTicksNS() - frame_start;
                avg_frame_time += (double) frame_time / SDL_NS_PER_MS;
                avg_frame_time_ct++;
            }
        } else if (gpu_worker_on() && !getenv("SPIKE_AUDIOPACE")) {
            // spike: pacing on the audio queue hands out frames in the bursts the
            // device takes samples in. Run on a steady clock, make up for a short
            // stall by not waiting, and hold the queue with a resampling ratio too
            // small to hear, so the clock never has to move.
            static Uint64 next_frame;
            static float queue_avg = -1;
            float queued = SDL_GetAudioStreamQueued(g_audio) / 4;
            if (queue_avg < 0) queue_avg = queued;
            queue_avg += (queued - queue_avg) * 0.02f;
            const float target = SAMPLE_RATE * 0.06f;
            static int ratio_wait;
            if (++ratio_wait >= 15) {
                ratio_wait = 0;
                // down to 3% slow when the game cannot keep 60 for a while: a lower pitch for a
                // moment is far less audible than the queue running dry
                float ratio = 1 + (queue_avg - target) / target * 0.04f;
                if (ratio < 0.97f) ratio = 0.97f;
                if (ratio > 1.01f) ratio = 1.01f;
                if (!getenv("SPIKE_NORATIO")) SDL_SetAudioStreamFrequencyRatio(g_audio, ratio);
            }

            Uint64 now = SDL_GetTicksNS();
            next_frame += frame_ticks;
            if (getenv("SPIKE_NOGUI") && ctremu.initialized && !getenv("SPIKE_NOJITLIST")) {
                static bool listed;
                if (!listed && ctremu.romfilenoext) {
                    listed = true;
                    jit_list_start(ctremu.romfilenoext);
                }
                Uint64 tnow = SDL_GetTicksNS();
                if (next_frame > tnow + 2 * SDL_NS_PER_MS && !rgds_bridge_busy()) {
                    jit_list_step(&ctremu.system.cpu, next_frame - tnow - 2 * SDL_NS_PER_MS);
                }
            }
            if (next_frame + 100 * SDL_NS_PER_MS < now || next_frame > now + 2 * frame_ticks) {
                next_frame = now;
            } else if (next_frame > now) {
                SDL_DelayPrecise(next_frame - now);
            }
        } else {
            if (ctremu.audiosync && !ctremu.mute && ctremu.initialized &&
                SDL_GetAudioStreamQueued(g_audio) != 0) {
                while (SDL_GetAudioStreamQueued(g_audio) > 100 * FRAME_SAMPLES)
                    SDL_Delay(1);
            } else if (!ctremu.vsync) {
                Uint64 elapsed = SDL_GetTicksNS() - prev_frame_time;
                Sint64 wait = frame_ticks - elapsed;
                if (wait > 0) {
                    SDL_DelayPrecise(wait);
                }
            }
        }

        prev_frame_time = SDL_GetTicksNS();
    }

    if (getenv("SPIKE_PROF_FROM")) {
        fprintf(stderr, "spike slow frames %d\n", spike_slow_frames);
        spike_prof_dump();
    }
    if (gpu_worker_on()) {
        // spike: tearing down under the renderer thread crashes, and nothing after this matters
        // but the saves, which are stdio files
        fflush(nullptr);
        { extern void gpu_gl_print_texstat(void); gpu_gl_print_texstat(); }
        fprintf(stderr, "spike clean exit\n");
        _exit(0);
    }
    gpu_worker_stop();

    cImGui_ImplOpenGL3_Shutdown();
    cImGui_ImplSDL3_Shutdown();
    ImGui_DestroyContext(nullptr);

    if (g_camera) SDL_CloseCamera(g_camera);

    if (g_audio_input) SDL_DestroyAudioStream(g_audio_input);
    SDL_DestroyAudioStream(g_audio);

    SDL_GL_DestroyContext(glcontext);
    SDL_DestroyWindow(g_window);
    SDL_CloseGamepad(g_gamepad);

    SDL_Quit();

    emulator_quit();

    return 0;
}
