#include "gpu.h"

#include "3ds.h"
#include "emulator.h"
#include "kernel/memory.h"

#include "renderer_gl.h"
#include "shader.h"
#include "shaderdec.h"
#include "shadergen_fs.h"

#include "gpuptr.inc"
#include "gpu_hash.h"

#include <linux/futex.h>
#include <sched.h>
#include <stdatomic.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

// spike: the renderer on a second thread. The thread owns the whole GPU struct
// and the gl context. The emulation thread never touches them: for each command
// list it follows the jumps, copies the words into cmd_ring and copies each
// draw's vertices and indices into a gl buffer, so the game can overwrite its
// buffers right after the (immediate) interrupt. Textures are still read from
// guest memory late.
#define GW_MSGS (1u << 15)
#define GW_SNAPS (1u << 14)
#define GW_CMD_RING (16u << 20)
#define GW_FLUSH_DRAWS 1024

enum {
    GW_CMDS,
    GW_INVAL,
    GW_REHASH,
    GW_TRANSFER,
    GW_LCD,
    GW_TEXCOPY,
    GW_CLEAR,
    GW_START_FRAME,
    GW_PRESENT,
    GW_SYNC,
    GW_QUIT,
};

typedef struct {
    u32 type;
    u32 a[8];
} GWMsg;

static struct {
    bool on;
    pthread_t thread;
    void (*bind)(bool);
    void (*present)(int, int);

    GWMsg* msgs;
    alignas(64) atomic_uint msg_head;
    alignas(64) atomic_uint msg_tail;

    u32* cmd;
    u64 cmd_head;
    alignas(64) _Atomic u64 cmd_tail;

    DrawSnap* snaps;
    u32 snap_head;
    u32 snap_read;
    alignas(64) atomic_uint snap_tail;
    int pending_draws;

    u32 presents;
    alignas(64) atomic_uint presents_done;

    GPURegs regs; // what the command lists have written so far

    bool skip;
    bool can_skip;
    u32 behind_sum, behind_n;
    u64 last_present;
    _Atomic u64 maxgap;
    u32 skip_after;
    atomic_int drawn, transfers, lost;
    bool new_image, only_new;

    _Atomic u64 busy_ns, present_ns;
    u64 wait_ns;
    atomic_int copies, wraps;
} g_w;

static thread_local bool g_is_worker;
void spike_gap_hook(double ms);

static u64 gw_now() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000ull + ts.tv_nsec;
}

static inline void gw_relax(int* n) {
    if ((*n)++ < 2000) {
        sched_yield();
    } else {
        usleep(20);
    }
}

// spike: the renderer thread used to spin on sched_yield while waiting for commands, and the
// main thread while waiting for presents: two cores busy with nothing, which is heat, which
// trips the chip's 70 C throttle after about 90 s of a race. Now each side spins briefly and
// then sleeps on a futex; the other side wakes it only when it said it was asleep
// (both store then load, so no wake-up is lost). SPIKE_SPINWAIT=1 keeps the old spinning.
static atomic_uint g_worker_sleeping, g_main_sleeping;
static int g_spinwait = -1;
static inline void gw_cpu_relax(void) {
#if defined(__aarch64__)
    __asm__ volatile("yield");
#endif
}
static void gw_futex_wait(atomic_uint* addr, u32 expected) {
    syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, expected, NULL, NULL, 0);
}
static void gw_futex_wake(atomic_uint* addr) {
    syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, 1, NULL, NULL, 0);
}
// main thread: waits while *addr still reads `seen`
static void gw_wait_main(atomic_uint* addr, u32 seen) {
    if (g_spinwait < 0) g_spinwait = getenv("SPIKE_SPINWAIT") != nullptr;
    for (int i = 0; i < 1000; i++) {
        if (atomic_load_explicit(addr, memory_order_acquire) != seen) return;
        gw_cpu_relax();
    }
    if (g_spinwait) {
        int n = 0;
        while (atomic_load_explicit(addr, memory_order_acquire) == seen) gw_relax(&n);
        return;
    }
    atomic_store_explicit(&g_main_sleeping, 1, memory_order_seq_cst);
    if (atomic_load_explicit(addr, memory_order_seq_cst) == seen) gw_futex_wait(addr, seen);
    atomic_store_explicit(&g_main_sleeping, 0, memory_order_seq_cst);
}
static inline void gw_wake_main(atomic_uint* addr) {
    if (atomic_load_explicit(&g_main_sleeping, memory_order_seq_cst)) gw_futex_wake(addr);
}

static void gw_push(u32 type, u32 a0, u32 a1, u32 a2, u32 a3, u32 a4, u32 a5,
                    u32 a6, u32 a7) {
    u32 head = atomic_load_explicit(&g_w.msg_head, memory_order_relaxed);
    if (head - atomic_load_explicit(&g_w.msg_tail, memory_order_acquire) >=
        GW_MSGS) {
        u64 t0 = gw_now();
        int n = 0;
        while (head - atomic_load_explicit(&g_w.msg_tail,
                                           memory_order_acquire) >= GW_MSGS)
            gw_relax(&n);
        g_w.wait_ns += gw_now() - t0;
    }
    g_w.msgs[head % GW_MSGS] =
        (GWMsg) {type, {a0, a1, a2, a3, a4, a5, a6, a7}};
    atomic_store_explicit(&g_w.msg_head, head + 1, memory_order_seq_cst);
    if (atomic_load_explicit(&g_worker_sleeping, memory_order_seq_cst)) gw_futex_wake(&g_w.msg_head);
}

bool gpu_worker_is_this_thread() {
    return g_is_worker;
}

bool gpu_worker_on() {
    return g_w.on;
}

double gpu_worker_maxgap() {
    return atomic_exchange(&g_w.maxgap, 0) / 1e6;
}

double gpu_worker_behind() {
    double r = g_w.behind_n ? (double) g_w.behind_sum / g_w.behind_n : 0;
    g_w.behind_sum = g_w.behind_n = 0;
    return r;
}

int gpu_worker_lost() {
    return atomic_exchange(&g_w.lost, 0);
}

int gpu_worker_transfers() {
    return atomic_exchange(&g_w.transfers, 0);
}

int gpu_worker_drawn() {
    return atomic_exchange(&g_w.drawn, 0);
}

void gpu_worker_stats(double* busy, double* present, double* wait, int* copies,
                      int* wraps) {
    *busy = atomic_exchange(&g_w.busy_ns, 0) / 1e6;
    *present = atomic_exchange(&g_w.present_ns, 0) / 1e6;
    *wait = g_w.wait_ns / 1e6;
    g_w.wait_ns = 0;
    *copies = atomic_exchange(&g_w.copies, 0);
    *wraps = atomic_exchange(&g_w.wraps, 0);
}

// waits until the thread has run everything pushed so far
void gpu_worker_sync(bool finish) {
    if (!g_w.on || g_is_worker) {
        if (finish) glFinish();
        return;
    }
    if (finish) atomic_fetch_add(&g_w.wraps, 1);
    gw_push(GW_SYNC, finish, 0, 0, 0, 0, 0, 0, 0);
    u32 head = atomic_load_explicit(&g_w.msg_head, memory_order_relaxed);
    u64 t0 = gw_now();
    for (;;) {
        u32 tail = atomic_load_explicit(&g_w.msg_tail, memory_order_acquire);
        if (tail == head) break;
        gw_wait_main(&g_w.msg_tail, tail);
    }
    g_w.wait_ns += gw_now() - t0;
}

void gpu_worker_start_frame(GPU* gpu) {
    if (g_w.on) {
        g_w.behind_sum += g_w.presents -
                          atomic_load_explicit(&g_w.presents_done, memory_order_acquire);
        g_w.behind_n++;
        // the renderer is still on the previous frame: run this one for its
        // game logic only
        bool skip = g_w.can_skip &&
                    g_w.presents - atomic_load_explicit(&g_w.presents_done,
                                                        memory_order_acquire) >=
                        g_w.skip_after;
        gw_push(GW_START_FRAME, skip, 0, 0, 0, 0, 0, 0, 0);
    } else {
        gpu_gl_start_frame(gpu);
    }
}

// the game may run one whole frame ahead of what is on screen, no more
void gpu_worker_present(int w, int h) {
    gw_push(GW_PRESENT, w, h, 0, 0, 0, 0, 0, 0);
    g_w.presents++;
    g_vstream_presents++;
    u64 t0 = gw_now();
    int n = 0;
    // with skipping the rule is only there to stop runaway queues
    u32 ahead = g_w.can_skip ? 6 : 1;
    (void) n;
    for (;;) {
        u32 done = atomic_load_explicit(&g_w.presents_done, memory_order_acquire);
        if (done + ahead >= g_w.presents) break;
        gw_wait_main(&g_w.presents_done, done);
    }
    g_w.wait_ns += gw_now() - t0;
}

static void gw_exec_cmds(GPU* gpu, u32* cur, u32* end);
static int g_layout_log = -1; // SPIKE_LAYOUTLOG=1, see gw_layout_log
static void gw_invalidate(GPU* gpu, u32 paddr, u32 len);
static void gw_rehash(GPU* gpu);

static void* gw_main(void* arg) {
    GPU* gpu = arg;
    g_is_worker = true;
    if (getenv("SPIKE_PIN")) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(2, &set);
        pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    }
    g_w.bind(true);
    if (g_spinwait < 0) g_spinwait = getenv("SPIKE_SPINWAIT") != nullptr;
    int idle = 0;
    bool quit = false;
    while (!quit) {
        u32 tail = atomic_load_explicit(&g_w.msg_tail, memory_order_relaxed);
        if (tail == atomic_load_explicit(&g_w.msg_head, memory_order_acquire)) {
            if (g_spinwait || ++idle < 2000) {
                if (g_spinwait) gw_relax(&idle);
                else gw_cpu_relax();
                continue;
            }
            atomic_store_explicit(&g_worker_sleeping, 1, memory_order_seq_cst);
            if (atomic_load_explicit(&g_w.msg_head, memory_order_seq_cst) == tail) {
                gw_futex_wait(&g_w.msg_head, tail);
            }
            atomic_store_explicit(&g_worker_sleeping, 0, memory_order_seq_cst);
            continue;
        }
        idle = 0;
        u64 t0 = gw_now();
        GWMsg m = g_w.msgs[tail % GW_MSGS];
        u32* a = m.a;
        switch (m.type) {
            case GW_CMDS: {
                u32* cmds = g_w.cmd + a[0] / 4;
                gw_exec_cmds(gpu, cmds, cmds + a[1]);
                atomic_store_explicit(&g_w.cmd_tail,
                                      a[2] | (u64) a[3] << 32,
                                      memory_order_release);
                break;
            }
            case GW_INVAL:
                gw_invalidate(gpu, a[0], a[1]);
                break;
            case GW_REHASH:
                gw_rehash(gpu);
                break;
            case GW_TRANSFER:
                atomic_fetch_add(&g_w.transfers, a[5] == 0);
                if (!g_w.skip) {
                    gpu_gl_display_transfer(gpu, a[0], a[1], a[2], a[3], a[4], a[5]);
                    g_w.new_image = true;
                } else if (a[5] == 0) {
                    atomic_fetch_add(&g_w.lost, 1);
                }
                break;
            case GW_LCD:
                gpu_gl_render_lcd_fb(gpu, a[0], a[1], a[2]);
                g_w.new_image = true;
                break;
            case GW_TEXCOPY:
                gpu_gl_texture_copy(gpu, a[0], a[1], a[2], a[3], a[4], a[5],
                                    a[6]);
                break;
            case GW_CLEAR:
                if (!g_w.skip) gpu_gl_clear_fb(gpu, a[0], a[1], a[2], a[3]);
                break;
            case GW_START_FRAME:
                g_w.skip = a[0];
                if (!g_w.skip) gpu_gl_start_frame(gpu);
                break;
            case GW_PRESENT: {
                // A game at 30 fps leaves every other refresh with nothing new: presenting it again
                // would only take time from the frame after it.
                extern bool rgds_bridge_wants_present();
                if (!g_w.skip && (g_w.new_image || !g_w.only_new || rgds_bridge_wants_present())) {
                    g_w.new_image = false;
                    g_w.present(a[0], a[1]);
                    gpu_gl_frame_fence();
                    atomic_fetch_add(&g_w.drawn, 1);
                    u64 now = gw_now();
                    if (g_w.last_present && now - g_w.last_present > atomic_load(&g_w.maxgap)) {
                        atomic_store(&g_w.maxgap, now - g_w.last_present);
                    }
                    if (g_w.last_present) spike_gap_hook((now - g_w.last_present) / 1e6);
                    g_w.last_present = now;
                }
                atomic_fetch_add_explicit(&g_w.presents_done, 1,
                                          memory_order_seq_cst);
                gw_wake_main(&g_w.presents_done);
                atomic_fetch_add(&g_w.present_ns, gw_now() - t0);
                break;
            }
            case GW_SYNC:
                if (a[0]) glFinish();
                break;
            case GW_QUIT:
                quit = true;
                break;
        }
        atomic_store_explicit(&g_w.msg_tail, tail + 1, memory_order_seq_cst);
        gw_wake_main(&g_w.msg_tail);
        atomic_fetch_add(&g_w.busy_ns, gw_now() - t0);
    }
    g_w.bind(false);
    return nullptr;
}

void gpu_worker_start(GPU* gpu, void (*bind)(bool), void (*present)(int, int)) {
    if (g_w.on) return;
    if (g_layout_log < 0) g_layout_log = getenv("SPIKE_LAYOUTLOG") != nullptr;
    if (!g_w.msgs) {
        g_w.msgs = malloc(GW_MSGS * sizeof(GWMsg));
        g_w.cmd = malloc(GW_CMD_RING);
        g_w.snaps = malloc(GW_SNAPS * sizeof(DrawSnap));
    }
    atomic_store(&g_w.msg_head, 0);
    atomic_store(&g_w.msg_tail, 0);
    g_w.cmd_head = 0;
    atomic_store(&g_w.cmd_tail, 0);
    g_w.snap_head = g_w.snap_read = 0;
    atomic_store(&g_w.snap_tail, 0);
    g_w.pending_draws = 0;
    g_w.presents = 0;
    atomic_store(&g_w.presents_done, 0);
    memcpy(&g_w.regs, &gpu->regs, sizeof g_w.regs);
    g_w.bind = bind;
    g_w.present = present;
    bind(false);
    g_w.can_skip = getenv("SPIKE_SKIP") != nullptr;
    g_w.skip_after = g_w.can_skip ? atoi(getenv("SPIKE_SKIP")) : 1;
    g_w.skip = false;
    g_w.new_image = true;
    g_w.only_new = !getenv("SPIKE_PRESENT_ALL");
    g_w.on = true;
    pthread_create(&g_w.thread, nullptr, gw_main, gpu);
}

void gpu_worker_stop() {
    if (!g_w.on) return;
    gw_push(GW_QUIT, 0, 0, 0, 0, 0, 0, 0, 0);
    pthread_join(g_w.thread, nullptr);
    g_w.on = false;
    g_w.bind(true);
}

void gpu_init(GPU* gpu) {
    LRU_init(gpu->fbs);
    // ensure this is pointing to something
    gpu->curfb = &gpu->fbs.root;
    LRU_init(gpu->textures);
    LRU_init(gpu->vshaders_sw);
    LRU_init(gpu->vshaders_hw);
    LRU_init(gpu->fshaders);

    gpu_vshrunner_init(gpu);

    renderer_gl_init(&gpu->gl, gpu);
}

void gpu_destroy(GPU* gpu) {
    gpu_worker_stop();
    renderer_gl_destroy(&gpu->gl, gpu);

    shaderjit_free_all(gpu);

    gpu_vshrunner_destroy(gpu);
}

void gpu_write_internalreg(GPU* gpu, u16 id, u32 param, u32 mask) {
    if (id >= GPUREG_MAX) {
        lerror("out of bounds gpu reg");
        return;
    }
    linfo("command %03x (0x%08x) & %08x (%f)", id, param, mask, I2F(param));
    gpu->regs.w[id] &= ~mask;
    gpu->regs.w[id] |= param & mask;
    switch (id) {
        case GPUREG(geom.cmdbuf.jmp[0]):
            // with the thread on, the jumps were followed when copying the list
            if (!g_w.on)
                gpu_run_command_list(gpu, gpu->regs.geom.cmdbuf.addr[0] << 3,
                                     gpu->regs.geom.cmdbuf.size[0] << 3);
            return;
        case GPUREG(geom.cmdbuf.jmp[1]):
            if (!g_w.on)
                gpu_run_command_list(gpu, gpu->regs.geom.cmdbuf.addr[1] << 3,
                                     gpu->regs.geom.cmdbuf.size[1] << 3);
            return;
        case GPUREG(geom.drawarrays):
            gpu_draw(gpu, false, false);
            break;
        case GPUREG(geom.drawelements):
            gpu_draw(gpu, true, false);
            break;
        case GPUREG(geom.fixattr_data[0])... GPUREG(geom.fixattr_data[2]): {
            fvec4* fattr;
            bool immediatemode = false;
            if (gpu->regs.geom.fixattr_idx == 0xf) {
                Vec_grow(gpu->immattrs);
                fattr = &gpu->immattrs.d[gpu->immattrs.size];
                immediatemode = true;
            } else {
                fattr = &gpu->fixattrs[gpu->regs.geom.fixattr_idx];
            }
            switch (gpu->curfixi) {
                case 0: {
                    (*fattr)[3] = cvtf24(param >> 8);
                    gpu->curfixattr = (param & 0xff) << 16;
                    gpu->curfixi = 1;
                    break;
                }
                case 1: {
                    (*fattr)[2] = cvtf24(param >> 16 | gpu->curfixattr);
                    gpu->curfixattr = (param & MASK(16)) << 8;
                    gpu->curfixi = 2;
                    break;
                }
                case 2: {
                    (*fattr)[1] = cvtf24(param >> 24 | gpu->curfixattr);
                    (*fattr)[0] = cvtf24(param & MASK(24));
                    gpu->curfixi = 0;
                    if (immediatemode) gpu->immattrs.size++;
                    break;
                }
            }
            break;
        }
        case GPUREG(geom.start_draw_func0):
            // this register must be written to after any draw call so we can
            // use it to end an immediate draw call, since there is no explicit
            // way to end an immediate mode draw call like glEnd
            if (gpu->immattrs.size) {
                gpu_draw(gpu, false, true);
            }
            break;
        case GPUREG(lighting.lutData[0])... GPUREG(lighting.lutData[7]): {
            u16 val = param & MASK(12);
            gpu->lightLuts[gpu->regs.lighting.lutNum]
                          [gpu->regs.lighting.lutIndex++] = val << 4 | val >> 8;
            gpu->lightLutDirty = true;
            break;
        }
        case GPUREG(tex.fogLutData[0])... GPUREG(tex.fogLutData[7]): {
            u16 val = param >> 13 & MASK(11);
            gpu->fogLut[gpu->regs.tex.fogLutIdx++] = val << 5 | val >> 6;
            // fog lut input comes with value, difference to next value
            // for the most part this difference can be ignored since it
            // is just there to make linear interpolation easier
            // however for the last entry of the table we cannot ignore it
            // since effectively it gives a 128th entry of the table
            // depth = 127/128 -> entry 127
            // depth = 128/128 = 1 -> entry 128
            // depth = 127..128 -> interpolate between the above
            // in the fragment shader when sampling we will also
            // modify texture coordinate to allow proper sampling in this way
            if (gpu->regs.tex.fogLutIdx == 0) {
                val += (sbit(13)) param;
                if ((s16) val < 0) val = 0;
                if ((s16) val > MASK(11)) val = MASK(11);
                gpu->fogLut[128] = val << 5 | val >> 6;
            }
            gpu->fogLutDirty = true;
            break;
        }
        case GPUREG(tex.proctexLutData[0])... GPUREG(tex.proctexLutData[7]): {
            switch (gpu->regs.tex.proctexLutIndex.table) {
                case 0: {
                    u16 val = param & MASK(12);
                    val = val << 4 | val >> 8;
                    gpu->proctexNoiseLut[gpu->regs.tex.proctexLutIndex.index++ %
                                         128] = val;
                    gpu->proctexNoiseLutDirty = true;
                    break;
                }
                case 2: {
                    u16 val = param & MASK(12);
                    val = val << 4 | val >> 8;
                    gpu->proctexMapLut[gpu->regs.tex.proctexLutIndex.index++ %
                                       128][0] = val;
                    gpu->proctexMapLutDirty = true;
                    break;
                }
                case 3: {
                    u16 val = param & MASK(12);
                    val = val << 4 | val >> 8;
                    gpu->proctexMapLut[gpu->regs.tex.proctexLutIndex.index++ %
                                       128][1] = val;
                    gpu->proctexMapLutDirty = true;
                    break;
                }
                case 4:
                    gpu->proctexLut[gpu->regs.tex.proctexLutIndex.index++] =
                        param;
                    gpu->proctexLutDirty = true;
                    break;
            }
            break;
        }
        case GPUREG(gsh.floatuniform_data[0])... GPUREG(
            gsh.floatuniform_data[7]): {
            u32 idx = gpu->regs.gsh.floatuniform_idx;
            if (idx >= 96) {
                lwarnonce("writing to out of bound uniform %d", idx);
                break;
            }
            fvec4* uniform = &gpu->gsh.floatuniform[idx];
            if (gpu->regs.gsh.floatuniform_mode) {
                (*uniform)[3 - gpu->gsh.curunifi] = I2F(param);
                if (++gpu->gsh.curunifi == 4) {
                    gpu->gsh.curunifi = 0;
                    gpu->regs.gsh.floatuniform_idx++;
                }
            } else {
                switch (gpu->gsh.curunifi) {
                    case 0: {
                        (*uniform)[3] = cvtf24(param >> 8);
                        gpu->gsh.curuniform = (param & 0xff) << 16;
                        gpu->gsh.curunifi = 1;
                        break;
                    }
                    case 1: {
                        (*uniform)[2] =
                            cvtf24(param >> 16 | gpu->gsh.curuniform);
                        gpu->gsh.curuniform = (param & MASK(16)) << 8;
                        gpu->gsh.curunifi = 2;
                        break;
                    }
                    case 2: {
                        (*uniform)[1] =
                            cvtf24(param >> 24 | gpu->gsh.curuniform);
                        (*uniform)[0] = cvtf24(param & MASK(24));
                        gpu->gsh.curunifi = 0;
                        gpu->regs.gsh.floatuniform_idx++;
                        break;
                    }
                }
            }
            break;
        }
        case GPUREG(vsh.floatuniform_data[0])... GPUREG(
            vsh.floatuniform_data[7]): {
            gpu->vsh_uniform_dirty = true;
            u32 idx = gpu->regs.vsh.floatuniform_idx;
            if (idx >= 96) {
                lwarnonce("writing to out of bound uniform %d", idx);
                break;
            }
            fvec4* uniform = &gpu->vsh.floatuniform[idx];
            if (gpu->regs.vsh.floatuniform_mode) {
                (*uniform)[3 - gpu->vsh.curunifi] = I2F(param);
                if (++gpu->vsh.curunifi == 4) {
                    gpu->vsh.curunifi = 0;
                    gpu->regs.vsh.floatuniform_idx++;
                }
            } else {
                switch (gpu->vsh.curunifi) {
                    case 0: {
                        (*uniform)[3] = cvtf24(param >> 8);
                        gpu->vsh.curuniform = (param & 0xff) << 16;
                        gpu->vsh.curunifi = 1;
                        break;
                    }
                    case 1: {
                        (*uniform)[2] =
                            cvtf24(param >> 16 | gpu->vsh.curuniform);
                        gpu->vsh.curuniform = (param & MASK(16)) << 8;
                        gpu->vsh.curunifi = 2;
                        break;
                    }
                    case 2: {
                        (*uniform)[1] =
                            cvtf24(param >> 24 | gpu->vsh.curuniform);
                        (*uniform)[0] = cvtf24(param & MASK(24));
                        gpu->vsh.curunifi = 0;
                        gpu->regs.vsh.floatuniform_idx++;
                        break;
                    }
                }
            }
            break;
        }
        case GPUREG(vsh.intuniform[0])... GPUREG(vsh.intuniform[3]):
        case GPUREG(vsh.booluniform):
            gpu->vsh_uniform_dirty = true;
            break;
        case GPUREG(gsh.entrypoint):
            gpu->gsh.code_dirty = true;
            break;
        case GPUREG(gsh.codetrans_data[0])... GPUREG(gsh.codetrans_data[8]):
            gpu->gsh.code_dirty = true;
            gpu->gsh
                .progdata[gpu->regs.gsh.codetrans_idx++ % SHADER_CODE_SIZE] =
                param;
            break;
        case GPUREG(gsh.opdescs_data[0])... GPUREG(gsh.opdescs_data[8]):
            gpu->gsh.code_dirty = true;
            gpu->gsh.opdescs[gpu->regs.gsh.opdescs_idx++ % SHADER_OPDESC_SIZE] =
                param;
            break;
        case GPUREG(vsh.entrypoint):
        case GPUREG(raster.sh_outmap[0])... GPUREG(raster.sh_outmap[6]):
            // entrypoint and outmap both affect the decompiled vs
            gpu->vsh.code_dirty = true;
            break;
        case GPUREG(vsh.codetrans_data[0])... GPUREG(vsh.codetrans_data[8]):
            gpu->vsh.code_dirty = true;
            gpu->vsh
                .progdata[gpu->regs.vsh.codetrans_idx++ % SHADER_CODE_SIZE] =
                param;
            break;
        case GPUREG(vsh.opdescs_data[0])... GPUREG(vsh.opdescs_data[8]):
            gpu->vsh.code_dirty = true;
            gpu->vsh.opdescs[gpu->regs.vsh.opdescs_idx++ % SHADER_OPDESC_SIZE] =
                param;
            break;
    }
}

static void gw_rehash(GPU* gpu) {
    // this is called every time gsp starts a new command list, since
    // the cpu cant modify a texture within a command list, so no need to rehash
    // textures more often than that
    for (int i = 0; i < TEX_MAX; i++) {
        gpu->textures.d[i].needs_rehash = true;
    }
}

void gpu_reset_needs_rehesh(GPU* gpu) {
    if (g_w.on) {
        gw_push(GW_REHASH, 0, 0, 0, 0, 0, 0, 0, 0);
    } else {
        gw_rehash(gpu);
    }
}

static void gw_exec_cmds(GPU* gpu, u32* cur, u32* end) {
    while (cur < end) {
        GPUCommand c = {cur[1]};
        u32 mask = 0;
        if (c.mask & BIT(0)) mask |= 0xff << 0;
        if (c.mask & BIT(1)) mask |= 0xff << 8;
        if (c.mask & BIT(2)) mask |= 0xff << 16;
        if (c.mask & BIT(3)) mask |= 0xff << 24;

        gpu_write_internalreg(gpu, c.id, cur[0], mask);
        cur += 2;
        if (c.incmode) c.id++;
        for (int i = 0; i < c.nparams; i++) {
            gpu_write_internalreg(gpu, c.id, *cur++, mask);
            if (c.incmode) c.id++;
        }
        // each command must be 8 byte aligned
        if (c.nparams & 1) cur++;
    }
}

void gpu_vbo_layout(const GPURegs* regs, int vbo, VboLayout* out) {
    static const int typesize[4] = {1, 1, 2, 4};
    u32 instride = regs->geom.attrbuf[vbo].size;
    u32 inoff = 0, off = 0, maxcs = 1;
    out->n = 0;
    out->changed = false;
    for (int c = 0; c < regs->geom.attrbuf[vbo].count; c++) {
        int attr = (regs->geom.attrbuf[vbo].comp >> 4 * c) & 0xf;
        if (attr >= 0xc) {
            inoff += 4 * (attr - 0xb);
            continue;
        }
        int fmt = (regs->geom.attr_format >> 4 * attr) & 0xf;
        int size = (fmt >> 2) + 1, type = fmt & 3, cs = typesize[type];
        int outsize = type == 2 && size == 3 ? 4 : size;
        off = (off + cs - 1) & ~(u32) (cs - 1);
        auto a = &out->a[out->n++];
        a->attr = attr;
        a->size = outsize;
        a->insize = size;
        a->type = type;
        a->inoff = inoff;
        a->off = off;
        if (off != inoff || outsize != size) out->changed = true;
        if ((u32) cs > maxcs) maxcs = cs;
        inoff += size * cs;
        off += outsize * cs;
    }
    if (!out->changed && instride % maxcs == 0) {
        out->stride = instride;
    } else {
        out->changed = true;
        out->stride = (off + maxcs - 1) & ~(maxcs - 1);
    }
}

// SPIKE_LAYOUTLOG=1: each distinct vertex buffer layout once in the run log, with the index size
static void gw_layout_log(const GPURegs* regs, int vbo, const VboLayout* lay, int index) {
    static u64 seen[256];
    static int nseen;
    u64 key = XXH3_64bits(lay, sizeof *lay) ^ regs->geom.attrbuf[vbo].size ^ (u64) index << 60;
    for (int i = 0; i < nseen; i++) if (seen[i] == key) return;
    if (nseen == countof(seen)) return;
    seen[nseen++] = key;
    static const char* tn[4] = {"s8", "u8", "s16", "f32"};
    char line[512];
    int n = snprintf(line, sizeof line, "layout: idx%s stride %u -> %u%s:", index == 0 ? "-" : index == 1 ? "8" : "16",
                     (u32) regs->geom.attrbuf[vbo].size, lay->stride, lay->changed ? " relaid" : "");
    for (int i = 0; i < lay->n && n < (int) sizeof line - 40; i++)
        n += snprintf(line + n, sizeof line - n, " a%d %sx%d@%u", lay->a[i].attr, tn[lay->a[i].type], lay->a[i].insize,
                      lay->a[i].inoff);
    spike_logf("%s\n", line);
}

// copies one draw's guest data out, the same ranges the draw will use
static void gw_snap_draw(GPU* gpu, GPURegs* regs, bool elements, DrawSnap* snap) {
    int basevert = regs->geom.vtx_off;
    int nbufverts = regs->geom.nverts;
    snap->minind = snap->maxind = 0;
    snap->index = 0;
    snap->vhash = snap->ihash = 0;
    if (elements) {
        u32 minind = 0xffff, maxind = 0;
        void* indexbuf = PTR(regs->geom.attr_base * 8 + regs->geom.indexbufoff);
        bool indexsize = regs->geom.indexfmt;
        int nverts = regs->geom.nverts;
        if (indexsize) {
            u16* ind = indexbuf;
            for (int i = 0; i < nverts; i++) {
                if (ind[i] < minind) minind = ind[i];
                if (ind[i] > maxind) maxind = ind[i];
            }
        } else {
            u8* ind = indexbuf;
            for (int i = 0; i < nverts; i++) {
                if (ind[i] < minind) minind = ind[i];
                if (ind[i] > maxind) maxind = ind[i];
            }
        }
        if (!nverts) minind = maxind = 0;
        snap->index = gpu_gl_vstream_put(indexbuf, nverts * BIT(indexsize), 4);
        if (g_draw_dump) snap->ihash = XXH3_64bits(indexbuf, nverts * BIT(indexsize));
        snap->minind = minind;
        snap->maxind = maxind;
        basevert = minind;
        nbufverts = maxind + 1 - minind;
    }
    for (int vbo = 0; vbo < 12; vbo++) {
        if (regs->geom.attrbuf[vbo].count == 0) continue;
        void* data =
            PTR(regs->geom.attr_base * 8 + regs->geom.attrbuf[vbo].offset);
        u32 stride = regs->geom.attrbuf[vbo].size;
        VboLayout lay;
        gpu_vbo_layout(regs, vbo, &lay);
        if (g_layout_log) gw_layout_log(regs, vbo, &lay, elements ? 1 + regs->geom.indexfmt : 0);
        if (lay.changed && nbufverts > 0) {
            // laid out for the driver, attribute by attribute (see VboLayout)
            static const int typesize[4] = {1, 1, 2, 4};
            u32 bytes = nbufverts * lay.stride;
            u32 at = gpu_gl_vstream_put(nullptr, bytes, 16);
            u8* dst = gpu_gl_vstream_ptr() + at;
            const u8* src = data + basevert * stride;
            for (int v = 0; v < nbufverts; v++, src += stride, dst += lay.stride) {
                for (int i = 0; i < lay.n; i++) {
                    int cs = typesize[lay.a[i].type];
                    memcpy(dst + lay.a[i].off, src + lay.a[i].inoff, lay.a[i].insize * cs);
                    if (lay.a[i].size != lay.a[i].insize) {
                        // short3 -> short4: w = 1, as a scaled integer it reads as 1.0
                        *(s16*) (dst + lay.a[i].off + 6) = 1;
                    }
                }
            }
            snap->vbo[vbo] = at;
            g_draws_relaid++;
        } else {
            snap->vbo[vbo] = gpu_gl_vstream_put(data + basevert * stride,
                                                nbufverts * stride, 16);
        }
        if (g_draw_dump) snap->vhash ^= XXH3_64bits(data + basevert * stride, nbufverts * stride);
    }
}

static void gw_flush(u32* from, u32* to) {
    size_t bytes = (to - from) * 4;
    if (bytes > GW_CMD_RING) {
        lerror("command list too long");
        return;
    }
    atomic_thread_fence(memory_order_release);
    if (bytes) {
        u64 head = g_w.cmd_head;
        size_t pos = head % GW_CMD_RING;
        if (pos + bytes > GW_CMD_RING) {
            head += GW_CMD_RING - pos;
            pos = 0;
        }
        if (head + bytes - atomic_load_explicit(&g_w.cmd_tail,
                                                memory_order_acquire) >
            GW_CMD_RING) {
            u64 t0 = gw_now();
            int n = 0;
            while (head + bytes - atomic_load_explicit(&g_w.cmd_tail,
                                                       memory_order_acquire) >
                   GW_CMD_RING)
                gw_relax(&n);
            g_w.wait_ns += gw_now() - t0;
        }
        memcpy((u8*) g_w.cmd + pos, from, bytes);
        head += bytes;
        g_w.cmd_head = head;
        gw_push(GW_CMDS, pos, bytes / 4, head, head >> 32, 0, 0, 0, 0);
    }
    g_w.pending_draws = 0;
    if (g_w.snap_head - atomic_load_explicit(&g_w.snap_tail,
                                             memory_order_acquire) >
        GW_SNAPS - 2 * GW_FLUSH_DRAWS) {
        u64 t0 = gw_now();
        int n = 0;
        while (g_w.snap_head - atomic_load_explicit(&g_w.snap_tail,
                                                    memory_order_acquire) >
               GW_SNAPS - 2 * GW_FLUSH_DRAWS)
            gw_relax(&n);
        g_w.wait_ns += gw_now() - t0;
    }
}

static inline void gw_front_write(GPU* gpu, u32 id, u32 param, u32 mask,
                                  int* jump) {
    if (id >= GPUREG_MAX) return;
    g_w.regs.w[id] = (g_w.regs.w[id] & ~mask) | (param & mask);
    switch (id) {
        case GPUREG(geom.cmdbuf.jmp[0]):
            *jump = 0;
            break;
        case GPUREG(geom.cmdbuf.jmp[1]):
            *jump = 1;
            break;
        case GPUREG(geom.drawarrays):
        case GPUREG(geom.drawelements):
            gw_snap_draw(gpu, &g_w.regs, id == GPUREG(geom.drawelements),
                         &g_w.snaps[g_w.snap_head++ % GW_SNAPS]);
            g_w.pending_draws++;
            break;
    }
}

static void gw_front_walk(GPU* gpu, u32 paddr, u32 size, int depth) {
    paddr &= ~15;
    size &= ~15;

    u32* cur = PTR(paddr);
    u32* end = cur + (size / 4);
    u32* run = cur;
    while (cur < end) {
        GPUCommand c = {cur[1]};
        u32 mask = 0;
        if (c.mask & BIT(0)) mask |= 0xff << 0;
        if (c.mask & BIT(1)) mask |= 0xff << 8;
        if (c.mask & BIT(2)) mask |= 0xff << 16;
        if (c.mask & BIT(3)) mask |= 0xff << 24;

        int jump = -1;
        u32 id = c.id;
        gw_front_write(gpu, id, cur[0], mask, &jump);
        cur += 2;
        if (c.incmode) id++;
        for (int i = 0; i < c.nparams; i++) {
            gw_front_write(gpu, id, *cur++, mask, &jump);
            if (c.incmode) id++;
        }
        if (c.nparams & 1) cur++;

        if (jump >= 0 || g_w.pending_draws >= GW_FLUSH_DRAWS) {
            gw_flush(run, cur);
            run = cur;
            if (jump >= 0 && depth < 20000) { // games chain lists through jumps, deeply
                gw_front_walk(gpu, g_w.regs.geom.cmdbuf.addr[jump] << 3,
                              g_w.regs.geom.cmdbuf.size[jump] << 3, depth + 1);
            }
        }
    }
    gw_flush(run, cur);
}

void gpu_run_command_list(GPU* gpu, u32 paddr, u32 size) {
    if (g_w.on) {
        gw_front_walk(gpu, paddr, size, 0);
        return;
    }

    paddr &= ~15;
    size &= ~15;

    u32* cmds = PTR(paddr);
    gw_exec_cmds(gpu, cmds, cmds + (size / 4));
}

// searches the framebuffer cache and return nullptr if not found
// need to search starting at most recently used FB because
// there can be overlapping fbs apparently
FBInfo* gpu_fbcache_find_within(GPU* gpu, u32 color_paddr) {
    FBInfo* newfb = nullptr;
    for (auto fb = gpu->fbs.root.next; fb != &gpu->fbs.root; fb = fb->next) {
        if (fb->color_paddr <= color_paddr &&
            color_paddr <
                fb->color_paddr + fb->width * fb->height * fb->color_Bpp) {
            newfb = fb;
            break;
        }
    }
    if (newfb) LRU_use(gpu->fbs, newfb);
    return newfb;
}

TexInfo* gpu_texcache_find_within(GPU* gpu, u32 paddr) {
    TexInfo* tex = nullptr;
    for (int i = 0; i < TEX_MAX; i++) {
        if (gpu->textures.d[i].paddr <= paddr &&
            paddr < gpu->textures.d[i].paddr + gpu->textures.d[i].size) {
            tex = &gpu->textures.d[i];
            break;
        }
    }
    if (tex) LRU_use(gpu->textures, tex);
    return tex;
}

// the first wall of defense for texture cache invalidation
void gpu_invalidate_range(GPU* gpu, u32 paddr, u32 len) {
    if (g_w.on && !g_is_worker) {
        gw_push(GW_INVAL, paddr, len, 0, 0, 0, 0, 0, 0);
        return;
    }
    gw_invalidate(gpu, paddr, len);
}

static void gw_invalidate(GPU* gpu, u32 paddr, u32 len) {
    linfo("invalidating cache at %08x-%08x", paddr, paddr + len);

    // probably should optimize this at some point to not be linear
    for (int i = 0; i < TEX_MAX; i++) {
        auto t = &gpu->textures.d[i];
        // The same bytes are often written again (3D Land copies its textures to VRAM each frame):
        // whether that matters is decided by a hash on the next use.
        if (t->paddr && t->paddr < paddr + len && paddr < t->paddr + t->size) {
            u32 lo = paddr > t->paddr ? paddr - t->paddr : 0;
            u32 hi = paddr + len < t->paddr + t->size ? paddr + len - t->paddr : t->size;
            if (!t->stale || lo < t->dirty_lo) t->dirty_lo = lo;
            if (!t->stale || hi > t->dirty_hi) t->dirty_hi = hi;
            t->stale = true;
        }
    }
}

void gpu_display_transfer(GPU* gpu, u32 paddr, int yoff, bool scalex,
                          bool scaley, bool vflip, int screenid) {
    if (g_w.on) {
        gw_push(GW_TRANSFER, paddr, yoff, scalex, scaley, vflip, screenid, 0, 0);
        return;
    }
    gpu_gl_display_transfer(gpu, paddr, yoff, scalex, scaley, vflip, screenid);
}

void gpu_render_lcd_fb(GPU* gpu, u32 paddr, u32 fmt, int screenid) {
    if (g_w.on) {
        gw_push(GW_LCD, paddr, fmt, screenid, 0, 0, 0, 0, 0);
        return;
    }
    gpu_gl_render_lcd_fb(gpu, paddr, fmt, screenid);
}

void gpu_texture_copy(GPU* gpu, u32 srcpaddr, u32 dstpaddr, u32 size,
                      u32 srcpitch, u32 srcgap, u32 dstpitch, u32 dstgap) {
    if (g_w.on) {
        // reads and writes guest memory the game may reuse right after
        atomic_fetch_add(&g_w.copies, 1);
        gw_push(GW_TEXCOPY, srcpaddr, dstpaddr, size, srcpitch, srcgap, dstpitch,
                dstgap, 0);
        // the copy still reads guest memory late; waiting here costs a third of a frame
        if (getenv("SPIKE_COPYSYNC")) gpu_worker_sync(false);
        return;
    }
    gpu_gl_texture_copy(gpu, srcpaddr, dstpaddr, size, srcpitch, srcgap,
                        dstpitch, dstgap);
}

void gpu_clear_fb(GPU* gpu, u32 paddr, u32 len, u32 value, u32 datasz) {
    if (g_w.on) {
        gw_push(GW_CLEAR, paddr, len, value, datasz, 0, 0, 0, 0);
        if (getenv("SPIKE_CLEARSYNC")) gpu_worker_sync(false);
        return;
    }
    gpu_gl_clear_fb(gpu, paddr, len, value, datasz);
}

u32 morton_swizzle(u32 w, u32 x, u32 y) {
    u32 swizzle[8] = {
        0x00, 0x01, 0x04, 0x05, 0x10, 0x11, 0x14, 0x15,
    };

    // textures are stored as 8x8 tiles, and within each each tile the x and y
    // coordinates are interleaved

    u32 tx = x >> 3;
    u32 fx = x & 7;
    u32 ty = y >> 3;
    u32 fy = y & 7;

    return (ty * (w >> 3) + tx) * 64 + (swizzle[fx] | swizzle[fy] << 1);
}

typedef struct {
    void* base;
    u32 stride;
    u32 fmt;
} AttrConfig[12];

void vtx_loader_setup(GPU* gpu, AttrConfig cfg) {
    for (int i = 0; i < 12; i++) {
        cfg[i].base = gpu->fixattrs[i];
        cfg[i].stride = 0;
        cfg[i].fmt = 0b1111;
    }
    for (int vbo = 0; vbo < 12; vbo++) {
        void* vtx = PTR(gpu->regs.geom.attr_base * 8 +
                        gpu->regs.geom.attrbuf[vbo].offset);
        u32 stride = gpu->regs.geom.attrbuf[vbo].size;
        for (int c = 0; c < gpu->regs.geom.attrbuf[vbo].count; c++) {
            int attr = (gpu->regs.geom.attrbuf[vbo].comp >> 4 * c) & 0xf;
            if (attr >= 0xc) {
                vtx += 4 * (attr - 0xb);
                continue;
            }
            int fmt = (gpu->regs.geom.attr_format >> 4 * attr) & 0xf;

            cfg[attr].base = vtx;
            cfg[attr].stride = stride;
            cfg[attr].fmt = fmt;

            int size = (fmt >> 2) + 1;
            int type = fmt & 3;
            static const int typesize[4] = {1, 1, 2, 4};
            vtx += size * typesize[type];
        }
    }
}

void vtx_loader_imm_setup(GPU* gpu, AttrConfig cfg) {
    for (int i = 0; i < 12; i++) {
        cfg[i].base = gpu->fixattrs[i];
        cfg[i].stride = 0;
        cfg[i].fmt = 0b1111;
    }
    u32 nattrs = gpu->regs.geom.vsh_num_attr + 1;
    for (int i = 0; i < nattrs; i++) {
        cfg[i].base = gpu->immattrs.d + i;
        cfg[i].stride = nattrs * sizeof(fvec4);
        cfg[i].fmt = 0b1111;
    }
}

#define LOADVEC1(t)                                                            \
    ({                                                                         \
        t* attr = vtx;                                                         \
        dst[pa][0] = attr[0];                                                  \
        dst[pa][1] = dst[pa][2] = 0;                                           \
        dst[pa][3] = 1;                                                        \
    })

#define LOADVEC2(t)                                                            \
    ({                                                                         \
        t* attr = vtx;                                                         \
        dst[pa][0] = attr[0];                                                  \
        dst[pa][1] = attr[1];                                                  \
        dst[pa][2] = 0;                                                        \
        dst[pa][3] = 1;                                                        \
    })

#define LOADVEC3(t)                                                            \
    ({                                                                         \
        t* attr = vtx;                                                         \
        dst[pa][0] = attr[0];                                                  \
        dst[pa][1] = attr[1];                                                  \
        dst[pa][2] = attr[2];                                                  \
        dst[pa][3] = 1;                                                        \
    })

#define LOADVEC4(t)                                                            \
    ({                                                                         \
        t* attr = vtx;                                                         \
        dst[pa][0] = attr[0];                                                  \
        dst[pa][1] = attr[1];                                                  \
        dst[pa][2] = attr[2];                                                  \
        dst[pa][3] = attr[3];                                                  \
    })

void load_vtx(GPU* gpu, AttrConfig cfg, int i, fvec4* dst) {
    u32 nattrs = gpu->regs.geom.vsh_num_attr + 1;
    for (int a = 0; a < nattrs; a++) {
        void* vtx = cfg[a].base + i * cfg[a].stride;
        int pa = (gpu->regs.vsh.permutation >> 4 * a) & 0xf;
        if (cfg[a].fmt == 0b1111) {
            memcpy(dst[pa], vtx, sizeof(fvec4));
        } else {
            switch (cfg[a].fmt) {
                case 0b0000:
                    LOADVEC1(s8);
                    break;
                case 0b0001:
                    LOADVEC1(u8);
                    break;
                case 0b0010:
                    LOADVEC1(s16);
                    break;
                case 0b0011:
                    LOADVEC1(float);
                    break;
                case 0b0100:
                    LOADVEC2(s8);
                    break;
                case 0b0101:
                    LOADVEC2(u8);
                    break;
                case 0b0110:
                    LOADVEC2(s16);
                    break;
                case 0b0111:
                    LOADVEC2(float);
                    break;
                case 0b1000:
                    LOADVEC3(s8);
                    break;
                case 0b1001:
                    LOADVEC3(u8);
                    break;
                case 0b1010:
                    LOADVEC3(s16);
                    break;
                case 0b1011:
                    LOADVEC3(float);
                    break;
                case 0b1100:
                    LOADVEC4(s8);
                    break;
                case 0b1101:
                    LOADVEC4(u8);
                    break;
                case 0b1110:
                    LOADVEC4(s16);
                    break;
            }
        }
    }
}

void gpu_write_outmap_vtx(GPU* gpu, Vertex* dst, fvec4* src) {
    // if a semantic is specified multiple times in the outmap we only pick the
    // first one
    u32 written = 0;
    for (int o = 0; o < 7; o++) {
        for (int j = 0; j < 4; j++) {
            u8 sem = gpu->regs.raster.sh_outmap[o][j];
            if (sem < 0x18 && !(written & BIT(sem)))
                dst->semantics[sem] = src[o][j];
            written |= BIT(sem);
        }
    }
}

void gpu_init_vsh(GPU* gpu, ShaderUnit* shu) {
    shu->code = (PICAInstr*) gpu->vsh.progdata;
    shu->opdescs = (OpDesc*) gpu->vsh.opdescs;
    shu->entrypoint = gpu->regs.vsh.entrypoint;
    shu->outmap_mask = gpu->regs.vsh.outmap_mask;
    shu->c = gpu->vsh.floatuniform;
    shu->i = gpu->regs.vsh.intuniform;
    shu->b = gpu->regs.vsh.booluniform;
}

void gpu_init_gsh(GPU* gpu, ShaderUnit* shu) {
    shu->code = (PICAInstr*) gpu->gsh.progdata;
    shu->opdescs = (OpDesc*) gpu->gsh.opdescs;
    shu->entrypoint = gpu->regs.gsh.entrypoint;
    shu->outmap_mask = gpu->regs.gsh.outmap_mask;
    shu->c = gpu->gsh.floatuniform;
    shu->i = gpu->regs.gsh.intuniform;
    shu->b = gpu->regs.gsh.booluniform;
    Vec_init(shu->gsh.outvtx);
}

void vsh_run_range(GPU* gpu, AttrConfig cfg, int srcoff, int dstoff, int count,
                   fvec4 (*vbuf)[16]) {
    ShaderUnit vsh;
    gpu_init_vsh(gpu, &vsh);
    for (int i = 0; i < count; i++) {
        load_vtx(gpu, cfg, srcoff + i, vsh.v);
        gpu->vsh_runner.shaderfunc(&vsh);
        shader_write_outmap(&vsh, vbuf[dstoff + i]);
    }
}

void vsh_thrd_func(GPU* gpu) {
    int id = gpu->vsh_runner.cur++;

    while (true) {
        while (!gpu->vsh_runner.ready[id]) sched_yield();
        gpu->vsh_runner.ready[id] = false;

        if (gpu->vsh_runner.die) return;

        vsh_run_range(gpu, gpu->vsh_runner.attrcfg,
                      gpu->vsh_runner.base + id * gpu->vsh_runner.count,
                      id * gpu->vsh_runner.count, gpu->vsh_runner.count,
                      gpu->vsh_runner.vbuf);

        gpu->vsh_runner.cur++;
    }
}

void gpu_vshrunner_init(GPU* gpu) {
    gpu->vsh_runner.cur = 0;
    for (int i = 0; i < ctremu.vshthreads; i++) {
        gpu->vsh_runner.ready[i] = false;
        pthread_create(&gpu->vsh_runner.threads[i], nullptr,
                       (void*) vsh_thrd_func, gpu);
    }
    while (gpu->vsh_runner.cur < ctremu.vshthreads);
}

void gpu_vshrunner_destroy(GPU* gpu) {
    gpu->vsh_runner.die = true;
    for (int i = 0; i < ctremu.vshthreads; i++) {
        gpu->vsh_runner.ready[i] = true;
    }
    for (int i = 0; i < ctremu.vshthreads; i++) {
        pthread_join(gpu->vsh_runner.threads[i], nullptr);
    }
}

void dispatch_vsh(GPU* gpu, void* attrcfg, int base, int count, void* vbuf) {
    if (ctremu.shaderjit) {
        if (gpu->vsh.code_dirty) {
            ShaderUnit shu;
            gpu_init_vsh(gpu, &shu);
            gpu->vsh_runner.shaderfunc = shaderjit_get(gpu, &shu);
            gpu->vsh.code_dirty = false;
        }
    } else {
        gpu->vsh_runner.shaderfunc = pica_shader_exec;
    }

    gpu->vsh_runner.attrcfg = attrcfg;
    gpu->vsh_runner.vbuf = vbuf;
    gpu->vsh_runner.base = base;
    gpu->vsh_runner.count = count / (ctremu.vshthreads + 1);
    if (gpu->vsh_runner.count) {
        gpu->vsh_runner.cur = 0;
        for (int i = 0; i < ctremu.vshthreads; i++) {
            gpu->vsh_runner.ready[i] = true;
        }
        vsh_run_range(
            gpu, attrcfg, base + gpu->vsh_runner.count * ctremu.vshthreads,
            gpu->vsh_runner.count * ctremu.vshthreads,
            gpu->vsh_runner.count + count % (ctremu.vshthreads + 1), vbuf);
        while (gpu->vsh_runner.cur < ctremu.vshthreads);
    } else {
        vsh_run_range(gpu, attrcfg, base, 0, count, vbuf);
    }
}

void gpu_run_vsh(GPU* gpu, bool immediate, int basevert, int nbufverts,
                 fvec4 (*vshout)[16]) {
    AttrConfig cfg;
    if (immediate) {
        vtx_loader_imm_setup(gpu, cfg);
    } else {
        vtx_loader_setup(gpu, cfg);
    }

    // run the vertex shader (possibly parallelized)
    dispatch_vsh(gpu, cfg, basevert, nbufverts, vshout);
}

void gpu_run_gsh(GPU* gpu, ShaderUnit* gsh, bool elements, int basevert,
                 int nverts, fvec4 (*vshout)[16], bool indexsize,
                 void* indexbuf) {
    // there are 3 geom shader modes, rn we only care about the
    // normal mode

    if (gpu->regs.geom.gsh_misc0.mode != 0) {
        lwarnonce("unknown geoshader mode %d", gpu->regs.geom.gsh_misc0.mode);
        return;
    }

    int vshoutct = gpu->regs.geom.vsh_outmap_total1 + 1;
    int gshinct = gpu->regs.gsh.inconfig.inattrs + 1;
    int stride = gshinct / vshoutct;

    for (int p = 0; p < nverts; p += stride) {
        for (int v = 0; v < stride; v++) {
            int idx = p + v;
            if (elements) {
                if (indexsize) {
                    idx = ((u16*) indexbuf)[idx] - basevert;
                } else {
                    idx = ((u8*) indexbuf)[idx] - basevert;
                }
            }
            for (int i = 0; i < vshoutct; i++) {
                int attr = v * vshoutct + i;
                attr = (gpu->regs.gsh.permutation >> 4 * attr) & 0xf;
                memcpy(gsh->v[attr], vshout[idx][i], sizeof(fvec4));
            }
        }

        pica_shader_exec(gsh);
    }
}

void gpu_draw(GPU* gpu, bool elements, bool immediate) {
    DrawSnap snap;
    if (g_w.on && g_w.skip) {
        if (!immediate) {
            g_w.snap_read++;
            atomic_store_explicit(&g_w.snap_tail, g_w.snap_read,
                                  memory_order_release);
        }
        Vec_free(gpu->immattrs);
        return;
    }
    if (immediate) {
        // the data came in the command list
    } else if (g_w.on) {
        snap = g_w.snaps[g_w.snap_read++ % GW_SNAPS];
        atomic_store_explicit(&g_w.snap_tail, g_w.snap_read,
                              memory_order_release);
    } else {
        gw_snap_draw(gpu, &gpu->regs, elements, &snap);
    }
    gpu_gl_draw(gpu, elements, immediate, &snap);
}
