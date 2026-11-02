#include "jit.h"

#include <time.h>

#include "backend/backend.h"
#include "optimizer.h"
#include "register_allocator.h"
#include "translator.h"

#include <sys/stat.h>
#include <unistd.h>

#include "emulator.h"
#include "kernel/memory.h"
#include "ras/ras.h"
#include "kernel/svc_types.h"
#include <xxh3.h>

// #define JIT_DISASM
// #define JIT_CPULOG

#ifdef JIT_DISASM
#define IR_DISASM
#define BACKEND_DISASM
#endif

JITConfig g_jit_config;
u64 g_jit_ns, g_jit_max;
int g_jit_blocks, g_jit_roots;

bool jit_addr_mapped(u32 addr) {
    PageEntry* l2 = ctremu.system.process.ptab[(addr >> 22) & MASK(10)];
    return l2 && l2[(addr >> 12) & MASK(10)].state != MEMST_FREE;
}

// Blocks entered from the dispatcher are remembered per game. A later run
// compiles them ahead of time in the spare time at the end of a frame, on the
// emulation thread, once the code that hashed the same is in memory.
typedef struct {
    u32 addr;
    u32 attrs_len; // attrs in the low 8 bits
    u64 hash;
} JitListEnt;

#define JL_SEEN_BITS 19
static struct {
    bool on, preloading, dirty, missed;
    JitListEnt* d;
    int n, cap, pos, idle, flush;
    u64* seen;
    FILE* fp;
} g_jl;
u64 g_jit_pre_ns;
int g_jit_pre_roots, g_jit_pre_blocks, g_jit_list_n;

static u64 jl_key(const JitListEnt* e) {
    u64 k = e->hash ^ (e->addr * 0x9e3779b97f4a7c15ull) ^ (e->attrs_len & 0xff);
    return k ? k : 1;
}

// false when it was already there
static bool jl_mark(const JitListEnt* e) {
    u64 k = jl_key(e);
    u32 mask = BIT(JL_SEEN_BITS) - 1;
    for (u32 i = k & mask;; i = (i + 1) & mask) {
        if (g_jl.seen[i] == k) return false;
        if (!g_jl.seen[i]) {
            g_jl.seen[i] = k;
            return true;
        }
    }
}

static void jl_push(const JitListEnt* e) {
    if (g_jl.n == g_jl.cap) {
        g_jl.cap = g_jl.cap ? g_jl.cap * 2 : 4096;
        g_jl.d = realloc(g_jl.d, g_jl.cap * sizeof *g_jl.d);
    }
    g_jl.d[g_jl.n++] = *e;
}

static u32 jl_len(JITBlock* block) {
    u32 end = block->end_addr;
    u32 pageend = (block->start_addr | 0xfff) + 1;
    if (end > pageend) end = pageend;
    return end > block->start_addr ? end - block->start_addr : 0;
}

static void jl_record(JITBlock* block) {
    if (!g_jl.on) return;
    if (g_jl.n >= BIT(JL_SEEN_BITS) / 2) return;
    u32 len = jl_len(block);
    JitListEnt e = {block->start_addr, block->attrs | len << 8,
                    XXH3_64bits(ctremu.system.virtmem + block->start_addr, len)};
    if (!jl_mark(&e)) return;
    jl_push(&e);
    if (g_jl.fp) {
        fwrite(&e, sizeof e, 1, g_jl.fp);
        g_jl.dirty = true;
    }
}

void jit_list_start(const char* rom) {
    g_jl.seen = calloc(BIT(JL_SEEN_BITS), sizeof(u64));
    g_jl.on = true;
    mkdir("jit", 0777);
    char* path;
    asprintf(&path, "jit/%s.list", rom);
    FILE* fp = fopen(path, "rb");
    long good = 0;
    if (fp) {
        JitListEnt e;
        while (fread(&e, sizeof e, 1, fp) == 1) {
            if (g_jl.n < BIT(JL_SEEN_BITS) / 2 && jl_mark(&e)) jl_push(&e);
            good += sizeof e;
        }
        fclose(fp);
        // a power cut can leave half a record behind
        truncate(path, good);
    }
    g_jit_list_n = g_jl.n;
    g_jl.fp = fopen(path, "ab");
    fprintf(stderr, "spike jit list %s: %d blocks\n", path, g_jl.n);
    free(path);
}

void jit_list_step(ArmCore* cpu, u64 budget_ns) {
    if (!g_jl.on) return;
    if (g_jl.dirty && ++g_jl.flush >= 60) {
        g_jl.flush = 0;
        g_jl.dirty = false;
        fflush(g_jl.fp);
    }
    if (g_jl.pos >= g_jl.n) {
        // code that was not loaded yet gets another look now and then
        if (!g_jl.missed || ++g_jl.idle < 240) return;
        g_jl.idle = 0;
        g_jl.pos = 0;
        g_jl.missed = false;
    }
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    u64 t0 = ts.tv_sec * 1000000000ull + ts.tv_nsec, now = t0;
    int blocks0 = g_jit_blocks;
    g_jl.preloading = true;
    while (g_jl.pos < g_jl.n && now - t0 < budget_ns) {
        JitListEnt e = g_jl.d[g_jl.pos++];
        u32 attrs = e.attrs_len & 0xff;
        u32 hi = e.addr >> 16, lo = (e.addr & 0xffff) >> 1;
        if (cpu->jit_cache[attrs] && cpu->jit_cache[attrs][hi] &&
            cpu->jit_cache[attrs][hi][lo]) {
            continue;
        }
        if (!jit_addr_mapped(e.addr) ||
            XXH3_64bits(ctremu.system.virtmem + e.addr, e.attrs_len >> 8) != e.hash) {
            g_jl.missed = true;
            continue;
        }
        get_jitblock(cpu, attrs, e.addr);
        g_jit_pre_roots++;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        now = ts.tv_sec * 1000000000ull + ts.tv_nsec;
    }
    g_jl.preloading = false;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    g_jit_pre_ns += ts.tv_sec * 1000000000ull + ts.tv_nsec - t0;
    g_jit_pre_blocks += g_jit_blocks - blocks0;
}

JITBlock* create_jit_block(ArmCore* cpu, u32 addr) {
    JITBlock* block = malloc(sizeof *block);
    block->attrs = cpu->cpsr.w & 0x3f;
    block->start_addr = addr;

    block->cpu = cpu;

    Vec_init(block->linkingblocks);

    IRBlock ir;
    irblock_init(&ir);

    compile_block(cpu, &ir, addr);

    block->numinstr = ir.numinstr;

    if (g_jit_config.optimize) {
        optimize_loadstore(&ir);
        optimize_constprop(&ir);
        if (g_jit_config.optimize_literals) optimize_literals(&ir, cpu);
        optimize_chainjumps(&ir);
        optimize_loadstore(&ir);
        optimize_constprop(&ir);
        optimize_chainjumps(&ir);
        optimize_deadcode(&ir);
        if (g_jit_config.linking) optimize_blocklinking(&ir, cpu);
    }

    block->end_addr = ir.end_addr;

    RegAllocation regalloc = allocate_registers(&ir);

    block->backend = backend_generate_code(&ir, &regalloc, cpu);
    block->code = backend_get_code(block->backend);

    cpu->jit_cache[block->attrs][addr >> 16][(addr & 0xffff) >> 1] = block;
    backend_patch_links(block);

#ifdef IR_DISASM
    ir_disassemble(&ir);
    regalloc_print(&regalloc);
#endif
#ifdef BACKEND_DISASM
    backend_disassemble(block->backend);
#endif

    regalloc_free(&regalloc);
    if (g_jit_config.ir_interpret) {
        block->ir = malloc(sizeof(IRBlock));
        *block->ir = ir;
    } else {
        irblock_free(&ir);
    }

    return block;
}

void destroy_jit_block(JITBlock* block) {
    if (g_jit_config.ir_interpret) {
        irblock_free(block->ir);
        free(block->ir);
    }

    block->cpu->jit_cache[block->attrs][block->start_addr >> 16]
                         [(block->start_addr & 0xffff) >> 1] = nullptr;
    backend_free(block->backend);
    Vec_foreach(l, block->linkingblocks) {
        if (!(block->cpu->jit_cache[l->attrs] &&
              block->cpu->jit_cache[l->attrs][l->addr >> 16]))
            continue;
        JITBlock* linkingblock =
            block->cpu
                ->jit_cache[l->attrs][l->addr >> 16][(l->addr & 0xffff) >> 1];
        if (linkingblock) destroy_jit_block(linkingblock);
    }
    Vec_free(block->linkingblocks);
    free(block);
}

void jit_exec(JITBlock* block) {
#ifdef JIT_CPULOG
    cpu_print_state(block->cpu);
#endif
    if (g_jit_config.ir_interpret) {
        ir_interpret(block->ir, block->cpu);
    } else {
        block->code();
    }
}

// the jit cache is formatted as follows
// 64 root entries corresponding to low 6 bits of cpsr of the block
// then BIT(16) entries corresponding to bit[31:16] of start addr
// then BIT(15) entries corresponding to bit[15:1] of start addr
// jit blocks will never cross page boundaries
JITBlock* get_jitblock(ArmCore* cpu, u32 attrs, u32 addr) {
    u32 addrhi = addr >> 16;
    u32 addrlo = (addr & 0xffff) >> 1;

    if (!cpu->jit_cache[attrs]) {
        cpu->jit_cache[attrs] = calloc(BIT(16), sizeof(JITBlock**));
    }

    if (!cpu->jit_cache[attrs][addrhi]) {
        cpu->jit_cache[attrs][addrhi] = calloc(BIT(16) >> 1, sizeof(JITBlock*));
    }

    JITBlock* block = nullptr;
    if (!cpu->jit_cache[attrs][addrhi][addrlo]) {
        u32 old = cpu->cpsr.jitattrs;
        cpu->cpsr.jitattrs = attrs;
        static int depth;
        struct timespec ts0, ts1;
        if (!depth++) clock_gettime(CLOCK_MONOTONIC, &ts0);
        block = create_jit_block(cpu, addr);
        g_jit_blocks++;
        if (!--depth) {
            clock_gettime(CLOCK_MONOTONIC, &ts1);
            u64 ns = (ts1.tv_sec - ts0.tv_sec) * 1000000000ull + ts1.tv_nsec - ts0.tv_nsec;
            if (!g_jl.preloading) {
                g_jit_ns += ns;
                g_jit_roots++;
                if (ns > g_jit_max) g_jit_max = ns;
                jl_record(block);
            }
        }
        cpu->cpsr.jitattrs = old;
        cpu->jit_cache[attrs][addrhi][addrlo] = block;
    } else {
        block = cpu->jit_cache[attrs][addrhi][addrlo];
    }

    return block;
}

// start is page aligned
void jit_invalidate_range(ArmCore* cpu, u32 start_addr, u32 len) {
    // newly loaded code may be on the list
    g_jl.pos = 0;
    g_jl.idle = 0;
    u32 end_addr = start_addr + len;
    u32 startpg = start_addr >> 16;
    u32 endpg = end_addr >> 16;
    for (int i = 0; i < 64; i++) {
        if (!cpu->jit_cache[i]) continue;
        for (int j = startpg; j < endpg; j++) {
            if (!cpu->jit_cache[i][j]) continue;
            for (int k = 0; k < BIT(16) >> 1; k++) {
                if (cpu->jit_cache[i][j][k]) {
                    destroy_jit_block(cpu->jit_cache[i][j][k]);
                    cpu->jit_cache[i][j][k] = nullptr;
                }
            }
        }
        if (!cpu->jit_cache[i][endpg]) continue;
        for (int k = 0; k < (end_addr & MASK(16)) >> 1; k++) {
            if (cpu->jit_cache[i][endpg][k]) {
                destroy_jit_block(cpu->jit_cache[i][endpg][k]);
                cpu->jit_cache[i][endpg][k] = nullptr;
            }
        }
    }
}

// A save state replaced the code and all it can branch to: every block goes at
// once, without following links, and the packed area starts over.
void jit_state_loaded(ArmCore* cpu) {
    for (int i = 0; i < 64; i++) {
        if (!cpu->jit_cache[i]) continue;
        for (int j = 0; j < BIT(16); j++) {
            if (!cpu->jit_cache[i][j]) continue;
            for (int k = 0; k < BIT(16) >> 1; k++) {
                JITBlock* block = cpu->jit_cache[i][j][k];
                if (!block) continue;
                backend_free(block->backend);
                Vec_free(block->linkingblocks);
                free(block);
                cpu->jit_cache[i][j][k] = nullptr;
            }
        }
    }
    rasArenaReset();
    g_jl.pos = 0;
    g_jl.idle = 0;
}

void jit_free_all(ArmCore* cpu) {
    for (int i = 0; i < 64; i++) {
        if (!cpu->jit_cache[i]) continue;
        for (int j = 0; j < BIT(16); j++) {
            if (!cpu->jit_cache[i][j]) continue;
            for (int k = 0; k < BIT(16) >> 1; k++) {
                if (cpu->jit_cache[i][j][k]) {
                    destroy_jit_block(cpu->jit_cache[i][j][k]);
                    cpu->jit_cache[i][j][k] = nullptr;
                }
            }
            free(cpu->jit_cache[i][j]);
            cpu->jit_cache[i][j] = nullptr;
        }
        free(cpu->jit_cache[i]);
        cpu->jit_cache[i] = nullptr;
    }
}

#ifdef __aarch64__
// spike: the only way into guest code. Saves what C expects kept, then x29 = cpu, x28 = guest memory
// and x27 = where a block goes when it ends, for all the blocks that run until the next return here.
__attribute__((naked)) static void jit_enter(JITFunc code, ArmCore* cpu, void* fastmem) {
    asm("stp x19, x20, [sp, #-96]!\n"
        "stp x21, x22, [sp, #16]\n"
        "stp x23, x24, [sp, #32]\n"
        "stp x25, x26, [sp, #48]\n"
        "stp x27, x28, [sp, #64]\n"
        "stp x29, x30, [sp, #80]\n"
        "mov x29, x1\n"
        "mov x28, x2\n"
        "adr x27, 1f\n"
        "br x0\n"
        "1:\n"
        "ldp x21, x22, [sp, #16]\n"
        "ldp x23, x24, [sp, #32]\n"
        "ldp x25, x26, [sp, #48]\n"
        "ldp x27, x28, [sp, #64]\n"
        "ldp x29, x30, [sp, #80]\n"
        "ldp x19, x20, [sp], #96\n"
        "ret\n");
}

void arm_exec_jit(ArmCore* cpu) {
    JITBlock* block = get_jitblock(cpu, cpu->cpsr.jitattrs, cpu->pc);
    jit_enter(block->code, cpu, cpu->fastmem);
}
#else
void arm_exec_jit(ArmCore* cpu) {
    JITBlock* block = get_jitblock(cpu, cpu->cpsr.jitattrs, cpu->pc);
    jit_exec(block);
}
#endif
