// A whole-machine save state. Kernel objects keep their pointers (kernel/karena.c
// at a fixed address, the E3DS struct in a static of a binary that is not position
// independent), which also ties a state to the exact binary that wrote it; the
// game's own save data is not in here and does not depend on this.
#include "savestate.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <xxh3.h>
#include <zstd.h>

#include "arm/jit/jit.h"
#include "emulator.h"
#include "kernel/karena.h"
#include "kernel/memory.h"
#include "kernel/svc_types.h"

#define SS_VERSION 1

typedef struct {
    char magic[8];
    u32 version;
    u32 e3ds_size;
    u64 build;
    u64 rom;
} SSHeader;

typedef struct {
    FILE* f;
    ZSTD_CCtx* c;
    ZSTD_DCtx* d;
    u8* buf;
    size_t cap;
    ZSTD_inBuffer in;
    bool err;
} SS;

static void ss_put(SS* ss, const void* p, size_t n, ZSTD_EndDirective mode) {
    ZSTD_inBuffer in = {p, n, 0};
    size_t left;
    do {
        ZSTD_outBuffer out = {ss->buf, ss->cap, 0};
        left = ZSTD_compressStream2(ss->c, &out, &in, mode);
        if (ZSTD_isError(left)) {
            ss->err = true;
            return;
        }
        if (out.pos && fwrite(ss->buf, 1, out.pos, ss->f) != out.pos) {
            ss->err = true;
            return;
        }
    } while (mode == ZSTD_e_end ? left != 0 : in.pos < in.size);
}

void savestate_put(void* ss, const void* p, size_t n) {
    ss_put(ss, p, n, ZSTD_e_continue);
}

bool savestate_get(void* ctx, void* p, size_t n) {
    SS* ss = ctx;
    ZSTD_outBuffer out = {p, n, 0};
    while (out.pos < out.size) {
        if (ss->in.pos == ss->in.size) {
            ss->in.size = fread(ss->buf, 1, ss->cap, ss->f);
            ss->in.pos = 0;
            if (!ss->in.size) {
                ss->err = true;
                return false;
            }
        }
        size_t r = ZSTD_decompressStream(ss->d, &out, &ss->in);
        if (ZSTD_isError(r)) {
            ss->err = true;
            return false;
        }
    }
    return true;
}

static u64 ss_build() {
    static u64 build;
    if (build) return build;
    int fd = open("/proc/self/exe", O_RDONLY);
    struct stat st;
    if (fd >= 0 && !fstat(fd, &st)) {
        void* m = mmap(nullptr, st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
        if (m != MAP_FAILED) {
            build = XXH3_64bits(m, st.st_size);
            munmap(m, st.st_size);
        }
    }
    if (fd >= 0) close(fd);
    return build;
}

static void ss_header(SSHeader* h) {
    memset(h, 0, sizeof *h);
    memcpy(h->magic, "T3DSSTAT", 8);
    h->version = SS_VERSION;
    h->e3ds_size = sizeof(E3DS);
    h->build = ss_build();
    h->rom = XXH3_64bits(ctremu.romfilenoext, strlen(ctremu.romfilenoext));
}

void* savestate_capture(u64* size) {
    extern void* gpu_gl_fbs_capture(GPU*, u64*);
    return gpu_gl_fbs_capture(&ctremu.system.gpu, size);
}

bool savestate_write(const char* path, const void* blob, u64 blob_size) {
    E3DS* s = &ctremu.system;
    SS ss = {};
    ss.f = fopen(path, "wb");
    if (!ss.f) return false;
    ss.cap = 1 << 20;
    ss.buf = malloc(ss.cap);
    ss.c = ZSTD_createCCtx();
    ZSTD_CCtx_setParameter(ss.c, ZSTD_c_compressionLevel, 1);
    ZSTD_CCtx_setParameter(ss.c, ZSTD_c_checksumFlag, 1);
    ZSTD_CCtx_setParameter(ss.c, ZSTD_c_nbWorkers, 2);

    SSHeader h;
    ss_header(&h);
    bool ok = fwrite(&h, sizeof h, 1, ss.f) == 1;

    if (ok) {
        extern void fs_files_put(E3DS*, void*);
        savestate_put(&ss, s, sizeof *s);
        u64 used = ka_used();
        savestate_put(&ss, &used, sizeof used);
        savestate_put(&ss, ka_base(), used);
        savestate_put(&ss, &s->physmem[FCRAM_PBASE], FCRAM_SIZE);
        savestate_put(&ss, &s->physmem[VRAM_PBASE], VRAM_SIZE);
        savestate_put(&ss, &s->physmem[DSPRAM_PBASE], DSPRAM_SIZE);
        fs_files_put(s, &ss);
        savestate_put(&ss, &blob_size, sizeof blob_size);
        savestate_put(&ss, blob, blob_size);
        ss_put(&ss, nullptr, 0, ZSTD_e_end);
    }
    ok = ok && !ss.err && !fflush(ss.f) && !fsync(fileno(ss.f));
    if (fclose(ss.f)) ok = false;
    ZSTD_freeCCtx(ss.c);
    free(ss.buf);
    return ok;
}

static void ss_map_virtual(E3DS* s) {
    void* r = mmap(s->virtmem, BITL(32), PROT_NONE,
                   MAP_PRIVATE | MAP_ANON | MAP_NORESERVE | MAP_FIXED, -1, 0);
    if (r == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
    extern u32 physaddr2memoff(u32 paddr);
    u32 runv = 0, runp = 0, n = 0;
    for (u32 l1 = 0; l1 < BIT(10); l1++) {
        PageEntry* t = s->process.ptab[l1];
        for (u32 l2 = 0; l2 < BIT(10); l2++) {
            bool used = t && t[l2].state != MEMST_FREE;
            u32 v = l1 << 22 | l2 << 12;
            if (n && (!used || t[l2].paddr != runp + n * PAGE_SIZE ||
                      physaddr2memoff(t[l2].paddr) != physaddr2memoff(runp) + n * PAGE_SIZE)) {
                if (mmap(&s->virtmem[runv], n * PAGE_SIZE, PROT_READ | PROT_WRITE,
                         MAP_SHARED | MAP_FIXED, s->mem_fd, physaddr2memoff(runp)) == MAP_FAILED) {
                    perror("mmap");
                    exit(1);
                }
                n = 0;
            }
            if (used) {
                if (!n) {
                    runv = v;
                    runp = t[l2].paddr;
                }
                n++;
            }
        }
    }
    if (n && mmap(&s->virtmem[runv], n * PAGE_SIZE, PROT_READ | PROT_WRITE,
                  MAP_SHARED | MAP_FIXED, s->mem_fd, physaddr2memoff(runp)) == MAP_FAILED) {
        perror("mmap");
        exit(1);
    }
}

struct SSLoaded {
    GPU* gpu;
    u8* blob;
};

// Everything but the renderer's side, which savestate_read_finish takes on the thread with the
// GL context. Nothing else may touch the machine meanwhile.
SSLoaded* savestate_read(const char* path, bool* wrecked) {
    E3DS* s = &ctremu.system;
    *wrecked = false;
    SS ss = {};
    ss.f = fopen(path, "rb");
    if (!ss.f) return nullptr;
    SSHeader h, want;
    ss_header(&want);
    if (fread(&h, sizeof h, 1, ss.f) != 1 || memcmp(&h, &want, sizeof h)) {
        fprintf(stderr, "spike state %s is not for this game and build\n", path);
        fclose(ss.f);
        return nullptr;
    }
    ss.cap = 1 << 20;
    ss.buf = malloc(ss.cap);
    ss.d = ZSTD_createDCtx();
    ss.in.src = ss.buf;

    SSLoaded* out = nullptr;
    E3DS* t = aligned_alloc(4096, (sizeof *t + 4095) & ~(size_t) 4095);
    u64 used = 0;
    bool ok = savestate_get(&ss, t, sizeof *t) && savestate_get(&ss, &used, sizeof used);
    if (!ok) goto end; // nothing has changed yet

    // from here on the machine is being replaced
    *wrecked = true;
    {
        extern void fs_files_close(E3DS*);
        extern void fs_files_get(E3DS*, void*);
        extern void jit_state_loaded(ArmCore*);

        fs_files_close(s);
        ok = savestate_get(&ss, ka_base(), used);
        ok = ok && savestate_get(&ss, &s->physmem[FCRAM_PBASE], FCRAM_SIZE);
        ok = ok && savestate_get(&ss, &s->physmem[VRAM_PBASE], VRAM_SIZE);
        ok = ok && savestate_get(&ss, &s->physmem[DSPRAM_PBASE], DSPRAM_SIZE);
        if (!ok) goto end;

        jit_state_loaded(&s->cpu);

        // what belongs to this process stays as it is
        t->cpu.fastmem = s->cpu.fastmem;
        memcpy(t->cpu.jit_cache, s->cpu.jit_cache, sizeof t->cpu.jit_cache);
        t->mem = s->mem;
        t->mem_fd = s->mem_fd;
        t->physmem = s->physmem;
        t->virtmem = s->virtmem;
        t->romimage.fp = s->romimage.fp;
        t->dsp.mem = s->dsp.mem;
        t->dsp.aac_handle = s->dsp.aac_handle;
        memset(t->services.fs.files, 0, sizeof t->services.fs.files);
        memset(t->services.fs.dirs, 0, sizeof t->services.fs.dirs);

        // the renderer goes on presenting from its own struct: all around it now, the guest's
        // side of it on the GL thread at the end
        out = calloc(1, sizeof *out);
        out->gpu = malloc(sizeof(GPU));
        memcpy(out->gpu, &t->gpu, sizeof(GPU));
        size_t before = offsetof(E3DS, gpu), after = before + sizeof(GPU);
        memcpy(s, t, before);
        memcpy((char*) s + after, (char*) t + after, sizeof *s - after);

        ss_map_virtual(s);
        fs_files_get(s, &ss);
        u64 blob_size = 0;
        ok = savestate_get(&ss, &blob_size, sizeof blob_size) && blob_size < (1ull << 30);
        if (ok) {
            out->blob = malloc(blob_size);
            ok = savestate_get(&ss, out->blob, blob_size);
        }
        if (!ok) goto end;
        *wrecked = false;
    }

end:
    free(t);
    ZSTD_freeDCtx(ss.d);
    free(ss.buf);
    fclose(ss.f);
    if (!ok && out) {
        free(out->gpu);
        free(out->blob);
        free(out);
        out = nullptr;
    }
    return out;
}

void savestate_read_finish(SSLoaded* loaded) {
    extern void gpu_gl_state_loaded(GPU*, GPU*, const u8*);
    gpu_gl_state_loaded(&ctremu.system.gpu, loaded->gpu, loaded->blob);
    free(loaded->gpu);
    free(loaded->blob);
    free(loaded);
}
