#include "karena.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define KA_BASE 0x4000000000ull
#define KA_SIZE (1ull << 28)
#define KA_CLASSES 1024
#define KA_STEP 16

typedef struct {
    char* cur;
    void* free[KA_CLASSES];
} KaHeader;

typedef struct {
    size_t size;
    size_t pad;
} KaBlock;

static KaHeader* ka_header() {
    static bool mapped;
    if (!mapped) {
        void* p = mmap((void*) KA_BASE, KA_SIZE, PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANON | MAP_NORESERVE | MAP_FIXED_NOREPLACE, -1, 0);
        if (p != (void*) KA_BASE) {
            perror("kernel object arena");
            abort();
        }
        mapped = true;
        ((KaHeader*) p)->cur = (char*) p + ((sizeof(KaHeader) + 15) & ~15ull);
    }
    return (KaHeader*) KA_BASE;
}

void* ka_malloc(size_t size) {
    KaHeader* h = ka_header();
    size = (size + KA_STEP - 1) & ~(size_t) (KA_STEP - 1);
    size_t c = size / KA_STEP;
    if (c < KA_CLASSES && h->free[c]) {
        void* p = h->free[c];
        h->free[c] = *(void**) p;
        return p;
    }
    if ((uint64_t) (h->cur - (char*) KA_BASE) + size + sizeof(KaBlock) > KA_SIZE) {
        fprintf(stderr, "kernel object arena is full\n");
        abort();
    }
    KaBlock* b = (KaBlock*) h->cur;
    b->size = size;
    h->cur += size + sizeof(KaBlock);
    return b + 1;
}

void* ka_calloc(size_t n, size_t size) {
    void* p = ka_malloc(n * size);
    memset(p, 0, n * size);
    return p;
}

void ka_free(void* p) {
    if (!p) return;
    if ((uint64_t) p < KA_BASE || (uint64_t) p >= KA_BASE + KA_SIZE) {
        free(p);
        return;
    }
    KaHeader* h = ka_header();
    size_t c = (((KaBlock*) p) - 1)->size / KA_STEP;
    if (c >= KA_CLASSES) return;
    *(void**) p = h->free[c];
    h->free[c] = p;
}

void* ka_base() {
    return ka_header();
}

size_t ka_used() {
    return ka_header()->cur - (char*) KA_BASE;
}

void ka_reset() {
    KaHeader* h = ka_header();
    memset(h->free, 0, sizeof h->free);
    h->cur = (char*) h + ((sizeof(KaHeader) + 15) & ~15ull);
}
