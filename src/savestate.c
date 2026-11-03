// A whole-machine save state that survives engine updates. Everything is stored field by field in
// fixed-width chunks, never as a struct, and what the running engine keeps as a pointer is stored as
// a name: a kernel object as its number, a service or scheduler callback as a string.
//
// To keep old states loading:
// - add a field at the end of its object or chunk: a state without it leaves the default;
// - never reorder, resize or remove a stored field, and only append to the embedded[] list;
// - a new callback goes in ports[] or callbacks[] (build-tanuki.sh fails the build otherwise);
// - raise SS_VERSION only for a change an old reader cannot skip, and keep reading the old format.
//
// The game's own save data is not in here and does not depend on this.
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

#define SS_VERSION 2
#define SS_OLDEST 2
#define SS_MAX_CHUNKS 32

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

static bool ss_get(SS* ss, void* p, size_t n) {
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

// One chunk in memory. The same code writes and reads a chunk, so the two cannot drift. Reading past
// the end leaves the field as it is: that is how a state older than a new field loads.
typedef struct {
    bool writing;
    u8* p;
    size_t n, cap, pos;
    bool err;
    E3DS* s;
    KObject** heap; // writing: found so far; reading: all of them
    u32 nheap, capheap;
} IO;

static void io_bytes(IO* io, void* p, size_t n) {
    if (io->writing) {
        if (io->n + n > io->cap) {
            io->cap = (io->n + n) * 2;
            io->p = realloc(io->p, io->cap);
        }
        memcpy(io->p + io->n, p, n);
        io->n += n;
    } else if (io->pos + n <= io->n) {
        memcpy(p, io->p + io->pos, n);
        io->pos += n;
    } else {
        io->pos = io->n;
    }
}

#define SCALAR(T, x)                                                           \
    do {                                                                       \
        T v_ = (T) (x);                                                        \
        size_t was_ = io->pos;                                                 \
        io_bytes(io, &v_, sizeof v_);                                          \
        if (!io->writing && io->pos != was_) (x) = v_;                         \
    } while (0)
#define S8(x) SCALAR(u8, x)
#define S16(x) SCALAR(u16, x)
#define S32(x) SCALAR(u32, x)
#define S64(x) SCALAR(u64, x)
#define SF32(x)                                                                \
    do {                                                                       \
        float v_ = (x);                                                        \
        size_t was_ = io->pos;                                                 \
        io_bytes(io, &v_, sizeof v_);                                          \
        if (!io->writing && io->pos != was_) (x) = v_;                         \
    } while (0)

// a fixed array of plain numbers: its size goes with it, so either side may have grown
static void io_array(IO* io, void* a, u32 size) {
    u32 stored = size;
    S32(stored);
    u32 common = stored < size ? stored : size;
    io_bytes(io, a, common);
    if (!io->writing) io->pos = stored > size && io->pos + (stored - size) <= io->n
                                    ? io->pos + (stored - size)
                                    : io->pos;
}
#define SARR(a) io_array(io, (a), sizeof(a))

static void io_string(IO* io, char* str, u32 cap) {
    u32 len = io->writing ? strlen(str) : 0;
    S32(len);
    if (len >= cap) {
        io->err = true;
        return;
    }
    io_bytes(io, str, len);
    str[len] = 0;
}

// A FIFO in order from its head, whatever size the ring is on either side.
#define S_FIFO(f, ITEM)                                                        \
    do {                                                                       \
        u32 count_ = (f).size;                                                 \
        S32(count_);                                                           \
        u32 head_ = (u32) (f).head;                                            \
        if (!io->writing) FIFO_clear(f);                                       \
        for (u32 i_ = 0; i_ < count_; i_++) {                                  \
            typeof((f).d[0]) it_;                                              \
            memset(&it_, 0, sizeof it_);                                       \
            if (io->writing) it_ = (f).d[(head_ + i_) & (FIFO_MAX(f) - 1)];    \
            ITEM(io, &it_);                                                    \
            if (!io->writing) {                                                \
                if ((f).size < FIFO_MAX(f)) FIFO_push(f, it_);                 \
                else io->err = true;                                           \
            }                                                                  \
        }                                                                      \
    } while (0)

// Kernel objects. 0 is no object, the embedded ones have their place in this list, the rest follow.
#define MAX_EMBEDDED 64
static u32 embedded(E3DS* s, KObject** e) {
    auto v = &s->services;
    KObject* list[] = {
        &s->process.hdr,          &v->notif_sem.hdr,        &v->apt.lock.hdr,
        &v->apt.notif_event.hdr,  &v->apt.resume_event.hdr, &v->apt.shared_font.hdr,
        &v->apt.capture_block.hdr, &v->gsp.sharedmem.hdr,   &v->dsp.sem_event.hdr,
        &v->hid.sharedmem.hdr,    &v->hid.events[0].hdr,    &v->hid.events[1].hdr,
        &v->hid.events[2].hdr,    &v->hid.events[3].hdr,    &v->hid.events[4].hdr,
        &v->cecd.cecinfo.hdr,     &v->y2r.transferend.hdr,  &v->ir.connection_status.hdr,
        &v->mic.event.hdr,        &v->cam.recvEvent.hdr,    &v->cam.vsyncEvent.hdr,
        &v->cam.errEvent.hdr,     &v->csnd.shmem.hdr,
    };
    static_assert(countof(list) <= MAX_EMBEDDED);
    memcpy(e, list, sizeof list);
    return countof(list);
}

// Every live object outside the machine struct has a handle. The services keep some pointers without
// a reference, and a game can close those handles: what is left is freed memory, stored as no object.
static bool object_alive(IO* io, KObject* o, KObject** e, u32 ne) {
    for (u32 i = 0; i < ne; i++)
        if (e[i] == o) return true;
    for (u32 i = 0; i < HANDLE_MAX; i++)
        if (io->s->process.handles[i] == o) return true;
    return false;
}

static void s_obj_(IO* io, KObject** o) {
    KObject* e[MAX_EMBEDDED];
    u32 ne = embedded(io->s, e);
    u32 id = 0;
    if (io->writing && *o && object_alive(io, *o, e, ne)) {
        for (u32 i = 0; i < ne && !id; i++)
            if (e[i] == *o) id = i + 1;
        for (u32 i = 0; i < io->nheap && !id; i++)
            if (io->heap[i] == *o) id = MAX_EMBEDDED + i + 1;
        if (!id) {
            if (io->nheap == io->capheap) {
                io->capheap = io->capheap ? io->capheap * 2 : 256;
                io->heap = realloc(io->heap, io->capheap * sizeof *io->heap);
            }
            io->heap[io->nheap++] = *o;
            id = MAX_EMBEDDED + io->nheap;
        }
    }
    S32(id);
    if (io->writing) return;
    *o = nullptr;
    if (!id) return;
    if (id <= ne) *o = e[id - 1];
    else if (id > MAX_EMBEDDED && id - MAX_EMBEDDED <= io->nheap)
        *o = io->heap[id - MAX_EMBEDDED - 1];
    else io->err = true;
}
#define S_OBJ(p)                                                               \
    do {                                                                       \
        KObject* o_ = (KObject*) (p);                                          \
        s_obj_(io, &o_);                                                       \
        if (!io->writing) (p) = (typeof(p)) o_;                                \
    } while (0)

static void s_list(IO* io, KListNode** head) {
    u32 count = 0;
    KObject* e[MAX_EMBEDDED];
    u32 ne = embedded(io->s, e);
    if (io->writing)
        for (KListNode* n = *head; n; n = n->next) count += object_alive(io, n->key, e, ne);
    S32(count);
    if (io->writing) {
        for (KListNode* n = *head; n; n = n->next) {
            if (!object_alive(io, n->key, e, ne)) continue;
            S_OBJ(n->key);
            S32(n->val);
        }
        return;
    }
    *head = nullptr;
    KListNode** tail = head;
    for (u32 i = 0; i < count; i++) {
        KListNode* n = ka_calloc(1, sizeof *n);
        S_OBJ(n->key);
        S32(n->val);
        *tail = n;
        tail = &n->next;
    }
}

// What the running engine keeps as code addresses, by name.
DECL_PORT_ARG(stub, name);
DECL_PORT_ARG(fs_sysfile, file);
#define PORT(n) {#n, (PortRequestHandlerArg) port_handle_##n}
static const struct {
    const char* name;
    PortRequestHandlerArg f;
} ports[] = {
    PORT(am),      PORT(apt),         PORT(boss),       PORT(cam),     PORT(cecd),   PORT(cfg),
    PORT(csnd),    PORT(dsp),         PORT(errf),       PORT(frd),     PORT(fs),     PORT(fs_dir),
    PORT(fs_file), PORT(fs_selfncch), PORT(fs_sysfile), PORT(gsp_gpu), PORT(hid),    PORT(ir_user),
    PORT(ldr_ro),  PORT(mic),         PORT(nwm_uds),    PORT(ptm),     PORT(srv),    PORT(stub),
    PORT(y2r),
};

void apt_resume_app(E3DS* s);
void dsp_event(E3DS* s);
void y2r_event(E3DS* s);
#define CALLBACK(n, is_object) {#n, (SchedulerCallback) n, is_object}
static const struct {
    const char* name;
    SchedulerCallback f;
    bool object; // its argument is a kernel object, otherwise a number
} callbacks[] = {
    CALLBACK(gsp_handle_event, false),  CALLBACK(y2r_event, false),
    CALLBACK(dsp_event, false),         CALLBACK(apt_resume_app, false),
    CALLBACK(timer_signal, true),       CALLBACK(thread_wakeup_timeout, true),
};

void sem_event_handler(E3DS* s);
static void s_port(IO* io, PortRequestHandlerArg* f) {
    char name[64] = "";
    if (io->writing) {
        for (u32 i = 0; i < countof(ports); i++)
            if (ports[i].f == *f) strcpy(name, ports[i].name);
        if (!name[0]) {
            fprintf(stderr, "state: a service session with no name\n");
            io->err = true;
        }
    }
    io_string(io, name, sizeof name);
    if (io->writing) return;
    *f = nullptr;
    for (u32 i = 0; i < countof(ports); i++)
        if (!strcmp(ports[i].name, name)) *f = ports[i].f;
    if (!*f) {
        fprintf(stderr, "state: no service %s in this engine\n", name);
        io->err = true;
    }
}

static void s_object(IO* io, KObject* o) {
    S32(o->refcount);
    switch (o->type) {
        case KOT_PROCESS:
        case KOT_RESLIMIT:
            break;
        case KOT_THREAD: {
            auto t = (KThread*) o;
            SARR(t->ctx.r);
            S32(t->ctx.cpsr);
            SARR(t->ctx.d);
            S32(t->ctx.fpscr);
            S32(t->waiting_addr);
            s_list(io, &t->waiting_objs);
            S8(t->wait_any);
            s_list(io, &t->waiting_thrds);
            s_list(io, &t->owned_mutexes);
            S32(t->id);
            S32(t->priority);
            S32(t->state);
            S32(t->cpu);
            S32(t->tls);
            break;
        }
        case KOT_EVENT: {
            auto e = (KEvent*) o;
            S8(e->signal);
            S8(e->sticky);
            s_list(io, &e->waiting_thrds);
            u8 has = e->callback != nullptr;
            S8(has);
            if (!io->writing) e->callback = has ? sem_event_handler : nullptr;
            break;
        }
        case KOT_TIMER: {
            auto t = (KTimer*) o;
            S8(t->signal);
            S8(t->sticky);
            S8(t->repeat);
            S64(t->interval);
            s_list(io, &t->waiting_thrds);
            break;
        }
        case KOT_SEMAPHORE: {
            auto m = (KSemaphore*) o;
            s_list(io, &m->waiting_thrds);
            S32(m->count);
            S32(m->max);
            break;
        }
        case KOT_MUTEX: {
            auto m = (KMutex*) o;
            S_OBJ(m->locker_thrd);
            S32(m->recursive_lock_count);
            s_list(io, &m->waiting_thrds);
            break;
        }
        case KOT_ARBITER:
            s_list(io, &((KArbiter*) o)->waiting_thrds);
            break;
        case KOT_SHAREDMEM: {
            auto m = (KSharedMem*) o;
            S32(m->paddr);
            S32(m->mapaddr);
            S32(m->size);
            break;
        }
        case KOT_SESSION: {
            auto ses = (KSession*) o;
            s_port(io, &ses->handler);
            S64(ses->arg);
            break;
        }
        default:
            fprintf(stderr, "state: a kernel object of unknown type %d\n", o->type);
            io->err = true;
    }
}

// An object's stored length goes before it, so a later engine can add fields and an older state
// still lines up.
static void s_object_sized(IO* io, KObject* o) {
    u32 len = 0;
    if (io->writing) {
        size_t at = io->n;
        S32(len);
        s_object(io, o);
        len = io->n - at - sizeof len;
        memcpy(io->p + at, &len, sizeof len);
        return;
    }
    S32(len);
    size_t end = io->pos + len;
    if (end > io->n) {
        io->err = true;
        return;
    }
    size_t whole = io->n;
    io->n = end;
    s_object(io, o);
    io->n = whole;
    io->pos = end;
}

static void s_processor(IO* io, ArmCore* c) {
    SARR(c->r);
    S32(c->cpsr.w);
    S32(c->spsr);
    SARR(c->d);
    S32(c->fpscr.w);
    SARR(c->banked_r8_12);
    SARR(c->banked_sp);
    SARR(c->banked_lr);
    SARR(c->banked_spsr);
    S32(c->vector_base);
    S64(c->cycles);
    S8(c->halt);
    S8(c->irq);
}

static void s_event(IO* io, SchedulerEvent* ev) {
    char name[64] = "";
    u32 which = countof(callbacks);
    S64(ev->time);
    if (io->writing) {
        for (u32 i = 0; i < countof(callbacks); i++)
            if (callbacks[i].f == ev->handler) which = i;
        if (which == countof(callbacks)) {
            fprintf(stderr, "state: a scheduler event with no name\n");
            io->err = true;
            return;
        }
        strcpy(name, callbacks[which].name);
    }
    io_string(io, name, sizeof name);
    if (!io->writing) {
        for (u32 i = 0; i < countof(callbacks); i++)
            if (!strcmp(callbacks[i].name, name)) which = i;
        if (which == countof(callbacks)) {
            fprintf(stderr, "state: no scheduler event %s in this engine\n", name);
            io->err = true;
            return;
        }
        ev->handler = callbacks[which].f;
    }
    if (callbacks[which].object) {
        S_OBJ(ev->arg);
    } else {
        u64 number = (u64) ev->arg;
        S64(number);
        if (!io->writing) ev->arg = (void*) number;
    }
}

static void s_scheduler(IO* io, E3DS* s) {
    S64(s->sched.now);
    S_FIFO(s->sched.event_queue, s_event);
    S64(s->lastAudioFrame);
    S8(s->frame_complete);
}

// Page tables, memory blocks, handles and the ready list.
static void s_process(IO* io, E3DS* s) {
    auto p = &s->process;
    S32(p->nexttid);
    S32(p->allocTls);
    S32(p->used_memory);

    u32 tables = 0;
    if (io->writing)
        for (u32 l1 = 0; l1 < BIT(10); l1++) tables += p->ptab[l1] != nullptr;
    S32(tables);
    u32 l1 = 0;
    for (u32 t = 0; t < tables; t++) {
        if (io->writing)
            while (!p->ptab[l1]) l1++;
        S32(l1);
        if (l1 >= BIT(10)) {
            io->err = true;
            return;
        }
        if (!io->writing) p->ptab[l1] = ka_calloc(BIT(10), sizeof(PageEntry));
        u32 used = 0;
        if (io->writing)
            for (u32 l2 = 0; l2 < BIT(10); l2++) used += p->ptab[l1][l2].state != MEMST_FREE;
        S32(used);
        u32 l2 = 0;
        for (u32 e = 0; e < used; e++) {
            if (io->writing)
                while (p->ptab[l1][l2].state == MEMST_FREE) l2++;
            S32(l2);
            if (l2 >= BIT(10)) {
                io->err = true;
                return;
            }
            S32(p->ptab[l1][l2].paddr);
            S16(p->ptab[l1][l2].perm);
            S16(p->ptab[l1][l2].state);
            if (io->writing) l2++;
        }
        if (io->writing) l1++;
    }

    u32 count = 0;
    if (io->writing)
        for (VMBlock* b = p->vmblocks.next; b != &p->vmblocks; b = b->next) count++;
    S32(count);
    if (io->writing) {
        for (VMBlock* b = p->vmblocks.next; b != &p->vmblocks; b = b->next) {
            S32(b->startpg);
            S32(b->endpg);
            S32(b->perm);
            S32(b->state);
        }
    } else {
        p->vmblocks.next = p->vmblocks.prev = &p->vmblocks;
        for (u32 i = 0; i < count; i++) {
            VMBlock* b = ka_calloc(1, sizeof *b);
            S32(b->startpg);
            S32(b->endpg);
            S32(b->perm);
            S32(b->state);
            b->next = &p->vmblocks;
            b->prev = p->vmblocks.prev;
            b->prev->next = b;
            p->vmblocks.prev = b;
        }
    }

    count = 0;
    if (io->writing)
        for (FreeListNode* b = s->freelist.next; b != &s->freelist; b = b->next) count++;
    S32(count);
    if (io->writing) {
        for (FreeListNode* b = s->freelist.next; b != &s->freelist; b = b->next) {
            S32(b->startpg);
            S32(b->endpg);
        }
    } else {
        s->freelist.next = s->freelist.prev = &s->freelist;
        for (u32 i = 0; i < count; i++) {
            FreeListNode* b = ka_calloc(1, sizeof *b);
            S32(b->startpg);
            S32(b->endpg);
            b->next = &s->freelist;
            b->prev = s->freelist.prev;
            b->prev->next = b;
            s->freelist.prev = b;
        }
    }

    count = HANDLE_MAX;
    S32(count);
    for (u32 i = 0; i < count; i++) {
        if (i < HANDLE_MAX) {
            S_OBJ(p->handles[i]);
        } else {
            KObject* extra = nullptr;
            S_OBJ(extra);
        }
    }

    count = 0;
    if (io->writing)
        for (KThread* t = s->readylist.next; t != &s->readylist; t = t->next) count++;
    S32(count);
    if (io->writing) {
        for (KThread* t = s->readylist.next; t != &s->readylist; t = t->next) S_OBJ(t);
    } else {
        s->readylist.next = s->readylist.prev = &s->readylist;
        for (u32 i = 0; i < count; i++) {
            KThread* t = nullptr;
            S_OBJ(t);
            if (!t) {
                io->err = true;
                return;
            }
            t->next = &s->readylist;
            t->prev = s->readylist.prev;
            t->prev->next = t;
            s->readylist.prev = t;
        }
    }
}

static void s_lcdfb(IO* io, LCDFBInfo* f) {
    S32(f->vaddr);
    S32(f->fmt);
    S8(f->wasDisplayTransferred);
}

// gsp.c uses these as rings, by index, and their size only ever grows: every slot goes as it is.
static void s_lcdfbs(IO* io, ServiceData* v, int screen) {
    auto f = &v->gsp.lcdfbs[screen];
    u32 head = (u32) f->head, tail = (u32) f->tail, size = f->size, slots = FIFO_MAX(*f);
    S32(head);
    S32(tail);
    S32(size);
    S32(slots);
    if (!io->writing) {
        f->head = head;
        f->tail = tail;
        f->size = size;
    }
    for (u32 i = 0; i < slots; i++) {
        LCDFBInfo extra = {};
        s_lcdfb(io, i < FIFO_MAX(*f) ? &f->d[i] : &extra);
    }
}

// The services' own fields; their embedded objects are in the object chunk.
static void s_services(IO* io, E3DS* s) {
    auto v = &s->services;
    S32(v->apt.application_cpu_time_limit);
    S32(v->apt.swkbd.paramsvaddr);
    S32(v->apt.swkbd.shmemvaddr);
    S32(v->apt.nextparam.appid);
    S32(v->apt.nextparam.cmd);
    S32(v->apt.nextparam.paramsize);
    SARR(v->apt.nextparam.param);
    S_OBJ(v->apt.nextparam.kobj);

    S_OBJ(v->gsp.event);
    S8(v->gsp.registered);
    for (int i = 0; i < 2; i++) s_lcdfbs(io, v, i);

    S_OBJ(v->dsp.audio_event);
    S_OBJ(v->dsp.binary_event);
    S8(v->dsp.comp_loaded);
    S16(v->dsp.sem_mask);

    S32(v->fs.priority);

    S8(v->y2r.enableInterrupt);
    S8(v->y2r.busy);
    #define S_PLANE(p) S32(p.addr); S32(p.pitch); S32(p.gap); S32(p.size)
    S_PLANE(v->y2r.srcY);
    S_PLANE(v->y2r.srcU);
    S_PLANE(v->y2r.srcV);
    S_PLANE(v->y2r.dst);
    #undef S_PLANE
    S32(v->y2r.inputFmt);
    S32(v->y2r.outputFmt);
    S32(v->y2r.rotation);
    S32(v->y2r.blockMode);
    S16(v->y2r.width);
    S16(v->y2r.height);
    S16(v->y2r.alpha);
    SF32(v->y2r.coeffs.y_a);
    SF32(v->y2r.coeffs.r_v);
    SF32(v->y2r.coeffs.g_v);
    SF32(v->y2r.coeffs.g_u);
    SF32(v->y2r.coeffs.b_u);
    SF32(v->y2r.coeffs.r_off);
    SF32(v->y2r.coeffs.g_off);
    SF32(v->y2r.coeffs.b_off);

    S32(v->ldr.crs_addr);

    S_OBJ(v->mic.shmem);
    S8(v->mic.gain);
    S8(v->mic.sampling);
    S8(v->mic.encoding);
    S32(v->mic.sampleRate);
    S8(v->mic.loop);

    S8(v->cam.capturing);
    S32(v->cam.width);
    S32(v->cam.height);
    S8(v->cam.rgb);
    S8(v->cam.trimming);
    S16(v->cam.x0);
    S16(v->cam.y0);
    S16(v->cam.x1);
    S16(v->cam.y1);
    S32(v->cam.dstAddr);
}

static void s_bufinfo(IO* io, BufInfo* b) {
    S32(b->paddr);
    S32(b->len);
    S32(b->pos);
    S8(b->adpcm.indexScale);
    S16(b->adpcm.history[0]);
    S16(b->adpcm.history[1]);
    S8(b->looping);
    S8(b->adpcm_dirty);
    S16(b->id);
}

static void s_dsp(IO* io, DSP* d) {
    S32(d->audio_pipe_pos);
    SARR(d->binary_pipe);
    u32 channels = DSP_CHANNELS;
    S32(channels);
    for (u32 i = 0; i < channels; i++) {
        if (i < DSP_CHANNELS) {
            S_FIFO(d->bufQueues[i], s_bufinfo);
        } else {
            io->err = true;
            return;
        }
    }
}

// The game's side of the GPU. Textures and shaders come from guest memory again, the framebuffers'
// pictures from gpu_gl_fbs_capture.
static void s_gpu(IO* io, GPU* g) {
    #define S_SHADER(u)                                                        \
        SARR(u.progdata);                                                      \
        SARR(u.opdescs);                                                       \
        S32(u.code_idx);                                                       \
        S32(u.curuniform);                                                     \
        S32(u.curunifi);                                                       \
        SARR(u.floatuniform)
    S_SHADER(g->gsh);
    S_SHADER(g->vsh);
    #undef S_SHADER
    SARR(g->fixattrs);
    S32(g->curfixattr);
    S32(g->curfixi);
    SARR(g->lightLuts);
    SARR(g->fogLut);
    SARR(g->proctexMapLut);
    SARR(g->proctexNoiseLut);
    SARR(g->proctexLut);
    SARR(g->regs.w);
}

void savestate_put(void* ctx, const void* p, size_t n) {
    IO* io = ctx;
    io_bytes(io, (void*) p, n);
}

bool savestate_get(void* ctx, void* p, size_t n) {
    IO* io = ctx;
    if (io->pos + n > io->n) return false;
    io_bytes(io, p, n);
    return true;
}

static u64 ss_rom() {
    return XXH3_64bits(ctremu.romfilenoext, strlen(ctremu.romfilenoext));
}

void* savestate_capture(u64* size) {
    extern void* gpu_gl_fbs_capture(GPU*, u64*);
    return gpu_gl_fbs_capture(&ctremu.system.gpu, size);
}

static void ss_chunk(SS* ss, const char* tag, const void* p, u64 n) {
    ss_put(ss, tag, 4, ZSTD_e_continue);
    ss_put(ss, &n, sizeof n, ZSTD_e_continue);
    if (n) ss_put(ss, p, n, ZSTD_e_continue);
}

bool savestate_write(const char* path, const void* blob, u64 blob_size) {
    extern void fs_files_put(E3DS*, void*);
    E3DS* s = &ctremu.system;

    // Everything found while walking: the object bodies can bring more objects to light.
    IO io_ = {.writing = true, .s = s};
    IO* io = &io_;
    #define WRITTEN(name, ...)                                                 \
        io->p = nullptr;                                                       \
        io->n = io->cap = 0;                                                   \
        __VA_ARGS__;                                                           \
        u8* name = io->p;                                                      \
        u64 name##_n = io->n
    WRITTEN(cpu, s_processor(io, &s->cpu));
    WRITTEN(sched, s_scheduler(io, s));
    WRITTEN(proc, s_process(io, s));
    WRITTEN(svcs, s_services(io, s));
    WRITTEN(dsp, s_dsp(io, &s->dsp));
    WRITTEN(gpu, s_gpu(io, &s->gpu));
    WRITTEN(files, fs_files_put(s, io));
    KObject* e[MAX_EMBEDDED];
    u32 ne = embedded(s, e);
    WRITTEN(objs, {
        S32(ne);
        for (u32 i = 0; i < ne; i++) s_object_sized(io, e[i]);
        for (u32 i = 0; i < io->nheap; i++) s_object_sized(io, io->heap[i]);
    });
    WRITTEN(table, {
        S32(io->nheap);
        for (u32 i = 0; i < io->nheap; i++) S32(io->heap[i]->type);
    });
    #undef WRITTEN

    bool ok = !io->err;
    FILE* f = ok ? fopen(path, "wb") : nullptr;
    if (f) {
        SS ss_ = {.f = f, .cap = 1 << 20};
        SS* ss = &ss_;
        ss->buf = malloc(ss->cap);
        ss->c = ZSTD_createCCtx();
        ZSTD_CCtx_setParameter(ss->c, ZSTD_c_compressionLevel, 1);
        ZSTD_CCtx_setParameter(ss->c, ZSTD_c_checksumFlag, 1);
        ZSTD_CCtx_setParameter(ss->c, ZSTD_c_nbWorkers, 2);

        u32 version = SS_VERSION;
        u64 rom = ss_rom();
        ok = fwrite("T3DSSTAT", 8, 1, f) == 1 && fwrite(&version, 4, 1, f) == 1 &&
             fwrite(&rom, 8, 1, f) == 1;
        if (ok) {
            ss_chunk(ss, "CPU_", cpu, cpu_n);
            ss_chunk(ss, "SCHD", sched, sched_n);
            ss_chunk(ss, "OTAB", table, table_n);
            ss_chunk(ss, "OBJS", objs, objs_n);
            ss_chunk(ss, "PROC", proc, proc_n);
            ss_chunk(ss, "SVCS", svcs, svcs_n);
            ss_chunk(ss, "DSP_", dsp, dsp_n);
            ss_chunk(ss, "GPU_", gpu, gpu_n);
            ss_chunk(ss, "FILE", files, files_n);
            ss_chunk(ss, "MEMF", &s->physmem[FCRAM_PBASE], FCRAM_SIZE);
            ss_chunk(ss, "MEMV", &s->physmem[VRAM_PBASE], VRAM_SIZE);
            ss_chunk(ss, "MEMD", &s->physmem[DSPRAM_PBASE], DSPRAM_SIZE);
            ss_chunk(ss, "PICS", blob, blob_size);
            ss_chunk(ss, "END_", nullptr, 0);
            ss_put(ss, nullptr, 0, ZSTD_e_end);
        }
        ok = ok && !ss->err && !fflush(f) && !fsync(fileno(f));
        if (fclose(f)) ok = false;
        ZSTD_freeCCtx(ss->c);
        free(ss->buf);
    } else {
        ok = false;
    }
    free(cpu);
    free(sched);
    free(proc);
    free(svcs);
    free(dsp);
    free(gpu);
    free(files);
    free(objs);
    free(table);
    free(io->heap);
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

typedef struct {
    char tag[4];
    u8* p;
    u64 n;
} Chunk;

static Chunk* chunk_find(Chunk* c, u32 count, const char* tag) {
    for (u32 i = 0; i < count; i++)
        if (!memcmp(c[i].tag, tag, 4)) return &c[i];
    return nullptr;
}

static bool chunk_is_picture_or_memory(const char* tag) {
    return !memcmp(tag, "MEM", 3) || !memcmp(tag, "PICS", 4) || !memcmp(tag, "END_", 4);
}

static IO chunk_reader(E3DS* s, Chunk* c, KObject** heap, u32 nheap) {
    return (IO) {.p = c ? c->p : nullptr, .n = c ? c->n : 0, .s = s, .heap = heap, .nheap = nheap};
}

// A state this engine cannot read stays beside the one that replaces it, for an engine that can.
static void ss_keep_refused(const char* path, u32 version) {
    char kept[1024];
    snprintf(kept, sizeof kept, "%s.refused-%u", path, version);
    struct stat st;
    if (stat(kept, &st) || !st.st_size) rename(path, kept);
    fprintf(stderr, "state: %s is format %u, this engine reads %u..%u; kept as %s\n", path, version,
            SS_OLDEST, SS_VERSION, kept);
}

// Everything but the renderer's side, which savestate_read_finish takes on the thread with the
// GL context. Nothing else may touch the machine meanwhile.
SSLoaded* savestate_read(const char* path, bool* wrecked) {
    extern void fs_files_close(E3DS*);
    extern void fs_files_get(E3DS*, void*);
    extern void jit_state_loaded(ArmCore*);
    E3DS* s = &ctremu.system;
    *wrecked = false;
    SS ss = {};
    ss.f = fopen(path, "rb");
    if (!ss.f) return nullptr;
    char magic[8] = "";
    u32 version = 0;
    u64 rom = 0;
    if (fread(magic, 8, 1, ss.f) != 1 || memcmp(magic, "T3DSSTAT", 8) || fread(&version, 4, 1, ss.f) != 1) {
        fprintf(stderr, "state: %s is not a 3DS state\n", path);
        fclose(ss.f);
        return nullptr;
    }
    if (version < SS_OLDEST || version > SS_VERSION) {
        fclose(ss.f);
        ss_keep_refused(path, version);
        return nullptr;
    }
    if (fread(&rom, 8, 1, ss.f) != 1 || rom != ss_rom()) {
        fprintf(stderr, "state: %s is not for this game\n", path);
        fclose(ss.f);
        return nullptr;
    }
    ss.cap = 1 << 20;
    ss.buf = malloc(ss.cap);
    ss.d = ZSTD_createDCtx();
    ss.in.src = ss.buf;

    Chunk chunks[SS_MAX_CHUNKS] = {};
    u32 count = 0;
    SSLoaded* out = nullptr;
    KObject** heap = nullptr;
    bool ok = true;
    char tag[4];
    u64 len = 0;

    // All the small chunks first: up to here a broken file changes nothing.
    while (ok) {
        ok = ss_get(&ss, tag, 4) && ss_get(&ss, &len, sizeof len);
        if (!ok || chunk_is_picture_or_memory(tag)) break;
        ok = count < SS_MAX_CHUNKS && len < (64u << 20);
        if (!ok) break;
        memcpy(chunks[count].tag, tag, 4);
        chunks[count].n = len;
        chunks[count].p = malloc(len + 1);
        ok = ss_get(&ss, chunks[count].p, len);
        count++;
    }
    static const char* const required[] = {"CPU_", "SCHD", "OTAB", "OBJS", "PROC", "SVCS", "DSP_", "GPU_", "FILE"};
    for (u32 i = 0; i < countof(required); i++) ok = ok && chunk_find(chunks, count, required[i]);
    u32 nheap = 0;
    if (ok) {
        Chunk* c = chunk_find(chunks, count, "OTAB");
        ok = c->n >= 4;
        if (ok) memcpy(&nheap, c->p, 4);
        ok = ok && c->n >= 4 + (u64) nheap * 4;
        for (u32 i = 0; ok && i < nheap; i++) {
            u32 type;
            memcpy(&type, c->p + 4 + i * 4, 4);
            ok = type < KOT_MAX && type != KOT_PROCESS;
        }
    }
    if (!ok) {
        fprintf(stderr, "state: %s could not be read\n", path);
        goto end;
    }

    // from here on the machine is being replaced
    *wrecked = true;
    {
        jit_state_loaded(&s->cpu);
        fs_files_close(s);

        // every kernel object comes from the file
        ka_reset();
        memset(s->process.ptab, 0, sizeof s->process.ptab);
        memset(s->process.handles, 0, sizeof s->process.handles);

        heap = calloc(nheap ? nheap : 1, sizeof *heap);
        Chunk* c = chunk_find(chunks, count, "OTAB");
        for (u32 i = 0; i < nheap; i++) {
            static const u32 sizes[KOT_MAX] = {
                [KOT_THREAD] = sizeof(KThread),     [KOT_MUTEX] = sizeof(KMutex),
                [KOT_SEMAPHORE] = sizeof(KSemaphore), [KOT_EVENT] = sizeof(KEvent),
                [KOT_TIMER] = sizeof(KTimer),       [KOT_SHAREDMEM] = sizeof(KSharedMem),
                [KOT_ARBITER] = sizeof(KArbiter),   [KOT_SESSION] = sizeof(KSession),
                [KOT_RESLIMIT] = sizeof(KObject),
            };
            u32 type;
            memcpy(&type, c->p + 4 + i * 4, 4);
            heap[i] = ka_calloc(1, sizes[type]);
            heap[i]->type = type;
        }

        KObject* e[MAX_EMBEDDED];
        u32 ne = embedded(s, e);
        IO io_ = chunk_reader(s, chunk_find(chunks, count, "OBJS"), heap, nheap);
        IO* io = &io_;
        u32 stored = 0;
        S32(stored);
        for (u32 i = 0; i < stored; i++) {
            if (i < ne) s_object_sized(io, e[i]);
            else {
                // an embedded object this engine no longer has
                u32 skip = 0;
                S32(skip);
                io->pos += skip;
            }
        }
        // ones newer than the file keep their first state, but their lists were in the old arena
        for (u32 i = stored; i < ne; i++) {
            IO none = chunk_reader(s, nullptr, heap, nheap);
            s_object(&none, e[i]);
        }
        for (u32 i = 0; i < nheap; i++) s_object_sized(io, heap[i]);
        if (io->err) fprintf(stderr, "state: %s is broken in OBJS at %zu\n", path, io->pos);
        ok = !io->err;

        #define APPLY(tag_, call)                                              \
            do {                                                               \
                IO one_ = chunk_reader(s, chunk_find(chunks, count, tag_), heap, nheap); \
                io = &one_;                                                    \
                call;                                                          \
                if (io->err) fprintf(stderr, "state: %s is broken in %s at %zu\n", path, tag_, io->pos); \
                ok = ok && !io->err;                                           \
            } while (0)
        APPLY("CPU_", s_processor(io, &s->cpu));
        APPLY("SCHD", s_scheduler(io, s));
        APPLY("PROC", s_process(io, s));
        APPLY("SVCS", s_services(io, s));
        APPLY("DSP_", s_dsp(io, &s->dsp));
        memset(s->services.fs.files, 0, sizeof s->services.fs.files);
        memset(s->services.fs.dirs, 0, sizeof s->services.fs.dirs);
        APPLY("FILE", fs_files_get(s, io));

        // the renderer goes on presenting from its own struct: the guest's side of it on the GL
        // thread at the end
        out = calloc(1, sizeof *out);
        out->gpu = calloc(1, sizeof(GPU));
        APPLY("GPU_", s_gpu(io, out->gpu));
        #undef APPLY
        if (!ok) goto end;

        ss_map_virtual(s);

        // what is left: the memories and the pictures, in any order
        bool ended = false;
        while (ok && !ended) {
            if (!memcmp(tag, "END_", 4)) {
                ended = true;
            } else if (!memcmp(tag, "MEMF", 4) && len == FCRAM_SIZE) {
                ok = ss_get(&ss, &s->physmem[FCRAM_PBASE], len);
            } else if (!memcmp(tag, "MEMV", 4) && len == VRAM_SIZE) {
                ok = ss_get(&ss, &s->physmem[VRAM_PBASE], len);
            } else if (!memcmp(tag, "MEMD", 4) && len == DSPRAM_SIZE) {
                ok = ss_get(&ss, &s->physmem[DSPRAM_PBASE], len);
            } else if (!memcmp(tag, "PICS", 4) && len < (1ull << 30)) {
                free(out->blob);
                out->blob = malloc(len + 1);
                ok = ss_get(&ss, out->blob, len);
            } else {
                u8 skip[4096]; // from a later engine
                for (u64 left = len; ok && left;) {
                    size_t step = left < sizeof skip ? left : sizeof skip;
                    ok = ss_get(&ss, skip, step);
                    left -= step;
                }
            }
            if (ok && !ended) ok = ss_get(&ss, tag, 4) && ss_get(&ss, &len, sizeof len);
        }
        ok = ok && out->blob;
        if (!ok) goto end;
        *wrecked = false;
    }

end:
    for (u32 i = 0; i < count; i++) free(chunks[i].p);
    free(heap);
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
