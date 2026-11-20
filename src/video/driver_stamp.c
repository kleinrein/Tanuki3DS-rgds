#include <link.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>

// The warm step in renderer_gl.c compares this with the one each game's shader
// list was last built with: Mesa's disk cache is keyed by the same build id, so
// a different one means every program links from scratch.
typedef struct {
    char* out;
    int size;
} StampOut;

static int stamp_cb(struct dl_phdr_info* info, size_t size, void* data) {
    (void) size;
    if (!info->dlpi_name || !strstr(info->dlpi_name, "libgallium")) return 0;
    StampOut* o = data;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        const ElfW(Phdr)* ph = &info->dlpi_phdr[i];
        if (ph->p_type != PT_NOTE) continue;
        const unsigned char* p = (const unsigned char*) (info->dlpi_addr + ph->p_vaddr);
        const unsigned char* end = p + ph->p_memsz;
        while (p + sizeof(ElfW(Nhdr)) <= end) {
            const ElfW(Nhdr)* nh = (const ElfW(Nhdr)*) p;
            const unsigned char* name = p + sizeof *nh;
            const unsigned char* desc = name + ((nh->n_namesz + 3) & ~3u);
            if (nh->n_type == NT_GNU_BUILD_ID && nh->n_namesz == 4 && !memcmp(name, "GNU", 4)) {
                int n = 0;
                for (unsigned j = 0; j < nh->n_descsz && n + 3 < o->size; j++) {
                    n += snprintf(o->out + n, o->size - n, "%02x", desc[j]);
                }
                return 1;
            }
            p = desc + ((nh->n_descsz + 3) & ~3u);
        }
    }
    // no build id: the file's size and time stand in
    struct stat st;
    if (stat(info->dlpi_name, &st) == 0) {
        snprintf(o->out, o->size, "size %lld mtime %lld", (long long) st.st_size,
                 (long long) st.st_mtime);
    }
    return 1;
}

void gpu_gl_driver_stamp(char* out, int size) {
    out[0] = 0;
    StampOut o = {out, size};
    dl_iterate_phdr(stamp_cb, &o);
    if (!out[0]) snprintf(out, size, "unknown");
}
