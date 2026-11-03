#ifndef KARENA_H
#define KARENA_H

#include <stddef.h>

// Kernel objects live here. A save state rebuilds them all, so loading one starts it over.
void* ka_malloc(size_t size);
void* ka_calloc(size_t n, size_t size);
void ka_free(void* p);
void* ka_base();
size_t ka_used();
void ka_reset();

#endif
