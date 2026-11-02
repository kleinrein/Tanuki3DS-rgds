#ifndef KARENA_H
#define KARENA_H

#include <stddef.h>

// Kernel objects live here, at the same address on every run, so that a save
// state can hold them with their pointers as they are.
void* ka_malloc(size_t size);
void* ka_calloc(size_t n, size_t size);
void ka_free(void* p);
void* ka_base();
size_t ka_used();
void ka_ensure(size_t used);

#endif
