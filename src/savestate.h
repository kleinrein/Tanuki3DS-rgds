#ifndef SAVESTATE_H
#define SAVESTATE_H

#include <stdbool.h>

#include "common.h"

typedef struct SSLoaded SSLoaded;

// A state goes in steps, so that the renderer can keep showing the menu meanwhile.
// With the GL context, renderer stopped: the framebuffers' pictures.
void* savestate_capture(u64* size);
// On any thread, the machine not running.
bool savestate_write(const char* path, const void* blob, u64 blob_size);
// On any thread, the machine not running. Null when the file is not for this game and build and
// nothing was touched, or when it broke later on: then *wrecked says the machine cannot go on.
SSLoaded* savestate_read(const char* path, bool* wrecked);
// With the GL context, renderer stopped.
void savestate_read_finish(SSLoaded* loaded);

#endif
