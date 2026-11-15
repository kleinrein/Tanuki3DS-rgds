#ifndef NFC_H
#define NFC_H

#include "kernel/thread.h"

#include "srv.h"

// no amiibo ever comes into range: the reader works and finds nothing
typedef struct {
    KEvent in_range;
    KEvent out_of_range;
    u8 state;
} NFCData;

DECL_PORT(nfc);

#endif
