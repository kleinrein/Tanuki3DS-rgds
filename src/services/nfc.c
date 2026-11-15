#include "nfc.h"

#include "3ds.h"

enum {
    NFC_NOTINIT,
    NFC_IDLE,
    NFC_SCANNING,
    NFC_INRANGE,
    NFC_OUTOFRANGE,
    NFC_LOADED,
};

// what the real module answers when asked about a tag that is not there
#define NFC_WRONGSTATE 0xc8a17600

DECL_PORT(nfc) {
    u32* cmdbuf = PTR(cmd_addr);
    auto nfc = &s->services.nfc;
    switch (cmd.command) {
        case 0x0001:
            linfo("Initialize %d", cmdbuf[1] & 0xff);
            nfc->state = NFC_IDLE;
            cmdbuf[0] = IPCHDR(1, 0);
            cmdbuf[1] = 0;
            break;
        case 0x0002:
            linfo("Shutdown");
            nfc->state = NFC_NOTINIT;
            cmdbuf[0] = IPCHDR(1, 0);
            cmdbuf[1] = 0;
            break;
        case 0x0003:
        case 0x0004:
            linfo("Start/StopCommunication");
            cmdbuf[0] = IPCHDR(1, 0);
            cmdbuf[1] = 0;
            break;
        case 0x0005:
            linfo("StartTagScanning");
            if (nfc->state != NFC_NOTINIT) nfc->state = NFC_SCANNING;
            cmdbuf[0] = IPCHDR(1, 0);
            cmdbuf[1] = 0;
            break;
        case 0x0006:
            linfo("StopTagScanning");
            if (nfc->state != NFC_NOTINIT) nfc->state = NFC_IDLE;
            cmdbuf[0] = IPCHDR(1, 0);
            cmdbuf[1] = 0;
            break;
        case 0x0008:
            linfo("ResetTagScanState");
            cmdbuf[0] = IPCHDR(1, 0);
            cmdbuf[1] = 0;
            break;
        case 0x000b:
        case 0x000c: {
            u32 handle = srvobj_make_handle(
                s, cmd.command == 0x000b ? &nfc->in_range.hdr
                                         : &nfc->out_of_range.hdr);
            linfo("GetTag%sRangeEvent %08x",
                  cmd.command == 0x000b ? "In" : "OutOf", handle);
            cmdbuf[0] = IPCHDR(1, 2);
            cmdbuf[1] = 0;
            cmdbuf[2] = 0;
            cmdbuf[3] = handle;
            break;
        }
        case 0x000d:
            linfo("GetTagState");
            cmdbuf[0] = IPCHDR(2, 0);
            cmdbuf[1] = 0;
            cmdbuf[2] = nfc->state;
            break;
        case 0x000f:
            linfo("CommunicationGetStatus");
            cmdbuf[0] = IPCHDR(2, 0);
            cmdbuf[1] = 0;
            cmdbuf[2] = 2; // initialized
            break;
        case 0x0012:
            linfo("CommunicationGetResult");
            cmdbuf[0] = IPCHDR(2, 0);
            cmdbuf[1] = 0;
            cmdbuf[2] = 0;
            break;
        default:
            // everything else reads or writes a tag
            lwarn("no tag for command 0x%04x (%x,%x,%x,%x,%x)", cmd.command,
                  cmdbuf[1], cmdbuf[2], cmdbuf[3], cmdbuf[4], cmdbuf[5]);
            cmdbuf[0] = IPCHDR(1, 0);
            cmdbuf[1] = NFC_WRONGSTATE;
            break;
    }
}
