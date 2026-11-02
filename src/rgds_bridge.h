#ifndef RGDS_BRIDGE_H
#define RGDS_BRIDGE_H

#include <stdbool.h>

#include "emulator.h"

void rgds_bridge_start();
bool rgds_bridge_on();
// Between frames, on the emulation thread: commands, saves and loads. True while
// the shared menu is up: the game holds still and the frames go on.
bool rgds_bridge_tick();
// The state the launcher placed for this start, before the first frame.
void rgds_bridge_autoload();
// On the renderer thread, once per panel after the game's picture is drawn.
void rgds_bridge_panel(int panel);
// A state is being written or read on its own thread.
bool rgds_bridge_busy();
// What went wrong, in crashes.log in the game's settings folder.
void rgds_crash_note(const char* what, void* addr, void* pc, void* lr);
// The Graphics page's choices, on the renderer thread before a panel is drawn.
int rgds_bridge_filter();
void rgds_bridge_layout(int panel, Rect* top, Rect* bot);

#endif
