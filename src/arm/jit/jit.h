#ifndef JIT_BLOCK_H
#define JIT_BLOCK_H

#include "arm/arm_core.h"

#include "ir.h"

typedef void (*JITFunc)();

typedef struct {
    u32 attrs;
    u32 addr;
} BlockLocation;

typedef struct _JITBlock {
    JITFunc code;
    void* backend;

    u32 attrs;
    u32 start_addr;
    u32 end_addr;
    u32 numinstr;

    ArmCore* cpu;
    IRBlock* ir;

    Vec(BlockLocation) linkingblocks;

} JITBlock;

typedef struct {
    bool ir_interpret;

    int max_block_instrs;
    bool optimize;
    bool optimize_literals;
    bool linking;
} JITConfig;

extern JITConfig g_jit_config;

JITBlock* create_jit_block(ArmCore* cpu, u32 addr);
void destroy_jit_block(JITBlock* block);

void jit_exec(JITBlock* block);
JITBlock* get_jitblock(ArmCore* cpu, u32 attrs, u32 addr);

void jit_invalidate_range(ArmCore* cpu, u32 start_addr, u32 len);
bool jit_addr_mapped(u32 addr);
void jit_list_start(const char* rom);
void jit_list_step(ArmCore* cpu, u64 budget_ns);
void jit_free_all(ArmCore* cpu);

void arm_exec_jit(ArmCore* cpu);

// spike: the inline cache for block ends that cannot link (bx lr, pointer calls, mode changes): the
// block looks its target up here before returning to C. Indexed by guest pc bits 1.., checked by pc
// and the block attributes (the low 6 bits of the cpsr). Filled by arm_exec_jit, emptied whenever a
// block is destroyed.
// 4096 entries left 1750 conflict misses a refresh in 3D Land (ARM code only ever used half of
// them); 65536 (1 MB) is the size that holds a 3D game's call graph.
#define JIT_IC_BITS 16
#define JIT_IC_INDEX(pc) ((((pc) >> 2) ^ ((pc) >> 1)) & ((1 << JIT_IC_BITS) - 1))
typedef struct {
    u32 pc, attrs;
    JITFunc code;
} JitIcEnt;
extern JitIcEnt g_jit_ic[1 << JIT_IC_BITS];

#endif
