
#include "base.h"
#include "platform.h"
#include "arena.h"

#include "arena.c"
#include "platform.c"

int main(void) {
    mem_arena* perm_arena = arena_create(GiB(1), MiB(1));

    while (1) {
        printf("allocating 16 MiB\n");
        arena_push(perm_arena, MiB(16), false);
        getc(stdin);
    }

    arena_destroy(perm_arena);

    return 0;
}

