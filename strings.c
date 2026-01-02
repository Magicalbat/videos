#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdbool.h>

typedef int32_t i32;

typedef uint8_t u8;
typedef uint64_t u64;

typedef i32 b32;

#define MIN(a, b) ((a) < (b) ? (a) : (b))

typedef struct {
    u8* str;
    u64 size;
} string8;

#define STR8_LIT(s) (string8){ (u8*)(s), sizeof((s)) - 1 }
#define STR8_FMT(s8) (int)(s8).size, (s8).str

string8 str8_substr(string8 str, u64 start, u64 end);

int main(void) {
    string8 test = STR8_LIT("hello");

    return 0;
}

string8 str8_substr(string8 str, u64 start, u64 end) {
    end = MIN(end, str.size);
    start = MIN(start, end);
    return (string8){ str.str + start, end - start };
}

