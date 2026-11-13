#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define UNICODE
#include <Windows.h>
#endif

typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef i8 b8;
typedef i32 b32;

typedef float f32;
typedef double f64;

#define ABS(x) ((x) < 0 ? -(x) : (x))

typedef struct {
    u8* str;

    // `size` denotes the number of bytes *NOT* the number of codepoints
    u64 size;
} string8;

typedef struct {
    u32 char_width, char_height;
    u32* codepoints;
} braille_map;

braille_map braille_map_init(u32 char_width, u32 char_height) {
    braille_map out = { 
        .char_width = char_width,
        .char_height = char_height,
        .codepoints = (u32*)malloc(char_width * char_height * sizeof(u32)),
    };

    u32 size = char_width * char_height;
    for (u32 i = 0; i < size; i++) {
        out.codepoints[i] = 0x2800;
    }

    return out;
}

void braille_map_set_at(braille_map* bm, u32 dot_x, u32 dot_y) {
    if (dot_x >= bm->char_width * 2 || dot_y >= bm->char_height * 4) {
        return;
    }

    static u8 dot_map[] = {
        0x01, 0x08,
        0x02, 0x10,
        0x04, 0x20, 
        0x40, 0x80
    };

    u32 char_x = dot_x / 2;
    u32 char_y = dot_y / 4;

    u32 local_x = dot_x % 2;
    u32 local_y = dot_y % 4;

    u32 codepoint_index = char_x + char_y * bm->char_width;
    u32 local_index = local_x + local_y * 2;

    bm->codepoints[codepoint_index] |= dot_map[local_index];
}

void braille_map_line_at(braille_map* bm, i32 x0, i32 y0, i32 x1, i32 y1) {
    i32 dx = ABS(x1 - x0);
    i32 dy = ABS(y1 - y0);

    i32 sx = x0 < x1 ? 1 : -1;
    i32 sy = y0 < y1 ? 1 : -1;

    i32 err = (dx > dy ? dx : -dy) / 2;

    i32 x = x0, y = y0;
    while (1) {
        if (x >= 0 && y >= 0) {
            braille_map_set_at(bm, x, y);
        }

        if (x == x1 && y == y1) { break; }

        i32 e2 = err;
        if (e2 > -dx) { err -= dy; x += sx; }
        if (e2 < dy) { err += dx; y += sy; }
    }
}

string8 braille_map_encode(braille_map* bm) {
    u32 encoded_size = bm->char_width * bm->char_height * 3 +
        (bm->char_height - 1);

    string8 out = {
        .size = encoded_size,
        .str = malloc(encoded_size)
    };

    u64 pos = 0;

    for (u32 y = 0; y < bm->char_height; y++) {
        for (u32 x = 0; x < bm->char_width; x++) {
            u32 codepoint = bm->codepoints[x + y * bm->char_width];

            out.str[pos++] = 0b11100000 | ((codepoint >> 12) & 0xf);
            out.str[pos++] = 0b10000000 | ((codepoint >> 6) & 0b111111);
            out.str[pos++] = 0b10000000 | (codepoint & 0b111111);
        }

        if (y < bm->char_height - 1) {
            out.str[pos++] = '\n';
        }
    }

    return out;
}

#define N 16
#define PI 3.14159265358979323

int main(void) {
    #ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    #endif

    braille_map map = braille_map_init(60, 20);

    f32 xs[N];
    f32 ys[N];

    for (u32 i = 0; i < N; i++) {
        xs[i] = 2.0f * PI * (f32)i / (f32)(N-1);
        ys[i] = sinf(xs[i]);
    }

    f32 xscale = ((f32)(map.char_width * 2) / (2.0f * PI));
    f32 yscale = -(f32)(map.char_height * 2);

    i32 x0 = (i32)(xs[0] * xscale);
    i32 y0 = (i32)((ys[0] - 1.0f) * yscale);

    for (u32 i = 1; i < N; i++) {
        i32 x1 = (i32)(xs[i] * xscale);
        i32 y1 = (i32)((ys[i] - 1.0f) * yscale);

        braille_map_line_at(&map, x0, y0, x1, y1);

        x0 = x1;
        y0 = y1;
    }

    string8 encoded = braille_map_encode(&map);
    printf("%.*s\n", (int)encoded.size, (char*)encoded.str);

    return 0;
}

