#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>


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

typedef struct {
    u32 rows, cols;
    u64 alloc;
    f32* data;
} matrix;

matrix* mat_create(u32 rows, u32 cols);
void mat_free(matrix* m);
b32 mat_add(matrix* out, const matrix* a, const matrix* b);
b32 mat_comp_mul(matrix* out, const matrix* a, const matrix* b);
b32 mat_scale(matrix* out, const matrix* in, f32 scale);
b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 transpose_a, b8 transpose_b
);

int main(void) {



    return 0;
}

matrix* mat_create(u32 rows, u32 cols) {
    matrix* out = (matrix*)malloc(sizeof(matrix));

    u64 alloc = (u64)rows * cols;

    out->rows = rows;
    out->cols = cols;
    out->alloc = alloc;
    out->data = (f32*)malloc(alloc * sizeof(f32));

    return out;
}

void mat_free(matrix* m) {
    free(m->data);
    free(m);
}

b32 mat_add(matrix* out, const matrix* a, const matrix* b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return false;
    }

    u64 required_alloc = (u64)a->rows * a->cols;

    for (u32 row = 0; row < out->rows; row++) {
        for (u32 col = 0; col < out->cols; col++) {
        }
    }
}

b32 mat_comp_mul(matrix* out, const matrix* a, const matrix* b);

b32 mat_scale(matrix* out, const matrix* in, f32 scale);

b32 mat_mul(
    matrix* out, const matrix* a, const matrix* b,
    b8 transpose_a, b8 transpose_b
);


