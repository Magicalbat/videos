
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef int8_t i8;
typedef int32_t i32;
typedef uint8_t u8;
typedef uint32_t u32;
typedef uint64_t u64;

typedef struct {
	unsigned int width;
	unsigned int height;
	unsigned char channels;
	unsigned char colorspace;
} qoi_desc;

typedef struct {
    u8 r, g, b, a;
} _qoi_rgba8;

#define _QOI_READ_BE32(p) (((u32)(p)[0] << 24) | ((u32)(p)[1] << 16) | ((u32)(p)[2] << 8) | (u32)(p)[3])

void* qoi_decode(const u8* data, u32 size, qoi_desc *desc, int channels) {
    if (channels != 3 && channels != 4) { return NULL; }
    if (size <= 14) { return NULL; }

    if (
        data[0] != 'q' || 
        data[1] != 'o' || 
        data[2] != 'i' || 
        data[3] != 'f' 
    ) {
        return NULL;
    }

    desc->width = _QOI_READ_BE32(data + 4);
    desc->height = _QOI_READ_BE32(data + 8);
    desc->channels = data[12];
    desc->colorspace = data[13];

    if (desc->channels != 3 && desc->channels != 4) {
        return NULL;
    }

    _qoi_rgba8 lookup[64] = { 0 };
    _qoi_rgba8 prev_pixel = { .a = 255 };

    u64 out_size = desc->width * desc->height * desc->channels;
    u8* out = (u8*)malloc(out_size);
    memset(out, 0, out_size);
    u64 out_pos = 0;

#define WRITE_PIXEL(pix) do { \
    out[out_pos++] = (pix).r; \
    out[out_pos++] = (pix).g; \
    out[out_pos++] = (pix).b; \
    if (channels == 4) { out[out_pos++] = (pix).a; } \
    prev_pixel = (pix); \
    lookup[((pix).r * 3 + (pix).g * 5 + (pix).b * 7 + (pix).a * 11) % 64] = (pix); \
} while(0);

    u32 pos = 14;
    while (pos < size && out_pos < out_size) {
        u8 cur_byte = data[pos++];

        if (cur_byte == 0b11111110) {
            // RGB
            _qoi_rgba8 pixel = { 0 };

            pixel.r = data[pos++];
            pixel.g = data[pos++];
            pixel.b = data[pos++];
            pixel.a = prev_pixel.a;

            WRITE_PIXEL(pixel);
        } else if (cur_byte == 0b11111111) {
            // RGBA
            _qoi_rgba8 pixel = { 0 };

            pixel.r = data[pos++];
            pixel.g = data[pos++];
            pixel.b = data[pos++];
            pixel.a = data[pos++];

            WRITE_PIXEL(pixel);
        } else {
            u8 test_bits = cur_byte >> 6;

            switch (test_bits) {
                // INDEX
                case 0b00: {
                    u8 index = cur_byte & 0b111111;
                    WRITE_PIXEL(lookup[index]);
                } break;

                // DIFF
                case 0b01: {
                    i8 dr = ((cur_byte >> 4) & 0b11) - 2;
                    i8 dg = ((cur_byte >> 2) & 0b11) - 2;
                    i8 db = ((cur_byte >> 0) & 0b11) - 2;

                    _qoi_rgba8 pixel = {
                        .r = (u8)(prev_pixel.r + dr),
                        .g = (u8)(prev_pixel.g + dg),
                        .b = (u8)(prev_pixel.b + db),
                        .a = prev_pixel.a,
                    };

                    WRITE_PIXEL(pixel);
                } break;

                // LUMA
                case 0b10: {
                    u8 next_byte = data[pos++];

                    i8 dg = ((i32)cur_byte & 0b111111) - 32;
                    i8 dr_dg = (i8)(next_byte >> 4) - 8;
                    i8 db_dg = (i8)(next_byte & 0b1111) - 8;

                    i8 dr = dr_dg + dg;
                    i8 db = db_dg + dg;

                    _qoi_rgba8 pixel = {
                        .r = (u8)(prev_pixel.r + dr),
                        .g = (u8)(prev_pixel.g + dg),
                        .b = (u8)(prev_pixel.b + db),
                        .a = prev_pixel.a
                    };

                    WRITE_PIXEL(pixel);
                } break;

                // RUN
                case 0b11: {
                    u32 run_len = (cur_byte & 0b111111) + 1;

                    for (u32 i = 0; i < run_len; i++) {
                        WRITE_PIXEL(prev_pixel);
                    }
                } break;
            }
        }
    }

#undef WRITE_PIXEL

    return out;
}
