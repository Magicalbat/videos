
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <stdbool.h>
#include <stdint.h>
#include <math.h>

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

#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define READ_BE16(m) (u16)( \
    ((u16)(((u8*)(m))[0]) << 8) | \
    ((u16)(((u8*)(m))[1]) << 0)   \
)

#define READ_BE32(m) (u32)( \
    ((u32)(((u8*)(m))[0]) << 24) | \
    ((u32)(((u8*)(m))[1]) << 16) | \
    ((u32)(((u8*)(m))[2]) <<  8) | \
    ((u32)(((u8*)(m))[3]) <<  0)   \
)

#define TAG(m) READ_BE32(m)

typedef struct {
    u8* str;
    u64 size;
} string8;

typedef struct {
    i16 x, y;
} v2_i16;

typedef struct {
    f32 x, y;
} v2_f32;

typedef struct {
    u32 glyf, loca, head;

    u32 cmap_subtable;

    i16 loca_format;
} font_info;

typedef enum {
    FONT_POINT_FLAG_NONE        = 0b000,
    FONT_POINT_FLAG_ON_CURVE    = 0b001,
    FONT_POINT_FLAG_CONTOUR_END = 0b010,
    FONT_POINT_FLAG_GENERATED   = 0b100,
} font_point_flag_enum;

typedef u8 font_point_flag;

typedef struct {
    i16 x_min;
    i16 y_min;
    i16 x_max;
    i16 y_max;

    u32 num_segments;
    u32 num_points;

    u32 capacity;

    font_point_flag* flags;
    v2_i16* points;
} font_glyph;

typedef struct {
    u32 width, height;
    u8* data; 
} bitmap_r8;

void font_init(string8 file, font_info* info);
u32 font_glyph_index(string8 file, font_info* info, u32 codepoint);
void font_load_glyph(string8 file, font_info* info, font_glyph* glyph, u32 glyph_index);
void font_free_glyph(font_glyph* glyph);
f32 font_scale_for_em(string8 file, font_info* info, f32 pixels_per_em);
bitmap_r8 font_raster_glyph(font_glyph* glyph, f32 scale, u32 padding);

string8 read_file(const char* file_name);

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("not enough args\n");
        return 0;
    }

    string8 file = read_file("Symbola.ttf");
    font_info font = { 0 };
    font_init(file, &font);

    u32 glyph_index = font_glyph_index(file, &font, argv[1][0]);
    font_glyph glyph = { 0 };

    font_load_glyph(file, &font, &glyph, glyph_index);

    f32 scale = font_scale_for_em(file, &font, atoi(argv[2]));
    bitmap_r8 bitmap = font_raster_glyph(&glyph, scale, 1);

    for (u32 y = 0; y < bitmap.height; y++) {
        for (u32 x = 0; x < bitmap.width; x++) {
            u8 c = " .-oaA@#"[bitmap.data[x + y * bitmap.width] >> 5];
            printf("%c%c", c, c);
        }
        printf("\n");
    }

    return 0;
}

void font_init(string8 file, font_info* info) {
    memset(info, 0, sizeof(font_info));

    u32 cmap = 0;

    u32 num_tables = READ_BE16(file.str + 4);
    for (u32 i = 0; i < num_tables; i++) {
        u32 mem_offset = 12 + 16 * i;

        u32 tag = READ_BE32(file.str + mem_offset + 0);
        u32 offset = READ_BE32(file.str + mem_offset + 8);
        u32 length = READ_BE32(file.str + mem_offset + 12);

        if (tag == TAG("cmap")) { cmap = offset; }
        else if (tag == TAG("glyf")) { info->glyf = offset; }
        else if (tag == TAG("loca")) { info->loca = offset; }
        else if (tag == TAG("head")) { info->head = offset; }
    }

    u32 cmap_num_subtables = READ_BE16(file.str + cmap + 2);
    for (u32 i = 0; i < cmap_num_subtables; i++) {
        u32 mem_offset = cmap + 4 + 8 * i;

        u16 platform_id = READ_BE16(file.str + mem_offset + 0);
        u16 encoding_id = READ_BE16(file.str + mem_offset + 2);
        u32 offset      = READ_BE32(file.str + mem_offset + 4);

        u16 format = READ_BE16(file.str + cmap + offset);

        if (format == 12) {
            info->cmap_subtable = cmap + offset;
        }
    }

    info->loca_format = (i16)READ_BE16(file.str + info->head + 50);
}

u32 font_glyph_index(string8 file, font_info* info, u32 codepoint) {
    u8* subtable = file.str + info->cmap_subtable;

    u32 num_groups = READ_BE32(subtable + 12);
    for (u32 i = 0; i < num_groups; i++) {
        u32 group_offset = 16 + i * 12;

        u32 start_code = READ_BE32(subtable + group_offset + 0);
        u32 end_code = READ_BE32(subtable + group_offset + 4);
        u32 index_offset = READ_BE32(subtable + group_offset + 8);

        if (start_code <= codepoint && codepoint <= end_code) {
            return codepoint - start_code + index_offset;
        }
    }

    return 0;
}

void _font_glyph_init(font_glyph* glyph) {
    memset(glyph, 0, sizeof(font_glyph));

    glyph->capacity = 8;
    glyph->flags = malloc(sizeof(font_point_flag) * glyph->capacity);
    glyph->points = malloc(sizeof(v2_i16) * glyph->capacity);
}

void _font_glyph_append(font_glyph* glyph, font_point_flag flag, v2_i16 point) {
    if (glyph->num_points >= glyph->capacity) {
        glyph->capacity *= 2;

        glyph->flags = realloc(glyph->flags, sizeof(font_point_flag) * glyph->capacity);
        glyph->points = realloc(glyph->points, sizeof(v2_i16) * glyph->capacity);
    }

    glyph->num_points++;

    glyph->flags[glyph->num_points-1] = flag;
    glyph->points[glyph->num_points-1] = point;
}

void font_load_glyph(string8 file, font_info* info, font_glyph* glyph, u32 glyph_index) {
    _font_glyph_init(glyph);

    u32 offset = info->loca_format == 0 ?
        (u32)READ_BE16(file.str + info->loca + glyph_index * 2) * 2 :
        READ_BE32(file.str + info->loca + glyph_index * 4);

    u8* glyf = file.str + info->glyf + offset;

    i16 num_contours = READ_BE16(glyf);

    if (num_contours < 0) {
        return;
    }

    glyph->x_min = READ_BE16(glyf + 2);
    glyph->y_min = READ_BE16(glyf + 4);
    glyph->x_max = READ_BE16(glyf + 6);
    glyph->y_max = READ_BE16(glyf + 8);

    u32 num_raw_points = READ_BE16(glyf + 10 + 2 * (num_contours - 1)) + 1;

    u8* flags_raw = malloc(num_raw_points);
    v2_i16* points_raw = malloc(sizeof(v2_i16) * num_raw_points);

    u16 num_instructions = READ_BE16(glyf + 10 + 2 * num_contours);
    u8* point_data = glyf + 12 + 2 * num_contours + num_instructions;

    u32 num_flags = 0;
    while (num_flags < num_raw_points) {
        u8 flag = *(point_data++);

        flags_raw[num_flags++] = flag;

        if (flag & 0x08) {
            u8 count = *(point_data++);

            while (count--) {
                flags_raw[num_flags++] = flag;
            }
        }
    }

    i16 x = 0;
    for (u32 i = 0; i < num_raw_points; i++) {
        if (flags_raw[i] & 0x02) {
            // 1 byte x
            u8 diff = *(point_data++);
            x += (flags_raw[i] & 0x10) ? diff : -(i16)diff;
        } else if ((flags_raw[i] & 0x10) != 0x10) {
            i16 diff = (i16)READ_BE16(point_data);
            point_data += 2;
            x += diff;
        }

        points_raw[i].x = x;
    }

    i16 y = 0;
    for (u32 i = 0; i < num_raw_points; i++) {
        if (flags_raw[i] & 0x04) {
            // 1 byte x
            u8 diff = *(point_data++);
            y += (flags_raw[i] & 0x20) ? diff : -(i16)diff;
        } else if ((flags_raw[i] & 0x20) != 0x20) {
            i16 diff = (i16)READ_BE16(point_data);
            point_data += 2;
            y += diff;
        }

        points_raw[i].y = y;
    }

    for (i32 c = 0; c < num_contours; c++) {
        u32 start_point = c == 0 ? 0 : READ_BE16(glyf + 10 + 2 * (c - 1)) + 1;
        u32 end_point = READ_BE16(glyf + 10 + 2 * c);
        i32 num_points = (i32)end_point - start_point + 1;

        i32 point_offset = 0;
        for (i32 i = 0; i < num_points; i++) {
            u32 p0_i = ((i + point_offset + 0) % num_points) + start_point;
            u32 p1_i = ((i + point_offset + 1) % num_points) + start_point;
            u32 p2_i = ((i + point_offset + 2) % num_points) + start_point;

            u32 on_curve = (
                ((flags_raw[p0_i] & 0x1) << 2) |
                ((flags_raw[p1_i] & 0x1) << 1) |
                ((flags_raw[p2_i] & 0x1) << 0)
            );

            v2_i16 p0 = points_raw[p0_i];
            v2_i16 p1 = points_raw[p1_i];
            v2_i16 p2 = points_raw[p2_i];

            font_point_flag f0 = FONT_POINT_FLAG_ON_CURVE;
            font_point_flag f1 = 0;
            font_point_flag f2 = FONT_POINT_FLAG_ON_CURVE;

            b32 bezier = true;

            switch (on_curve) {
                case 0b110:
                case 0b111: {
                    // Line segment
                    p2 = p1;
                    bezier = false;
                } break;

                case 0b101: {
                    // Bezier (all points explicit)
                    // Skip off curve point
                    i++;
                } break;

                case 0b100: {
                    // Bezier (first two explicit, last implicit)
                    p2 = (v2_i16){ 
                        (p1.x + p2.x) / 2,
                        (p1.y + p2.y) / 2,
                    };

                    f2 |= FONT_POINT_FLAG_GENERATED;
                } break;

                case 0b001: {
                    // Bezier (first implicit, last two explicit)
                    p0 = (v2_i16) {
                        (p0.x + p1.x) / 2,
                        (p0.y + p1.y) / 2,
                    };

                    f0 |= FONT_POINT_FLAG_GENERATED;

                    // Skip off curve point
                    i++;
                } break;

                case 0b000: {
                    // Bezier (middle explicit, the rest implicit)
                    p0 = (v2_i16) {
                        (p0.x + p1.x) / 2,
                        (p0.y + p1.y) / 2,
                    };
                    p2 = (v2_i16){ 
                        (p1.x + p2.x) / 2,
                        (p1.y + p2.y) / 2,
                    };

                    f0 |= FONT_POINT_FLAG_GENERATED;
                    f2 |= FONT_POINT_FLAG_GENERATED;
                } break;

                case 0b010:
                case 0b011: {
                    // Impossible case
                    point_offset++;
                    i--;
                } break;
            }

            glyph->num_segments++;

            _font_glyph_append(glyph, f0, p0);
            if (bezier) {
                _font_glyph_append(glyph, f1, p1);
            }

            if (i == num_points - 1) {
                f2 |= FONT_POINT_FLAG_CONTOUR_END;
                _font_glyph_append(glyph, f2, p2);
            }
        }
    }

    free(points_raw);
    free(flags_raw);
}

void font_free_glyph(font_glyph* glyph) {
    free(glyph->flags);
    free(glyph->points);

    memset(glyph, 0, sizeof(font_glyph));
}

f32 font_scale_for_em(string8 file, font_info* info, f32 pixels_per_em) {
    f32 units_per_em = READ_BE16(file.str + info->head + 18);
    return pixels_per_em / units_per_em;
}

bitmap_r8 font_raster_glyph(font_glyph* glyph, f32 scale, u32 padding) {
    f32 x_min_scaled = (f32)glyph->x_min * scale;
    f32 y_min_scaled = (f32)glyph->y_min * scale;
    f32 x_max_scaled = (f32)glyph->x_max * scale;
    f32 y_max_scaled = (f32)glyph->y_max * scale;

    bitmap_r8 bitmap = {
        .width = (u32)ceilf(x_max_scaled - x_min_scaled) + padding * 2,
        .height = (u32)ceilf(y_max_scaled - y_min_scaled) + padding * 2,
    };

    bitmap.data = malloc(bitmap.width * bitmap.height);
    memset(bitmap.data, 0, bitmap.width * bitmap.height);

    v2_f32* points = malloc(sizeof(v2_f32) * glyph->num_points);
    for (u32 i = 0; i < glyph->num_points; i++) {
        points[i] = (v2_f32){
            (f32)(glyph->points[i].x - glyph->x_min) * scale + padding,
            (f32)(glyph->points[i].y - glyph->y_max) * (-scale) + padding,
        };
    }

    f32* intersects = malloc(sizeof(f32) * glyph->num_segments * 2);

    for (u32 y_i = 0; y_i < bitmap.height; y_i++) {
        f32 y = (f32)y_i + 0.5f;

        u32 num_intersects = 0;
        u32 index = 0;
        for (u32 seg = 0; seg < glyph->num_segments; seg++) {
            f32 intersect0 = -1.0f, intersect1 = -1.0f;

            v2_f32 p0 = points[index++];
            v2_f32 p1 = points[index];

            if (glyph->flags[index] & FONT_POINT_FLAG_ON_CURVE) {
                // Line segment

                f32 min_y = MIN(p0.y, p1.y);
                f32 max_y = MAX(p0.y, p1.y);

                if (min_y <= y && y <= max_y) {
                    // y = p0.y + (p1.y - p0.y) * t;
                    // (y - p0.y) / (p1.y - p0.y) = t;
                    f32 t = (y - p0.y) / (p1.y - p0.y);
                    if (0.0f <= t && t <= 1.0f) {
                        intersect0 = p0.x + (p1.x - p0.x) * t;
                    }
                }
            } else {
                // Bezier
                v2_f32 p2 = points[++index];

                f32 min_y = MIN(p0.y, MIN(p1.y, p2.y));
                f32 max_y = MAX(p0.y, MAX(p1.y, p2.y));

                if (min_y <= y && y <= max_y) {
                    f32 a_y = p2.y - 2.0f * p1.y + p0.y;
                    f32 b_y = 2.0f * (p1.y - p0.y);
                    f32 c_y = p0.y - y;

                    f32 a_x = p2.x - 2.0f * p1.x + p0.x;
                    f32 b_x = 2.0f * (p1.x - p0.x);
                    f32 c_x = p0.x;

                    if (fabsf(a_y) <= 1e-6f) {
                        // Basically a line
                        // 0 = b_y * t + c_y
                        // -c_y / b_y = t

                        f32 t = -c_y / b_y;
                        if (0.0f <= t && t <= 1.0f) {
                            intersect0 = t * (t * a_x + b_x) + c_x;
                        }
                    } else {
                        f32 discr = b_y * b_y - 4.0f * a_y * c_y;

                        if (fabsf(discr) <= 1e-6f) {
                            f32 t = -b_y / (2.0f * a_y);

                            if (0.0f <= t && t <= 1.0f) {
                                intersect0 = t * (t * a_x + b_x) + c_x;
                            }
                        } else if (discr > 0.0f) {
                            f32 sqrt_discr = sqrtf(discr);
                            f32 t0 = (-b_y + sqrt_discr) / (2.0f * a_y);
                            f32 t1 = (-b_y - sqrt_discr) / (2.0f * a_y);

                            if (0.0f <= t0 && t0 <= 1.0f) {
                                intersect0 = t0 * (t0 * a_x + b_x) + c_x;
                            }
                            if (0.0f <= t1 && t1 <= 1.0f) {
                                intersect1 = t1 * (t1 * a_x + b_x) + c_x;
                            }
                        }
                    }
                }
            }

            if (glyph->flags[index] & FONT_POINT_FLAG_CONTOUR_END) {
                index++;
            }

            if (0.0f <= intersect0 && intersect0 < bitmap.width) {
                intersects[num_intersects++] = intersect0;
            }
            if (0.0f <= intersect1 && intersect1 < bitmap.width) {
                intersects[num_intersects++] = intersect1;
            }
        }

        for (u32 i = 1; i < num_intersects; i++) {
            for (i32 j = i; j >= 1 && intersects[j] < intersects[j-1]; j--) {
                f32 tmp = intersects[j-1];
                intersects[j-1] = intersects[j];
                intersects[j] = tmp;
            }
        }

        /*printf("%4.1f | ", y);
        for (u32 i = 0; i < num_intersects; i++) {
            printf("%5.2f ", intersects[i]);
        }
        printf("\n");*/

        u8* row = bitmap.data + y_i * bitmap.width;
        for (i32 i = 0; i < (i32)num_intersects - 1; i += 2) {
            f32 start = intersects[i];
            f32 end = intersects[i + 1];

            memset(row + (u32)start, 255, (u32)(end - start + 1));

            row[(u32)start] = (u8)(255 - 255 * (start - floorf(start)));
            row[(u32)end] = (u8)(255 - 255 * (end - floorf(end)));
        }
    }

    free(intersects);
    free(points);

    return bitmap;
}

string8 read_file(const char* file_name) {
    FILE* f = fopen(file_name, "rb");

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    string8 out = {
        .size = size,
        .str = malloc(size)
    };

    fread(out.str, 1, out.size, f);

    fclose(f);

    return out;
}

