
#define _CRT_SECURE_NO_WARNINGS

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include <stddef.h>
#include <stdlib.h>

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

typedef struct {
    u8* str;
    u64 size;
} string8;

string8 read_file(const char* path);
string8 gunzip(string8 input);
void inflate(string8 input, string8 out);

int main(void) {
    string8 test0_gz = read_file("test0.gz");
    string8 test1_gz = read_file("test1.gz");

    string8 test0 = gunzip(test0_gz);
    string8 test1 = gunzip(test1_gz);

    printf("%.*s\n\n", (int)test0.size, test0.str);
    printf("%.*s\n\n", (int)test1.size, test1.str);

    return 0;
}

string8 read_file(const char* path) {
    string8 out = { 0 };

    FILE* f = fopen(path, "rb");

    fseek(f, 0, SEEK_END);
    u64 size = ftell(f);
    fseek(f, 0, SEEK_SET);

    out.size = size;
    out.str = malloc(size);

    fread(out.str, 1, out.size, f);

    fclose(f);

    return out;
}

#define FTEXT    0b00001
#define FHCRC    0b00010
#define FEXTRA   0b00100
#define FNAME    0b01000
#define FCOMMENT 0b10000
 
string8 gunzip(string8 input) {
    u8 flags = input.str[3];

    u32 block_offset = 10;

    if (flags & FEXTRA) {
        u16 xlen = *(u16*)(input.str + 10);
        block_offset += 2 + xlen;
    }

    if (flags & FNAME) { while (input.str[block_offset++]); }
    if (flags & FCOMMENT) { while (input.str[block_offset++]); }
    if (flags & FHCRC) { block_offset += 2; }

    u32 isize = *(u32*)(input.str + input.size - 4);

    string8 out = {
        .size = isize,
        .str = malloc(isize)
    };

    string8 deflate_block = {
        .size = input.size - block_offset - 8,
        .str = input.str + block_offset,
    };

    inflate(deflate_block, out);

    return out;
}

u16 reverse_u16(u16 n) {
    u16 o = n;
    o = ((o & 0xAAAA) >> 1) | ((o & 0x5555) << 1);
    o = ((o & 0xCCCC) >> 2) | ((o & 0x3333) << 2);
    o = ((o & 0xF0F0) >> 4) | ((o & 0x0F0F) << 4);
    o = ((o & 0xFF00) >> 8) | ((o & 0x00FF) << 8);
    return o;
}

#define reverse_bits(n, bits) (reverse_u16(n) >> (16 - bits))

typedef struct {
    string8 backing;
    u64 bit_pos;
    u64 bit_size;
} bitstream;

bitstream bs_init(string8 backing) {
    return (bitstream){ 
        .backing = backing,
        .bit_pos = 0,
        .bit_size = backing.size * 8
    };
}

u32 bit_masks[] = {
    0b0000000000000000,
    0b0000000000000001,
    0b0000000000000011,
    0b0000000000000111,
    0b0000000000001111,
    0b0000000000011111,
    0b0000000000111111,
    0b0000000001111111,
    0b0000000011111111,
    0b0000000111111111,
    0b0000001111111111,
    0b0000011111111111,
    0b0000111111111111,
    0b0001111111111111,
    0b0011111111111111,
    0b0111111111111111,
    0b1111111111111111,
};

// nbits <= 16
u32 bs_peek(bitstream* bs, u8 nbits) {
    u64 byte_pos = bs->bit_pos / 8;
    u32 bits = *(u32*)(bs->backing.str + byte_pos);

    bits >>= bs->bit_pos % 8;

    return bits & bit_masks[nbits];
}

// nbits <= 16
u32 bs_take(bitstream* bs, u8 nbits) {
    u64 byte_pos = bs->bit_pos / 8;
    u32 bits = *(u32*)(bs->backing.str + byte_pos);

    bits >>= bs->bit_pos % 8;
    bs->bit_pos += nbits;

    return bits & bit_masks[nbits];
}

typedef struct {
    u16 len;
    u16 code;
} generic_huff_node;

#define MAX_BITS 15

void build_huff_tree(generic_huff_node* nodes, u32 node_count) {
    u32 bl_count[MAX_BITS] = { 0 };

    for (u32 i = 0; i < node_count; i++) {
        bl_count[nodes[i].len]++;
    }

    u32 next_code[MAX_BITS + 1] = { 0 };

    u32 code = 0;
    bl_count[0] = 0;
    for (u32 bits = 1; bits <= MAX_BITS; bits++) {
        code = (code + bl_count[bits-1]) << 1;
        next_code[bits] = code;
    }

    for (u32 i = 0; i < node_count; i++) {
        if (nodes[i].len != 0) {
            nodes[i].code = next_code[nodes[i].len]++;
        }
    }
}

typedef struct {
    u16 code;
    u8 bits_used;
    u8 extra_bits;
    u16 length_base;
} ll_lut_node;

typedef struct {
    u8 bits_used;
    u8 extra_bits;
    u16 dist_base;
} dist_lut_node;

u8 ll_extras[] = {
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5,
    5, 5, 5, 0,
};

u16 ll_bases[] = {
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67,
    83, 99, 115, 131, 163, 195, 227, 258,
};

u8 dist_extras[] = {
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10,
    11, 11, 12, 12, 13, 13,
};

u16 dist_bases[] = {
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513,
    769, 1025, 1537, 2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
};

u64 _huff_impl(
    bitstream* bs,
    u8* out,
    u64 pos,
    generic_huff_node* ll_nodes,
    generic_huff_node* dist_nodes
) {
    u32 max_ll_bits = 0;
    u32 max_dist_bits = 0;

    for (u32 i = 0; i < 286; i++) {
        if (ll_nodes[i].len > max_ll_bits) {
            max_ll_bits = ll_nodes[i].len;
        }
    }

    for (u32 i = 0; i < 30; i++) {
        if (dist_nodes[i].len > max_dist_bits) {
            max_dist_bits = dist_nodes[i].len;
        }
    }

    u32 ll_lut_size = sizeof(ll_lut_node) * (1 << max_ll_bits);
    u32 dist_lut_size = sizeof(dist_lut_node) * (1 << max_dist_bits);

    ll_lut_node* ll_lut = malloc(ll_lut_size);
    dist_lut_node* dist_lut = malloc(dist_lut_size);

    memset(ll_lut, 0, ll_lut_size);
    memset(dist_lut, 0, dist_lut_size);

    for (u32 i = 0; i < 286; i++) {
        u32 len = ll_nodes[i].len;
        u32 code = ll_nodes[i].code;

        if (len == 0) { continue; }

        ll_lut_node lut_node = {
            .code = i,
            .bits_used = len,
            .extra_bits = i < 257 ? 0 : ll_extras[i-257],
            .length_base = i < 257 ? 0 : ll_bases[i-257],
        };

        u32 leftover_bits = max_ll_bits - len;
        u32 leftover_max = (1 << leftover_bits) - 1;
        for (u32 leftover = 0; leftover <= leftover_max; leftover++) {
            u32 index = reverse_bits(
                (code << leftover_bits) | leftover,
                max_ll_bits
            );

            ll_lut[index] = lut_node;
        }
    }

    for (u32 i = 0; i < 30; i++) {
        u32 len = dist_nodes[i].len;
        u32 code = dist_nodes[i].code;

        if (len == 0) { continue; }

        dist_lut_node lut_node = {
            .bits_used = len,
            .extra_bits = dist_extras[i],
            .dist_base = dist_bases[i],
        };

        u32 leftover_bits = max_dist_bits - len;
        u32 leftover_max = (1 << leftover_bits) - 1;
        for (u32 leftover = 0; leftover <= leftover_max; leftover++) {
            u32 index = reverse_bits(
                (code << leftover_bits) | leftover,
                max_dist_bits
            );

            dist_lut[index] = lut_node;
        }
    }

    while (bs->bit_pos < bs->bit_size) {
        u32 ll_peek_bits = bs_peek(bs, max_ll_bits);
        ll_lut_node ll_node = ll_lut[ll_peek_bits];
        bs->bit_pos += ll_node.bits_used;

        u32 code = ll_node.code;

        if (code < 256) {
            out[pos++] = code;
        } else if (code == 256) {
            break;
        } else {
            u32 length = ll_node.length_base + bs_take(bs, ll_node.extra_bits);

            u32 dist_peek_bits = bs_peek(bs, max_dist_bits);
            dist_lut_node dist_node = dist_lut[dist_peek_bits];
            bs->bit_pos += dist_node.bits_used;

            u32 dist = dist_node.dist_base + bs_take(bs, dist_node.extra_bits);

            while (length--) {
                out[pos] = out[pos - dist];
                pos++;
            }
        }
    }

    free(ll_lut);
    free(dist_lut);

    return pos;
}

typedef struct {
    u8 code_len;
    u8 bits_used;
    u8 extra_bits;
    u8 repeat_base;
} cl_lut_node;

u8 cl_extras[19] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 7
};

u8 cl_bases[19] = {
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3, 11
};

u8 cl_order[] = {
    16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15
};

u8 _dyn_next_cl(
    bitstream* bs,
    generic_huff_node* nodes,
    u32* node_pos,
    cl_lut_node* cl_lut,
    u8 prev_codelen
) {
    u32 peek_bits = bs_peek(bs, 7);
    cl_lut_node cl_node = cl_lut[peek_bits];
    bs->bit_pos += cl_node.bits_used;

    u32 cl = cl_node.code_len;

    if (cl <= 15) {
        nodes[(*node_pos)++].len = cl;
        return cl;
    }

    u32 repeat_value = cl == 16 ? prev_codelen : 0;
    u32 repeat_count = cl_node.repeat_base + bs_take(bs, cl_node.extra_bits);

    while (repeat_count--) {
        nodes[(*node_pos)++].len = repeat_value;
    }

    return repeat_value;
}

u64 _dynamic_block(bitstream* bs, u8* out, u64 pos) {
    u32 num_ll_codes = bs_take(bs, 5) + 257;
    u32 num_dist_codes = bs_take(bs, 5) + 1;
    u32 num_cl_codes = bs_take(bs, 4) + 4;

    generic_huff_node cl_nodes[19] = { 0 };

    for (u32 i = 0; i < num_cl_codes; i++) {
        cl_nodes[cl_order[i]].len = bs_take(bs, 3);
    }

    build_huff_tree(cl_nodes, 19);

    cl_lut_node cl_lut[256] = { 0 };
    for (u32 i = 0; i < 19; i++) {
        u32 len = cl_nodes[i].len;
        u32 code = cl_nodes[i].code;

        if (len == 0) { continue; }

        cl_lut_node lut_node = {
            .code_len = i,
            .bits_used = len,
            .extra_bits = cl_extras[i],
            .repeat_base = cl_bases[i],
        };

        u32 leftover_bits = 7 - len;
        u32 leftover_max = (1 << leftover_bits) - 1;
        for (u32 leftover = 0; leftover <= leftover_max; leftover++) {
            u32 index = reverse_bits(
                (code << leftover_bits) | leftover,
                7
            );

            cl_lut[index] = lut_node;
        }
    }

    // Note(Ian) I have updated this logic from the video to be more spec
    // compliant
    generic_huff_node combined_nodes[286 + 30] = { 0 };

    u8 prev_codelen = 0;
    u32 node_pos = 0;
    while (node_pos < num_ll_codes + num_dist_codes) {
        prev_codelen = _dyn_next_cl(
            bs, combined_nodes, &node_pos,
            cl_lut, prev_codelen
        );
    }

    generic_huff_node ll_nodes[286] = { 0 };
    generic_huff_node dist_nodes[30] = { 0 };

    memcpy(ll_nodes, combined_nodes, num_ll_codes * sizeof(generic_huff_node));
    memcpy(
        dist_nodes,
        combined_nodes + num_ll_codes,
        num_dist_codes * sizeof(generic_huff_node)
    );

    build_huff_tree(ll_nodes, 286);
    build_huff_tree(dist_nodes, 30);

    return _huff_impl(bs, out, pos, ll_nodes, dist_nodes);
}

u64 _fixed_block(bitstream* bs, u8* out, u64 pos) {
    generic_huff_node ll_nodes[288] = { 0 };
    generic_huff_node dist_nodes[32] = { 0 };

    for (u32 i = 0; i <= 143; i++)   { ll_nodes[i].len = 8; }
    for (u32 i = 144; i <= 255; i++) { ll_nodes[i].len = 9; }
    for (u32 i = 256; i <= 279; i++) { ll_nodes[i].len = 7; }
    for (u32 i = 280; i <= 287; i++) { ll_nodes[i].len = 8; }

    for (u32 i = 0; i < 32; i++) { dist_nodes[i].len = 5; }

    build_huff_tree(ll_nodes, 288);
    build_huff_tree(dist_nodes, 32);

    return _huff_impl(bs, out, pos, ll_nodes, dist_nodes);
}

void inflate(string8 input, string8 out) {
    bitstream bs = bs_init(input);

    u64 pos = 0;

    b32 last_block = false;
    while (!last_block) {
        last_block = bs_take(&bs, 1);
        u32 block_type = bs_take(&bs, 2);

        switch (block_type) {
            case 0b00: break; // no compression
            case 0b01: {
                pos = _fixed_block(&bs, out.str, pos);
            } break;
            case 0b10: {
                pos = _dynamic_block(&bs, out.str, pos);
            } break;
            case 0b11: break; // reserved
        }
    }
}

