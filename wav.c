#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef int16_t i16;
typedef uint16_t u16;
typedef uint32_t u32;

typedef float f32;

void write_le_16(FILE* f, u16 n) {
    fwrite(&n, sizeof(u16), 1, f);
}

void write_le_32(FILE* f, u32 n) {
    fwrite(&n, sizeof(u32), 1, f);
}

#define WRITE_STR_LIT(f, s) fwrite((s), 1, sizeof(s) - 1, f)

#define FREQ 44100

int main(void) {
    FILE* f = fopen("test.wav", "wb");

    struct {
        f32 freq;
        f32 dur;
    } notes[] = {
        { 392, 60.0f/ 76 },
        { 440, 60.0f/ 76 },
        { 294, 60.0f/114 },
        { 440, 60.0f/ 76 },
        { 494, 60.0f/ 76 },
    };

    u32 num_notes = sizeof(notes) / sizeof(notes[0]);

    f32 duration = 0.0f;
    for (u32 i = 0; i < num_notes; i++) {
        duration += notes[i].dur;
    }

    u32 num_samples = (u32)(duration * FREQ);
    u32 file_size = num_samples * sizeof(u16) + 44;

    WRITE_STR_LIT(f, "RIFF");
    write_le_32(f, file_size - 8);
    WRITE_STR_LIT(f, "WAVE");

    WRITE_STR_LIT(f, "fmt ");
    write_le_32(f, 16);
    write_le_16(f, 1);
    write_le_16(f, 1);
    write_le_32(f, FREQ);
    write_le_32(f, FREQ * sizeof(u16));
    write_le_16(f, sizeof(u16));
    write_le_16(f, sizeof(u16) * 8);

    WRITE_STR_LIT(f, "data");
    write_le_32(f, num_samples * sizeof(u16));

    u32 cur_note = 0;
    f32 cur_note_start = 0.0f;
    for (u32 i = 0; i < num_samples; i++) {
        f32 t = (f32)i / FREQ;

        f32 y = 0.0f;

        if (cur_note < num_notes) {
            y = 0.25f * sinf(t * notes[cur_note].freq * 2.0f * 3.1415926535f);

            if (t > cur_note_start + notes[cur_note].dur) {
                cur_note++;
                cur_note_start = t;
            }
        }

        i16 sample = (i16)(y * INT16_MAX);

        write_le_16(f, sample);
    }

    fclose(f);

    return 0;
}


