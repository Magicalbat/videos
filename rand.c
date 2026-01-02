#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include <math.h>

typedef uint32_t u32;
typedef uint64_t u64;

typedef float f32;

#define PI 3.14159265358979

// Based on pcg random number generator (https://www.pcg-random.org)
// Licensed under Apache License 2.0 (NO WARRANTY, etc. see website)

typedef struct {
    u64 state;
    u64 inc;

    f32 prev_norm;
} prng_state;

void prng_seed_r(prng_state* rng, u64 initstate, u64 initseq);
void prng_seed(u64 initstate, u64 initseq);

u32 prng_rand_r(prng_state* rng);
u32 prng_rand(void);

f32 prng_randf_r(prng_state* rng);
f32 prng_randf(void);

f32 prng_rand_norm_r(prng_state* rng);
f32 prng_rand_norm(void);

void plat_get_entropy(void* data, u32 size);

int main(void) {
    u64 seeds[2] = { 0 };
    plat_get_entropy(seeds, sizeof(seeds));

    prng_state rng = { };
    prng_seed_r(&rng, seeds[0], seeds[1]);

    for (u32 i = 0; i < 10; i++) {
        printf("%f\n", prng_rand_norm_r(&rng));
    }

    return 0;
}

static prng_state s_prng_state = { 
    0x853c49e6748fea9bULL, 0xda3e39cb94b95bdbULL,
    NAN
};

void prng_seed_r(prng_state* rng, u64 initstate, u64 initseq) {
    rng->state = 0U;
    rng->inc = (initseq << 1u) | 1u;
    prng_rand_r(rng);
    rng->state += initstate;
    prng_rand_r(rng);

    rng->prev_norm = NAN;
}

void prng_seed(u64 initstate, u64 initseq) {
    prng_seed_r(&s_prng_state, initstate, initseq);
}

u32 prng_rand_r(prng_state* rng) {
    u64 oldstate = rng->state;
    rng->state = oldstate * 6364136223846793005ULL + rng->inc;
    u32 xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
    u32 rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

u32 prng_rand(void) {
    return prng_rand_r(&s_prng_state);
}

f32 prng_randf_r(prng_state* rng) {
    return (f32)prng_rand_r(rng) / (f32)UINT32_MAX;
}

f32 prng_randf(void) {
    return prng_randf_r(&s_prng_state);
}

f32 prng_rand_norm_r(prng_state* rng) {
    if (!isnan(rng->prev_norm)) {
        f32 out = rng->prev_norm;
        rng->prev_norm = NAN;
        return out;
    }

    f32 u1 = 0.0f;
    do {
        u1 = prng_randf_r(rng);
    } while (u1 == 0.0f);

    f32 u2 = prng_randf_r(rng);

    f32 mag = sqrtf(-2.0f * logf(u1));

    f32 z0 = mag * cosf(2.0 * PI * u2);
    f32 z1 = mag * sinf(2.0 * PI * u2);

    rng->prev_norm = z1;
    return z0;
}

f32 prng_rand_norm(void) {
    return prng_rand_norm_r(&s_prng_state);
}

#if defined(_WIN32) 

#include <Windows.h>
#include <bcrypt.h>

void plat_get_entropy(void* data, u32 size) {
    BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

#endif

