#include "ed_internal.h"
#include <string.h>

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    uint8_t block[64];
    size_t used;
} ed_sha_ctx;

static uint32_t rotr(uint32_t x, unsigned n) { return (x >> n) | (x << (32u - n)); }

static void sha_block(ed_sha_ctx *c, const uint8_t *b) {
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u
    };
    uint32_t w[64], a, d, e, f, g, h, q, t1, t2;
    unsigned i;
    for (i = 0; i < 16u; ++i)
        w[i] = ((uint32_t)b[4u*i] << 24) | ((uint32_t)b[4u*i+1u] << 16) |
               ((uint32_t)b[4u*i+2u] << 8) | (uint32_t)b[4u*i+3u];
    for (; i < 64u; ++i) {
        uint32_t s0 = rotr(w[i-15u],7)^rotr(w[i-15u],18)^(w[i-15u]>>3);
        uint32_t s1 = rotr(w[i-2u],17)^rotr(w[i-2u],19)^(w[i-2u]>>10);
        w[i] = w[i-16u] + s0 + w[i-7u] + s1;
    }
    a=c->h[0]; q=c->h[1]; d=c->h[2]; e=c->h[3]; f=c->h[4]; g=c->h[5]; h=c->h[6]; t2=c->h[7];
    for (i = 0; i < 64u; ++i) {
        uint32_t s1=rotr(f,6)^rotr(f,11)^rotr(f,25), ch=(f&g)^((~f)&h);
        t1=t2+s1+ch+k[i]+w[i];
        { uint32_t s0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&q)^(a&d)^(q&d); uint32_t z=s0+maj;
          t2=h; h=g; g=f; f=e+t1; e=d; d=q; q=a; a=t1+z; }
    }
    c->h[0]+=a; c->h[1]+=q; c->h[2]+=d; c->h[3]+=e;
    c->h[4]+=f; c->h[5]+=g; c->h[6]+=h; c->h[7]+=t2;
}

static void sha_update(ed_sha_ctx *c, const uint8_t *p, size_t n) {
    c->bits += (uint64_t)n * 8u;
    while (n) {
        size_t take = 64u - c->used;
        if (take > n) take = n;
        memcpy(c->block + c->used, p, take);
        c->used += take; p += take; n -= take;
        if (c->used == 64u) { sha_block(c, c->block); c->used = 0u; }
    }
}

void ed_sha256(const void *data, size_t size, uint8_t digest[32]) {
    ed_sha_ctx c = {{0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,
                     0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u},0,{0},0};
    uint64_t bits;
    unsigned i;
    sha_update(&c, (const uint8_t *)data, size);
    bits = c.bits;
    c.block[c.used++] = 0x80u;
    if (c.used > 56u) { while (c.used < 64u) c.block[c.used++]=0; sha_block(&c,c.block); c.used=0; }
    while (c.used < 56u) c.block[c.used++]=0;
    for (i=0;i<8u;++i) c.block[63u-i]=(uint8_t)(bits>>(8u*i));
    sha_block(&c,c.block);
    for (i=0;i<8u;++i) {
        digest[4u*i]=(uint8_t)(c.h[i]>>24); digest[4u*i+1u]=(uint8_t)(c.h[i]>>16);
        digest[4u*i+2u]=(uint8_t)(c.h[i]>>8); digest[4u*i+3u]=(uint8_t)c.h[i];
    }
}

int ed_digest_equal(const uint8_t a[32], const uint8_t b[32]) {
    uint8_t d=0; unsigned i; for(i=0;i<32u;++i)d|=(uint8_t)(a[i]^b[i]); return d==0;
}
