/*
 * dump_rime_gram: list every (key, value) pair of a RIME octagram grammar
 * file (`*.gram`, format "Rime::Grammar/1.0") as UTF-8 TSV:
 *
 *     key <TAB> value
 *
 * Used by build_pinyin_ngrams_v2.py to read the 万象 (RIME-LMDG) model.
 *
 * File layout (librime-octagram src/gram_db.h, BSD-3-Clause):
 *   0   char[32]  format, "Rime::Grammar/1.0"
 *   32  u32       db_checksum
 *   36  u32       double_array_size   (units)
 *   40  i32       OffsetPtr: the image starts at 40 + this value
 *   then a darts-clone double array, little-endian u32 units.
 *
 * darts-clone unit (BSD-2-Clause):
 *   has_leaf = (u >> 8) & 1
 *   value    = u & 0x7FFFFFFF               (on the leaf unit)
 *   label    = u & (0x80000000 | 0xFF)
 *   offset   = (u >> 10) << ((u & (1 << 9)) >> 6)
 *   child of node n on byte c: n ^ offset(unit[n]) ^ c, valid if its label == c;
 *   a node's value sits at n ^ offset(unit[n]) when has_leaf.
 *
 * Key encoding (librime-octagram src/gram_encoding.cc), decoded here:
 *   byte < 0x80                   ASCII
 *   0x80..0xDF, lo                U+4000..U+9FFF: ((b - 0x40) << 8) | lo
 *   0xE1, b                       same with lo == 0
 *   0xE0|n (n >= 2), n bytes      7 bits each, big-endian, code point = bits >> 3
 * Values are max(0, int(ln(count) * 10000)).
 *
 * Build:  clang -O2 -o dump_rime_gram dump_rime_gram.c -lpthread
 * Usage:  dump_rime_gram FILE.gram OUT_PREFIX [THREADS]
 *         writes OUT_PREFIX.0 .. OUT_PREFIX.(THREADS-1); concatenate them.
 */
#include <fcntl.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

static const uint32_t *units;
static uint32_t n_units;
static int n_threads;
static const char *out_prefix;

#define MAX_KEY 256

static inline uint32_t offset_of(uint32_t u) { return (u >> 10) << ((u & (1u << 9)) >> 6); }
static inline uint32_t label_of(uint32_t u) { return u & ((1u << 31) | 0xFFu); }
static inline int has_leaf(uint32_t u) { return (u >> 8) & 1; }

static int put_utf8(uint32_t cp, char *o) {
    if (cp < 0x80) { o[0] = (char)cp; return 1; }
    if (cp < 0x800) { o[0] = (char)(0xC0 | (cp >> 6)); o[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) {
        o[0] = (char)(0xE0 | (cp >> 12)); o[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
        o[2] = (char)(0x80 | (cp & 0x3F)); return 3;
    }
    o[0] = (char)(0xF0 | (cp >> 18)); o[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    o[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); o[3] = (char)(0x80 | (cp & 0x3F)); return 4;
}

/* Returns the UTF-8 length written, or -1 on a malformed key. */
static int decode(const unsigned char *k, int len, char *out) {
    int i = 0, o = 0;
    while (i < len) {
        unsigned b = k[i];
        uint32_t cp;
        if (b < 0x80) { cp = b; i += 1; }
        else if (b < 0xE0) { if (i + 1 >= len) return -1; cp = ((b - 0x40) << 8) | k[i + 1]; i += 2; }
        else if (b == 0xE1) { if (i + 1 >= len) return -1; cp = (uint32_t)(k[i + 1] - 0x40) << 8; i += 2; }
        else if (b == 0xE0) { cp = 0; i += 1; }
        else {
            int n = b & 0x0F;
            if (i + n >= len) return -1;
            uint64_t acc = 0;
            for (int j = 1; j <= n; j++) acc = (acc << 7) | (k[i + j] & 0x7F);
            cp = (uint32_t)(acc >> 3);
            i += n + 1;
        }
        o += put_utf8(cp, out + o);
    }
    return o;
}

struct job { int id; FILE *f; unsigned long long keys, bad; };

static void dfs(uint32_t node, unsigned char *key, int depth, struct job *jb) {
    uint32_t u = units[node];
    uint32_t base = node ^ offset_of(u);
    if (has_leaf(u)) {
        uint32_t value = units[base] & 0x7FFFFFFFu;
        char buf[MAX_KEY * 2];
        int n = decode(key, depth, buf);
        if (n < 0) { jb->bad++; }
        else { fwrite(buf, 1, (size_t)n, jb->f); fprintf(jb->f, "\t%u\n", value); jb->keys++; }
    }
    if (depth >= MAX_KEY - 1) return;
    for (uint32_t c = 1; c < 256; c++) {
        uint32_t child = base ^ c;
        if (child >= n_units) continue;
        if (label_of(units[child]) != c) continue;
        key[depth] = (unsigned char)c;
        dfs(child, key, depth + 1, jb);
    }
}

static void *run(void *arg) {
    struct job *jb = arg;
    char path[4096];
    snprintf(path, sizeof path, "%s.%d", out_prefix, jb->id);
    jb->f = fopen(path, "w");
    if (!jb->f) { perror(path); exit(1); }
    setvbuf(jb->f, NULL, _IOFBF, 1 << 20);
    unsigned char key[MAX_KEY];
    uint32_t root = 0, base = offset_of(units[root]);
    /* Root children split round-robin by first byte. */
    for (uint32_t c = 1; c < 256; c++) {
        if ((int)(c % (uint32_t)n_threads) != jb->id) continue;
        uint32_t child = base ^ c;
        if (child >= n_units || label_of(units[child]) != c) continue;
        key[0] = (unsigned char)c;
        dfs(child, key, 1, jb);
    }
    fclose(jb->f);
    return NULL;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s FILE.gram OUT_PREFIX [THREADS]\n", argv[0]); return 2; }
    out_prefix = argv[2];
    n_threads = argc > 3 ? atoi(argv[3]) : 8;
    if (n_threads < 1) n_threads = 1;
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror(argv[1]); return 1; }
    struct stat st;
    fstat(fd, &st);
    const unsigned char *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 1; }
    if (st.st_size < 44 || memcmp(map, "Rime::Grammar/", 14) != 0) { fprintf(stderr, "not a Rime grammar\n"); return 1; }
    uint32_t size; int32_t off;
    memcpy(&size, map + 36, 4);
    memcpy(&off, map + 40, 4);
    size_t start = 40 + (size_t)off;
    if (start + (size_t)size * 4 > (size_t)st.st_size) { fprintf(stderr, "image past end of file\n"); return 1; }
    units = (const uint32_t *)(map + start);
    n_units = size;
    fprintf(stderr, "format %.32s, %u units at byte %zu\n", (const char *)map, size, start);

    pthread_t th[64];
    struct job jobs[64];
    if (n_threads > 64) n_threads = 64;
    for (int i = 0; i < n_threads; i++) { memset(&jobs[i], 0, sizeof jobs[i]); jobs[i].id = i; pthread_create(&th[i], NULL, run, &jobs[i]); }
    unsigned long long keys = 0, bad = 0;
    for (int i = 0; i < n_threads; i++) { pthread_join(th[i], NULL); keys += jobs[i].keys; bad += jobs[i].bad; }
    fprintf(stderr, "%llu keys, %llu malformed\n", keys, bad);
    return 0;
}
