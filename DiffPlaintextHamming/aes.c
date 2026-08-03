/*
gcc -O3 -march=native -funroll-loops aes.c -o aes

This script:
- Retains identical IV and key values
- Measures hamming distances across bit flips in plaintexts
*/

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/types.h>

/* CONF */
const int NK = 8;      
static int NR = 14; 
const int NUMTRIALS = 2048;
const int SUBTRIALS = 4096;
const int PLAINTEXTSIZE = 512;

uint8_t gmul(uint8_t a, uint8_t b) {
    uint8_t p = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) {
            p ^= a;
        }
        uint8_t hibit = a & 128;
        a <<= 1;
        if (hibit) { 
            a ^= 27;
        }
        b >>= 1;
    }
    return p;
}

static uint8_t SBOX[256] = {
    99,124,119,123,242,107,111,197,48,1,103,43,254,215,171,118,202,130,201,125,
    250,89,71,240,173,212,162,175,156,164,114,192,183,253,147,38,54,63,247,204,
    52,165,229,241,113,216,49,21,4,199,35,195,24,150,5,154,7,18,128,226,235,39,
    178,117,9,131,44,26,27,110,90,160,82,59,214,179,41,227,47,132,83,209,0,237,
    32,252,177,91,106,203,190,57,74,76,88,207,208,239,170,251,67,77,51,133,69,
    249,2,127,80,60,159,168,81,163,64,143,146,157,56,245,188,182,218,33,16,255,
    243,210,205,12,19,236,95,151,68,23,196,167,126,61,100,93,25,115,96,129,79,
    220,34,42,144,136,70,238,184,20,222,94,11,219,224,50,58,10,73,6,36,92,194,
    211,172,98,145,149,228,121,231,200,55,109,141,213,78,169,108,86,244,234,101,
    122,174,8,186,120,37,46,28,166,180,198,232,221,116,31,75,189,139,138,112,
    62,181,102,72,3,246,14,97,53,87,185,134,193,29,158,225,248,152,17,105,217,
    142,148,155,30,135,233,206,85,40,223,140,161,137,13,191,230,66,104,65,153,
    45,15,176,84,187,22
};

static uint8_t INVSBOX[256] = {
    82,9,106,213,48,54,165,56,191,64,163,158,129,243,215,251,
    124,227,57,130,155,47,255,135,52,142,67,68,196,222,233,203,
    84,123,148,50,166,194,35,61,238,76,149,11,66,250,195,78,
    8,46,161,102,40,217,36,178,118,91,162,73,109,139,209,37,
    114,248,246,100,134,104,152,22,212,164,92,204,93,101,182,146,
    108,112,72,80,253,237,185,218,94,21,70,87,167,141,157,132,
    144,216,171,0,140,188,211,10,247,228,88,5,184,179,69,6,
    208,44,30,143,202,63,15,2,193,175,189,3,1,19,138,107,
    58,145,17,65,79,103,220,234,151,242,207,206,240,180,230,115,
    150,172,116,34,231,173,53,133,226,249,55,232,28,117,223,110,
    71,241,26,113,29,41,197,137,111,183,98,14,170,24,190,27,
    252,86,62,75,198,210,121,32,154,219,192,254,120,205,90,244,
    31,221,168,51,136,7,199,49,177,18,16,89,39,128,236,95,
    96,81,127,169,25,181,74,13,45,229,122,159,147,201,156,239,
    160,224,59,77,174,42,245,176,200,235,187,60,131,83,153,97,
    23,43,4,126,186,119,214,38,225,105,20,99,85,33,12,125
};

uint8_t *RCON;

void bytestostate(uint8_t state[4][4], const uint8_t *block) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] = block[i + 4 * j];
        }
    }
}

void statetobytes(uint8_t *out, uint8_t state[4][4]) {
    for (int j = 0; j < 4; j++) {
        for (int i = 0; i < 4; i++) {
            out[i + 4 * j] = state[i][j];
        }
    }
}

void subbytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] = SBOX[state[i][j]];
        }
    }
}

void invsubbytes(uint8_t state[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] = INVSBOX[state[i][j]];
        }
    }
}

void shiftrows(uint8_t s[4][4]) {
    for (int r=1;r<4;r++) {
        uint8_t tmp[4];
        memcpy(tmp, s[r], 4);
        for (int i=0;i<4;i++)
            s[r][i] = tmp[(i + r) % 4];
    }
}

void invshiftrows(uint8_t state[4][4]) {
    for (int r = 1; r < 4; r++) {
        uint8_t temp[4];
        memcpy(temp, state[r], 4);
        for (int i = 0; i < 4; i++) {
            state[r][i] = temp[(i - r + 4) % 4];
        }
    }
}

void mixcolumns(uint8_t s[4][4]) {
    for (int c = 0; c < 4; c++) {
        uint8_t a0 = s[0][c];
        uint8_t a1 = s[1][c];
        uint8_t a2 = s[2][c];
        uint8_t a3 = s[3][c];

        s[0][c] = gmul(2, a0) ^ gmul(3,a1) ^ a2 ^ a3;
        s[1][c] = a0 ^ gmul(2, a1) ^ gmul(3, a2) ^ a3;
        s[2][c] = a0 ^ a1 ^ gmul(2, a2) ^ gmul(3, a3);
        s[3][c] = gmul(3, a0) ^ a1 ^ a2 ^ gmul(2, a3);
    }
}

void invmixcolumns(uint8_t s[4][4]) {
    for (int c = 0; c < 4; c++) {
        uint8_t a0 = s[0][c];
        uint8_t a1 = s[1][c];
        uint8_t a2 = s[2][c];
        uint8_t a3 = s[3][c];

        s[0][c] = gmul(14, a0) ^ gmul(11, a1) ^ gmul(13, a2) ^ gmul(9, a3);
        s[1][c] = gmul(9, a0) ^ gmul(14, a1) ^ gmul(11, a2) ^ gmul(13, a3);
        s[2][c] = gmul(13, a0) ^ gmul(9, a1) ^ gmul(14, a2) ^ gmul(11, a3);
        s[3][c] = gmul(11, a0) ^ gmul(13, a1) ^ gmul(9, a2) ^ gmul(14, a3);
    }
}

void addroundkey(uint8_t state[4][4], uint8_t roundkey[4][4]) {
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            state[i][j] ^= roundkey[i][j];
        }
    }
}

void rotword(uint8_t word[4]) {
    uint8_t temp = word[0];
    word[0] = word[1];
    word[1] = word[2];
    word[2] = word[3];
    word[3] = temp;
}

void subword(uint8_t word[4]) {
    for (int  i = 0; i < 4; i++) {
        word[i] = SBOX[word[i]];
    }
}

void keyexpansion(uint8_t (**roundkeys)[4][4], const uint8_t *key) {
    int totalwords = 4 * (NR + 1);
    uint8_t (*word)[4] = malloc(totalwords * sizeof(*word));

    for (int i = 0; i < NK; i++) {
        memcpy(word[i], key + 4 * i, 4);
    }

    for (int i = NK; i < totalwords; i++) {
        uint8_t temp[4];
        memcpy(temp, word[i - 1], 4);

        if (i % NK == 0) {
            rotword(temp);
            subword(temp);
            temp[0] ^= RCON[(i / NK) - 1];
        } else if (NK > 6 && i % NK == 4) {
            subword(temp);
        }

        for (int j = 0; j < 4; j++) {
            word[i][j] = word[i - NK][j] ^ temp[j];
        }
    }

    *roundkeys = malloc((NR + 1) * sizeof(uint8_t[4][4]));
    for (int r = 0; r <= NR; r++) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                (*roundkeys)[r][i][j] = word[4 * r + j][i];
            }
        }
    }

    free(word);
}

void encryptblock(uint8_t out[16], const uint8_t in[16], uint8_t (*roundkeys)[4][4]) {
    uint8_t state[4][4];
    bytestostate(state, in);

    addroundkey(state, roundkeys[0]);

    for (int r = 1; r < NR; r++) {
        subbytes(state);
        shiftrows(state);
        mixcolumns(state);
        addroundkey(state, roundkeys[r]);
    }

    subbytes(state);
    shiftrows(state);
    addroundkey(state, roundkeys[NR]);

    statetobytes(out, state);
}

void decryptblock(uint8_t out[16], const uint8_t in[16], uint8_t (*roundkeys)[4][4]) {
    uint8_t state[4][4];
    bytestostate(state, in);

    addroundkey(state, roundkeys[NR]);

    for (int r = NR - 1; r > 0; r--) {
        invshiftrows(state);
        invsubbytes(state);
        addroundkey(state, roundkeys[r]);
        invmixcolumns(state);
    }

    invshiftrows(state);
    invsubbytes(state);
    addroundkey(state, roundkeys[0]);

    statetobytes(out, state);
}

void xor(uint8_t *out, const uint8_t *a, const uint8_t *b, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] ^ b[i];
    }
}

int aescbcencrypt(uint8_t **out, const uint8_t *plaintext, int len, const uint8_t *key, const uint8_t iv[16]) {
    uint8_t (*roundkeys)[4][4];
    keyexpansion(&roundkeys, key);

    *out = malloc(len);
    uint8_t previous[16];
    memcpy(previous, iv, 16);

    uint8_t block[16], enc[16];

    for (int i = 0; i < len; i += 16) {
        xor(block, plaintext + i, previous, 16);
        encryptblock(enc, block, roundkeys);
        memcpy(*out + i, enc, 16);
        memcpy(previous, enc, 16);
    }

    free(roundkeys);
    return len;
}

int aescbcdecrypt(uint8_t **out, const uint8_t *ciphertext, int len, const uint8_t *key, const uint8_t iv[16]) {
    uint8_t (*roundkeys)[4][4];
    keyexpansion(&roundkeys, key);

    *out = malloc(len);

    uint8_t previous[16];
    memcpy(previous, iv, 16);

    uint8_t dec[16];

    for (int i = 0; i < len; i += 16) {
        decryptblock(dec, ciphertext + i, roundkeys);
        xor((*out) + i, dec, previous, 16);
        memcpy(previous, ciphertext + i, 16);
    }

    free(roundkeys);
    return len;
}

int hamming(const uint8_t *a, const uint8_t *b, int len) {
    int dist = 0;

    for (int i = 0; i < len; i++) {
        uint8_t x = a[i] ^ b[i];

        for (int j = 0; j < 8; j++) {
            dist += (x >> j) & 1;
        }
    }

    return dist;
}

void writehex(FILE *f, const uint8_t *buf, int n) {
    static const char h[] = "0123456789abcdef";
    for (int i = 0; i < n; i++) {
        fputc(h[buf[i] >> 4], f);
        fputc(h[buf[i] & 0x0f], f);
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Pass args in the format ./aes NR");
        return 1;
    }

    NR = atoi(argv[1]);
    if (NR <= 0) {
        printf("Invalid NR.");
        return 1;
    }

    char foldername[32];
    snprintf(foldername, sizeof(foldername), "%d", NR);
    mkdir(foldername, 0700);

    RCON = malloc(NR * sizeof(uint8_t));
    RCON[0] = 1;
    for (int i = 1; i < NR; i++) {
        int next = RCON[i - 1] << 1;
        if (next & 256) next ^= 283;
        RCON[i] = next & 255;
    }

    FILE *urand = fopen("/dev/urandom", "rb");
    if (!urand) {
        perror("urandom");
        return 1;
    }

    char path[128];
    snprintf(path, sizeof(path), "%s/results.csv", foldername);
    FILE *fcsv = fopen(path, "w");
    if (!fcsv) {
        perror("results.csv");
        return 1;
    }
    fprintf(fcsv, "nr,trial,avg_hamming_distance,key_hex,iv_hex,plaintext_hex\n");

    for (int t = 0; t < NUMTRIALS; t++) {
        uint8_t key[32], iv[16];
        uint8_t *plaintext = malloc(PLAINTEXTSIZE); 
        fread(key, 1, sizeof(key), urand);
        fread(iv, 1, sizeof(iv), urand);
        fread(plaintext, 1, PLAINTEXTSIZE, urand);

        uint8_t *cipherbuf;
        aescbcencrypt(&cipherbuf, plaintext, PLAINTEXTSIZE, key, iv);

        uint8_t ciphertext[512];
        memcpy(ciphertext, cipherbuf, 512);
        free(cipherbuf);

        double trialsum = 0;

        for (int s = 0; s < SUBTRIALS; s++) {
            uint8_t flipped[512];
            memcpy(flipped, plaintext, PLAINTEXTSIZE);
            flipped[s / 8] ^= 1 << (s % 8);

            uint8_t *flipbuf;
            aescbcencrypt(&flipbuf, flipped, PLAINTEXTSIZE, key, iv);

            uint8_t flippedciphertext[512];
            memcpy(flippedciphertext, flipbuf, 512);
            free(flipbuf);

            int hd = hamming(ciphertext, flippedciphertext, PLAINTEXTSIZE);
            trialsum += hd;
        }

        double trialavg = trialsum / SUBTRIALS;

        fprintf(fcsv, "%d,%d,%.3f,", NR, t + 1, trialavg);
        writehex(fcsv, key, sizeof(key));
        fputc(',', fcsv);
        writehex(fcsv, iv, sizeof(iv));
        fputc(',', fcsv);
        writehex(fcsv, plaintext, PLAINTEXTSIZE);
        fputc('\n', fcsv);

        free(plaintext);
    }

    fclose(fcsv);
    fclose(urand);
    free(RCON);
    return 0;
}