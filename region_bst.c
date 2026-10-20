/*
 * heatwave_bplus.c
 * DSA IA-2 - Climate Intelligence for Heatwave Monitoring (KJS-CES-01)
 *
 * Structure (exactly as in the report):
 *   RegionIndex : hash table  (region name -> B+ Tree)        O(1) average
 *   B+ Tree     : one per region, key = (date, city)          O(log n)
 *   Leaves are linked left-to-right, so a time window is read in O(log n + k).
 *
 * Operations: insert, delete, search, rangeQuery, seasonQuery, CSV loader.
 *
 * Build : gcc -Wall -Wextra -O2 -o heatwave heatwave_bplus.c -lm
 *         (for the full IMD dataset add -DMAX_KEYS=64 for a shallower tree)
 * Run   : ./heatwave                                      built-in demo
 *         ./heatwave data.csv                             load CSV, list regions
 *         ./heatwave data.csv "Region" 19510101 19511231 [threshold]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <math.h>

#ifndef MAX_KEYS
#define MAX_KEYS 4              /* max records in a leaf / max keys in an internal node */
#endif
#if MAX_KEYS < 3
#error "MAX_KEYS must be at least 3"
#endif
#define MIN_KEYS   (MAX_KEYS / 2)   /* minimum fill for every non-root node */
#define CITY_LEN   32
#define REGION_LEN 48
#define BUCKETS    7                /* 7 IMD regions (collisions handled by chaining) */

/* ------------------------------------------------------------------ */
/* Data types                                                          */
/* ------------------------------------------------------------------ */
typedef struct {
    int    date;                    /* YYYYMMDD, so int order == time order */
    char   city[CITY_LEN];
    double lat, lon, tmax, tmin, rain;   /* tmin / rain are NAN when missing */
} Reading;

typedef struct {                    /* search key = (timestamp, city) */
    int  date;
    char city[CITY_LEN];
} Key;

typedef struct BNode {
    int isLeaf;
    int n;                          /* leaf: #records, internal: #keys */
    union {
        struct {                    /* leaf: sorted records + link to next leaf */
            Reading       recs[MAX_KEYS + 1];     /* +1 slot lets a leaf overflow before splitting */
            struct BNode *next;
        } leaf;
        struct {                    /* internal: routing keys + children */
            Key           keys[MAX_KEYS + 1];
            struct BNode *child[MAX_KEYS + 2];
        } in;
    } u;
} BNode;

typedef struct {
    BNode *root;
    long   count;
} BPlusTree;

typedef struct RegionEntry {        /* one chain node of the hash table */
    char                name[REGION_LEN];
    BPlusTree          *tree;
    struct RegionEntry *next;
} RegionEntry;

static RegionEntry *RegionIndex[BUCKETS];   /* the top-level hash table */

enum { R_OK = 0, R_SPLIT = 1, R_DUP = -1, R_NOMEM = -2 };

/* ------------------------------------------------------------------ */
/* Key helpers                                                         */
/* ------------------------------------------------------------------ */
static Key keyOf(const Reading *r) {
    Key k;
    k.date = r->date;
    strncpy(k.city, r->city, CITY_LEN - 1);
    k.city[CITY_LEN - 1] = '\0';
    return k;
}

static int cmpKey(const Key *a, const Key *b) {        /* date first, then city */
    if (a->date != b->date) return a->date < b->date ? -1 : 1;
    return strcmp(a->city, b->city);
}

static int cmpRecKey(const Reading *r, const Key *k) {
    Key rk = keyOf(r);
    return cmpKey(&rk, k);
}

/* index of the child to follow: first i with key < keys[i] (binary search) */
static int findChild(const BNode *node, const Key *k) {
    int lo = 0, hi = node->n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cmpKey(k, &node->u.in.keys[mid]) < 0) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

/* first record in the leaf with key >= k (binary search) */
static int leafLowerBound(const BNode *leaf, const Key *k) {
    int lo = 0, hi = leaf->n;
    while (lo < hi) {
        int mid = (lo + hi) / 2;
        if (cmpRecKey(&leaf->u.leaf.recs[mid], k) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

static BNode *newNode(int isLeaf) {
    BNode *n = calloc(1, sizeof(BNode));
    if (n) n->isLeaf = isLeaf;
    return n;
}

static BNode *findLeaf(BNode *node, const Key *k) {     /* root-to-leaf descent, O(log n) */
    while (!node->isLeaf) node = node->u.in.child[findChild(node, k)];
    return node;
}

/* ------------------------------------------------------------------ */
/* B+ Tree: create / free / height                                     */
/* ------------------------------------------------------------------ */
static BPlusTree *btNew(void) {
    BPlusTree *t = malloc(sizeof(BPlusTree));
    if (!t) return NULL;
    t->root = newNode(1);
    t->count = 0;
    if (!t->root) { free(t); return NULL; }
    return t;
}

static void freeNode(BNode *n) {
    if (!n) return;
    if (!n->isLeaf)
        for (int i = 0; i <= n->n; i++) freeNode(n->u.in.child[i]);
    free(n);
}

static void btFree(BPlusTree *t) {
    if (!t) return;
    freeNode(t->root);
    free(t);
}

static int btHeight(const BPlusTree *t) {
    int h = 1;
    for (const BNode *n = t->root; !n->isLeaf; n = n->u.in.child[0]) h++;
    return h;
}

/* ------------------------------------------------------------------ */
/* INSERT                                                              */
/* ------------------------------------------------------------------ */
/* Returns R_OK, R_SPLIT (child split: *upKey / *upNode must be added to the
 * parent), R_DUP or R_NOMEM. Any node that may be needed for a split is
 * allocated BEFORE the tree is modified, so a failed malloc never leaves
 * the tree half-updated. */
static int insertRec(BNode *node, const Reading *r, Key *upKey, BNode **upNode) {
    Key k = keyOf(r);

    if (node->isLeaf) {
        int pos = leafLowerBound(node, &k);
        if (pos < node->n && cmpRecKey(&node->u.leaf.recs[pos], &k) == 0)
            return R_DUP;                                   /* same (date, city) */

        BNode *right = NULL;
        if (node->n == MAX_KEYS) {                          /* will overflow -> need a sibling */
            right = newNode(1);
            if (!right) return R_NOMEM;
        }
        memmove(&node->u.leaf.recs[pos + 1], &node->u.leaf.recs[pos],
                (size_t)(node->n - pos) * sizeof(Reading));
        node->u.leaf.recs[pos] = *r;                        /* keeps leaf time-ordered */
        node->n++;
        if (!right) return R_OK;

        /* leaf split: upper half moves to the new leaf, chain is re-linked */
        int mid = node->n / 2;
        right->n = node->n - mid;
        memcpy(right->u.leaf.recs, &node->u.leaf.recs[mid], (size_t)right->n * sizeof(Reading));
        node->n = mid;
        right->u.leaf.next = node->u.leaf.next;
        node->u.leaf.next = right;
        *upKey = keyOf(&right->u.leaf.recs[0]);             /* copied up (B+ Tree rule) */
        *upNode = right;
        return R_SPLIT;
    }

    int i = findChild(node, &k);
    BNode *spare = NULL;
    if (node->n == MAX_KEYS) {                              /* this node may have to split too */
        spare = newNode(0);
        if (!spare) return R_NOMEM;
    }
    Key ck;
    BNode *cn = NULL;
    int res = insertRec(node->u.in.child[i], r, &ck, &cn);
    if (res != R_SPLIT) { free(spare); return res; }

    /* absorb the child's split: new key at keys[i], new child at child[i+1] */
    memmove(&node->u.in.keys[i + 1], &node->u.in.keys[i], (size_t)(node->n - i) * sizeof(Key));
    memmove(&node->u.in.child[i + 2], &node->u.in.child[i + 1], (size_t)(node->n - i) * sizeof(BNode *));
    node->u.in.keys[i] = ck;
    node->u.in.child[i + 1] = cn;
    node->n++;
    if (node->n <= MAX_KEYS) { free(spare); return R_OK; }

    /* internal split: middle key MOVES up, it does not stay in either half */
    int mid = node->n / 2;
    *upKey = node->u.in.keys[mid];
    spare->n = node->n - mid - 1;
    memcpy(spare->u.in.keys, &node->u.in.keys[mid + 1], (size_t)spare->n * sizeof(Key));
    memcpy(spare->u.in.child, &node->u.in.child[mid + 1], (size_t)(spare->n + 1) * sizeof(BNode *));
    node->n = mid;
    *upNode = spare;
    return R_SPLIT;
}

static int btInsert(BPlusTree *t, const Reading *r) {
    BNode *newRoot = NULL;
    if (t->root->n == MAX_KEYS) {                           /* root may split -> tree grows by one level */
        newRoot = newNode(0);
        if (!newRoot) return R_NOMEM;
    }
    Key up;
    BNode *right = NULL;
    int res = insertRec(t->root, r, &up, &right);
    if (res == R_SPLIT) {
        newRoot->n = 1;
        newRoot->u.in.keys[0] = up;
        newRoot->u.in.child[0] = t->root;
        newRoot->u.in.child[1] = right;
        t->root = newRoot;
        newRoot = NULL;
        res = R_OK;
    }
    free(newRoot);
    if (res == R_OK) t->count++;
    return res;
}

/* ------------------------------------------------------------------ */
/* SEARCH                                                              */
/* ------------------------------------------------------------------ */
static const Reading *btSearch(const BPlusTree *t, int date, const char *city) {
    Key k;
    k.date = date;
    strncpy(k.city, city, CITY_LEN - 1);
    k.city[CITY_LEN - 1] = '\0';
    const BNode *leaf = findLeaf(t->root, &k);
    int pos = leafLowerBound(leaf, &k);
    if (pos < leaf->n && cmpRecKey(&leaf->u.leaf.recs[pos], &k) == 0)
        return &leaf->u.leaf.recs[pos];
    return NULL;
}

/* ------------------------------------------------------------------ */
/* DELETE (borrow from a sibling, otherwise merge)                     */
/* ------------------------------------------------------------------ */
static void borrowFromLeft(BNode *p, int i) {
    BNode *c = p->u.in.child[i], *L = p->u.in.child[i - 1];
    if (c->isLeaf) {
        memmove(&c->u.leaf.recs[1], &c->u.leaf.recs[0], (size_t)c->n * sizeof(Reading));
        c->u.leaf.recs[0] = L->u.leaf.recs[L->n - 1];
        c->n++; L->n--;
        p->u.in.keys[i - 1] = keyOf(&c->u.leaf.recs[0]);
    } else {                                                /* rotate through the parent */
        memmove(&c->u.in.keys[1], &c->u.in.keys[0], (size_t)c->n * sizeof(Key));
        memmove(&c->u.in.child[1], &c->u.in.child[0], (size_t)(c->n + 1) * sizeof(BNode *));
        c->u.in.keys[0] = p->u.in.keys[i - 1];
        c->u.in.child[0] = L->u.in.child[L->n];
        p->u.in.keys[i - 1] = L->u.in.keys[L->n - 1];
        c->n++; L->n--;
    }
}

static void borrowFromRight(BNode *p, int i) {
    BNode *c = p->u.in.child[i], *R = p->u.in.child[i + 1];
    if (c->isLeaf) {
        c->u.leaf.recs[c->n] = R->u.leaf.recs[0];
        c->n++;
        memmove(&R->u.leaf.recs[0], &R->u.leaf.recs[1], (size_t)(R->n - 1) * sizeof(Reading));
        R->n--;
        p->u.in.keys[i] = keyOf(&R->u.leaf.recs[0]);
    } else {
        c->u.in.keys[c->n] = p->u.in.keys[i];
        c->u.in.child[c->n + 1] = R->u.in.child[0];
        c->n++;
        p->u.in.keys[i] = R->u.in.keys[0];
        memmove(&R->u.in.keys[0], &R->u.in.keys[1], (size_t)(R->n - 1) * sizeof(Key));
        memmove(&R->u.in.child[0], &R->u.in.child[1], (size_t)R->n * sizeof(BNode *));
        R->n--;
    }
}

static void mergeChildren(BNode *p, int j) {                /* merge child[j+1] into child[j] */
    BNode *a = p->u.in.child[j], *b = p->u.in.child[j + 1];
    if (a->isLeaf) {
        memcpy(&a->u.leaf.recs[a->n], b->u.leaf.recs, (size_t)b->n * sizeof(Reading));
        a->n += b->n;
        a->u.leaf.next = b->u.leaf.next;                    /* unlink b from the leaf chain */
    } else {
        a->u.in.keys[a->n] = p->u.in.keys[j];               /* separator comes down */
        memcpy(&a->u.in.keys[a->n + 1], b->u.in.keys, (size_t)b->n * sizeof(Key));
        memcpy(&a->u.in.child[a->n + 1], b->u.in.child, (size_t)(b->n + 1) * sizeof(BNode *));
        a->n += 1 + b->n;
    }
    free(b);
    memmove(&p->u.in.keys[j], &p->u.in.keys[j + 1], (size_t)(p->n - j - 1) * sizeof(Key));
    memmove(&p->u.in.child[j + 1], &p->u.in.child[j + 2], (size_t)(p->n - j - 1) * sizeof(BNode *));
    p->n--;
}

static void fixChild(BNode *p, int i) {                     /* child i is underfull */
    BNode *L = i > 0 ? p->u.in.child[i - 1] : NULL;
    BNode *R = i < p->n ? p->u.in.child[i + 1] : NULL;
    if (L && L->n > MIN_KEYS)       borrowFromLeft(p, i);
    else if (R && R->n > MIN_KEYS)  borrowFromRight(p, i);
    else if (L)                     mergeChildren(p, i - 1);
    else                            mergeChildren(p, i);
}

static int deleteRec(BNode *node, const Key *k) {
    if (node->isLeaf) {
        int pos = leafLowerBound(node, k);
        if (pos >= node->n || cmpRecKey(&node->u.leaf.recs[pos], k) != 0) return 0;
        memmove(&node->u.leaf.recs[pos], &node->u.leaf.recs[pos + 1],
                (size_t)(node->n - pos - 1) * sizeof(Reading));
        node->n--;
        return 1;
    }
    int i = findChild(node, k);
    int found = deleteRec(node->u.in.child[i], k);
    if (found && node->u.in.child[i]->n < MIN_KEYS) fixChild(node, i);
    return found;
}

static int btDelete(BPlusTree *t, int date, const char *city) {
    Key k;
    k.date = date;
    strncpy(k.city, city, CITY_LEN - 1);
    k.city[CITY_LEN - 1] = '\0';
    int found = deleteRec(t->root, &k);
    if (!found) return 0;
    t->count--;
    if (!t->root->isLeaf && t->root->n == 0) {              /* root lost its last key: tree shrinks */
        BNode *old = t->root;
        t->root = old->u.in.child[0];
        free(old);
    }
    return 1;
}

/* ------------------------------------------------------------------ */
/* RANGE QUERY: descend once, then walk the leaf chain                 */
/* ------------------------------------------------------------------ */
typedef struct { long count, hot; double sum, max; } RangeStats;

static void printReading(const Reading *r) {
    printf("%d,%s,%.4f,%.4f,%.2f,", r->date, r->city, r->lat, r->lon, r->tmax);
    if (isnan(r->tmin)) printf("NA,"); else printf("%.2f,", r->tmin);
    if (isnan(r->rain)) printf("NA\n"); else printf("%.2f\n", r->rain);
}

static void btRange(const BPlusTree *t, int start, int end, double threshold,
                    int print, RangeStats *st) {
    memset(st, 0, sizeof *st);
    Key lo;
    lo.date = start;
    lo.city[0] = '\0';                                      /* "" sorts before every city name */
    const BNode *leaf = findLeaf(t->root, &lo);             /* O(log n) */
    int i = leafLowerBound(leaf, &lo);
    while (leaf) {                                          /* O(k) along the chain */
        for (; i < leaf->n; i++) {
            const Reading *r = &leaf->u.leaf.recs[i];
            if (r->date > end) return;                      /* window closed: early exit */
            if (r->date < start) continue;
            if (print) printReading(r);
            if (st->count == 0 || r->tmax > st->max) st->max = r->tmax;
            st->count++;
            st->sum += r->tmax;
            if (r->tmax >= threshold) st->hot++;
        }
        leaf = leaf->u.leaf.next;
        i = 0;
    }
}

/* ------------------------------------------------------------------ */
/* Hash table: region name -> B+ Tree (separate chaining)              */
/* ------------------------------------------------------------------ */
static unsigned hashStr(const char *s) {                    /* djb2 */
    unsigned long h = 5381;
    while (*s) h = h * 33 + (unsigned char)*s++;
    return (unsigned)(h % BUCKETS);
}

static BPlusTree *getRegion(const char *name, int create) {
    unsigned b = hashStr(name);
    for (RegionEntry *e = RegionIndex[b]; e; e = e->next)
        if (strcmp(e->name, name) == 0) return e->tree;
    if (!create) return NULL;
    RegionEntry *e = calloc(1, sizeof *e);
    if (!e) return NULL;
    strncpy(e->name, name, REGION_LEN - 1);
    e->tree = btNew();
    if (!e->tree) { free(e); return NULL; }
    e->next = RegionIndex[b];
    RegionIndex[b] = e;
    return e->tree;
}

static void freeAll(void) {
    for (int b = 0; b < BUCKETS; b++) {
        RegionEntry *e = RegionIndex[b];
        while (e) {
            RegionEntry *nx = e->next;
            btFree(e->tree);
            free(e);
            e = nx;
        }
        RegionIndex[b] = NULL;
    }
}

/* ------------------------------------------------------------------ */
/* Public operations (region + key), as in the pseudocode              */
/* ------------------------------------------------------------------ */
static const char *insertReading(const char *region, const Reading *r) {
    BPlusTree *t = getRegion(region, 1);
    if (!t) return "out of memory";
    int res = btInsert(t, r);
    return res == R_OK ? "inserted" : res == R_DUP ? "duplicate record" : "out of memory";
}

static const char *deleteReading(const char *region, int date, const char *city) {
    BPlusTree *t = getRegion(region, 0);
    if (!t) return "region not found";
    return btDelete(t, date, city) ? "deleted" : "record not found";
}

static const Reading *searchReading(const char *region, int date, const char *city) {
    BPlusTree *t = getRegion(region, 0);
    return t ? btSearch(t, date, city) : NULL;
}

static void rangeQuery(const char *region, int start, int end, double threshold) {
    BPlusTree *t = getRegion(region, 0);
    if (!t) { printf("region not found: %s\n", region); return; }
    RangeStats st;
    printf("Date,City,Lat,Lon,Tmax,Tmin,Rain\n");
    btRange(t, start, end, threshold, 1, &st);
    printf("-- %s, %d..%d: records=%ld", region, start, end, st.count);
    if (st.count)
        printf(", avg Tmax=%.2f, max Tmax=%.2f, days >= %.1f C: %ld",
               st.sum / (double)st.count, st.max, threshold, st.hot);
    printf("\n");
}

/* Season is NOT stored: it is derived from the month, so a seasonal query is
 * just a time-window query on the same tree (IMD seasons). */
enum { WINTER, PRE_MONSOON, MONSOON, POST_MONSOON };
static const char *SEASON_NAME[4] = {
    "Winter (Jan-Feb)", "Pre-monsoon (Mar-May)", "Monsoon (Jun-Sep)", "Post-monsoon (Oct-Dec)"
};

static int seasonOf(int date) {
    int m = (date / 100) % 100;
    return m <= 2 ? WINTER : m <= 5 ? PRE_MONSOON : m <= 9 ? MONSOON : POST_MONSOON;
}

static void seasonQuery(const char *region, int year, int season, double threshold) {
    static const int fromM[4] = {1, 3, 6, 10}, toM[4] = {2, 5, 9, 12};
    int start = year * 10000 + fromM[season] * 100 + 1;
    int end   = year * 10000 + toM[season] * 100 + 31;      /* day 31 only acts as an upper bound */
    printf("Season query: %s %d in %s\n", SEASON_NAME[season], year, region);
    rangeQuery(region, start, end, threshold);
}

/* ------------------------------------------------------------------ */
/* Display helpers                                                     */
/* ------------------------------------------------------------------ */
static void dumpNode(const BNode *n, int depth) {
    printf("%*s", depth * 4, "");
    if (n->isLeaf) {
        printf("Leaf:");
        for (int i = 0; i < n->n; i++)
            printf(" (%d,%s)", n->u.leaf.recs[i].date, n->u.leaf.recs[i].city);
        printf("\n");
    } else {
        printf("Node: |");
        for (int i = 0; i < n->n; i++) printf(" %d,%s |", n->u.in.keys[i].date, n->u.in.keys[i].city);
        printf("\n");
        for (int i = 0; i <= n->n; i++) dumpNode(n->u.in.child[i], depth + 1);
    }
}

static void listRegions(void) {
    printf("%-22s %10s %7s\n", "Region", "Records", "Height");
    for (int b = 0; b < BUCKETS; b++)
        for (RegionEntry *e = RegionIndex[b]; e; e = e->next)
            printf("%-22s %10ld %7d\n", e->name, e->tree->count, btHeight(e->tree));
}

/* ------------------------------------------------------------------ */
/* CSV loader (header-driven; works with the Streamlit app's CSV)      */
/* ------------------------------------------------------------------ */
static char *trim(char *s) {
    while (isspace((unsigned char)*s)) s++;
    char *e = s + strlen(s);
    while (e > s && isspace((unsigned char)e[-1])) *--e = '\0';
    if (e - s >= 2 && s[0] == '"' && e[-1] == '"') { s++; *--e = '\0'; }
    return s;
}

static int splitCSV(char *line, char *f[], int maxf) {
    int n = 0;
    f[n++] = line;
    for (char *p = line; *p && n < maxf; p++)
        if (*p == ',') { *p = '\0'; f[n++] = p + 1; }
    for (int i = 0; i < n; i++) f[i] = trim(f[i]);
    return n;
}

static void normName(const char *s, char *out, size_t sz) {   /* "Temp Max" -> "tempmax" */
    size_t j = 0;
    for (; *s && j + 1 < sz; s++)
        if (isalnum((unsigned char)*s)) out[j++] = (char)tolower((unsigned char)*s);
    out[j] = '\0';
}

static double parseNum(const char *s) {
    if (!*s) return NAN;
    char *end;
    double v = strtod(s, &end);
    return end == s ? NAN : v;
}

static int parseDate(const char *s) {                       /* dd-mm-yyyy (as the app) or yyyy-mm-dd */
    char buf[64];
    strncpy(buf, s, sizeof buf - 1);
    buf[sizeof buf - 1] = '\0';
    for (char *p = buf; *p; p++) if (*p == '/' || *p == '.') *p = '-';
    int a, b, c;
    if (sscanf(buf, "%d-%d-%d", &a, &b, &c) != 3) return -1;
    int y, m = b, d;
    if (a > 31) { y = a; d = c; } else { d = a; y = c; }
    if (y < 1800 || y > 2200 || m < 1 || m > 12 || d < 1 || d > 31) return -1;
    return y * 10000 + m * 100 + d;
}

static long loadCSV(const char *path) {
    FILE *fp = fopen(path, "r");
    if (!fp) { printf("Cannot open %s\n", path); return -1; }
    char line[2048], *f[32];
    if (!fgets(line, sizeof line, fp)) { fclose(fp); return -1; }
    char *h = line;
    if ((unsigned char)h[0] == 0xEF && (unsigned char)h[1] == 0xBB && (unsigned char)h[2] == 0xBF) h += 3;
    int nf = splitCSV(h, f, 32);

    int cReg = -1, cCity = -1, cDate = -1, cLat = -1, cLon = -1, cMax = -1, cMin = -1, cRain = -1;
    for (int i = 0; i < nf; i++) {
        char nm[32];
        normName(f[i], nm, sizeof nm);
        if (!strcmp(nm, "region")) cReg = i;
        else if (!strcmp(nm, "city") || !strcmp(nm, "station")) cCity = i;
        else if (!strcmp(nm, "date") || !strcmp(nm, "timestamp")) cDate = i;
        else if (!strcmp(nm, "lat") || !strcmp(nm, "latitude")) cLat = i;
        else if (!strcmp(nm, "lon") || !strcmp(nm, "long") || !strcmp(nm, "longitude")) cLon = i;
        else if (!strcmp(nm, "tempmax") || !strcmp(nm, "tmax")) cMax = i;
        else if (!strcmp(nm, "tempmin") || !strcmp(nm, "tmin")) cMin = i;
        else if (!strcmp(nm, "rain") || !strcmp(nm, "rainfall")) cRain = i;
    }
    if (cReg < 0 || cDate < 0 || cLat < 0 || cLon < 0 || cMax < 0) {
        printf("CSV needs columns: region, Date, lat, lon, Temp Max (city, Temp Min, Rain optional)\n");
        fclose(fp);
        return -1;
    }

    long ok = 0, dup = 0, bad = 0;
    while (fgets(line, sizeof line, fp)) {
        nf = splitCSV(line, f, 32);
        int need = cReg > cDate ? cReg : cDate;
        if (cLat > need) need = cLat;
        if (cLon > need) need = cLon;
        if (cMax > need) need = cMax;
        if (nf <= need) { bad++; continue; }

        Reading r;
        memset(&r, 0, sizeof r);
        r.date = parseDate(f[cDate]);
        r.lat  = parseNum(f[cLat]);
        r.lon  = parseNum(f[cLon]);
        r.tmax = parseNum(f[cMax]);
        r.tmin = (cMin  >= 0 && cMin  < nf) ? parseNum(f[cMin])  : NAN;
        r.rain = (cRain >= 0 && cRain < nf) ? parseNum(f[cRain]) : NAN;
        if (r.date < 0 || isnan(r.lat) || isnan(r.lon) || isnan(r.tmax) || !*f[cReg]) { bad++; continue; }
        if (cCity >= 0 && cCity < nf && *f[cCity]) strncpy(r.city, f[cCity], CITY_LEN - 1);
        else snprintf(r.city, CITY_LEN, "%.2f_%.2f", r.lat, r.lon);   /* no city column: lat/lon identifies the station */

        const char *res = insertReading(f[cReg], &r);
        if (!strcmp(res, "inserted")) ok++;
        else if (!strcmp(res, "duplicate record")) dup++;
        else { bad++; }
    }
    fclose(fp);
    printf("Loaded %s: %ld inserted, %ld duplicates, %ld skipped\n", path, ok, dup, bad);
    return ok;
}

/* ------------------------------------------------------------------ */
/* Demo                                                                */
/* ------------------------------------------------------------------ */
static Reading mk(int date, const char *city, double lat, double lon,
                  double tmax, double tmin, double rain) {
    Reading r;
    memset(&r, 0, sizeof r);
    r.date = date;
    strncpy(r.city, city, CITY_LEN - 1);
    r.lat = lat; r.lon = lon; r.tmax = tmax; r.tmin = tmin; r.rain = rain;
    return r;
}

static void demo(void) {
    static const struct { const char *name; double lat, lon; } cities[] = {
        {"Nagpur", 21.1458, 79.0882}, {"Bhopal", 23.2599, 77.4126}, {"Raipur", 21.2514, 81.6296}
    };

    printf("== 1. INSERT: 3 cities x 10 days into Central India, 5 days of Mumbai into West Coast\n");
    for (int c = 0; c < 3; c++)
        for (int day = 1; day <= 10; day++) {
            double tmax = 38.0 + ((day * 5 + c * 3) % 9) * 0.6;
            Reading r = mk(20260500 + day, cities[c].name, cities[c].lat, cities[c].lon, tmax, tmax - 12.5, 0.0);
            insertReading("Central India", &r);
        }
    static const double mum[5] = {36.8, 37.2, 38.1, 39.5, 41.2};
    for (int day = 1; day <= 5; day++) {
        Reading r = mk(20260500 + day, "Mumbai", 19.0760, 72.8777, mum[day - 1], 27.0, 0.0);
        insertReading("West Coast", &r);
    }
    listRegions();

    printf("\n== 2. West Coast tree (5 records, MAX_KEYS=%d, so the leaf has split)\n", MAX_KEYS);
    dumpNode(getRegion("West Coast", 0)->root, 0);

    printf("\n== 3. INSERT duplicate (20260502, Mumbai)\n");
    {
        Reading dupR = mk(20260502, "Mumbai", 19.0760, 72.8777, 99.0, 27.0, 0.0);
        printf("   -> %s\n", insertReading("West Coast", &dupR));
    }

    printf("\n== 4. SEARCH (20260505, Mumbai) in West Coast\n");
    const Reading *hit = searchReading("West Coast", 20260505, "Mumbai");
    if (hit) printReading(hit); else printf("not found\n");
    printf("   SEARCH (20260599, Mumbai): %s\n",
           searchReading("West Coast", 20260599, "Mumbai") ? "found" : "not found");

    printf("\n== 5. RANGE QUERY Central India 20260503..20260505, threshold 40 C\n");
    rangeQuery("Central India", 20260503, 20260505, 40.0);

    printf("\n== 6. SEASON QUERY (derived from month)\n");
    seasonQuery("West Coast", 2026, PRE_MONSOON, 40.0);
    printf("   seasonOf(20260502) = %s\n", SEASON_NAME[seasonOf(20260502)]);

    printf("\n== 7. DELETE\n");
    printf("   delete (20260502, Mumbai): %s\n", deleteReading("West Coast", 20260502, "Mumbai"));
    printf("   delete (20260502, Mumbai) again: %s\n", deleteReading("West Coast", 20260502, "Mumbai"));
    printf("   delete in unknown region: %s\n", deleteReading("Atlantis", 20260502, "Mumbai"));
    for (int day = 1; day <= 10; day++)                     /* delete a whole city: forces borrow / merge */
        deleteReading("Central India", 20260500 + day, "Bhopal");
    printf("\n== 8. After deleting every Bhopal record\n");
    listRegions();
    rangeQuery("Central India", 20260501, 20260502, 40.0);
}

int main(int argc, char **argv) {
    if (argc >= 2) {
        if (loadCSV(argv[1]) < 0) return 1;
        if (argc >= 5) rangeQuery(argv[2], atoi(argv[3]), atoi(argv[4]), argc >= 6 ? atof(argv[5]) : 40.0);
        else listRegions();
    } else {
        demo();
    }
    freeAll();
    return 0;
}
