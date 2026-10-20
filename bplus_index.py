"""
bplus_index.py
DSA IA-2 - Climate Intelligence for Heatwave Monitoring (KJS-CES-01)

Python version of the structure in the report (same design as the C program):

    RegionIndex : hash table (7 buckets, chaining)  region name -> BPlusTree
    BPlusTree   : one per region, key = (date, city), leaves linked left to right

    insert / delete / search : O(log n)
    range_query              : O(log n + k)  (one descent, then walk the leaf chain)
"""
from bisect import bisect_left, bisect_right
from math import nan

# IMD season definitions: season -> (first month, last month). Season is NOT a hash key; it is derived
# from the month of the timestamp, so a seasonal query is a set of time-window range queries.
SEASONS = {"Winter": (1, 2), "Pre-monsoon": (3, 5), "Monsoon": (6, 9), "Post-monsoon": (10, 12)}


# ----------------------------------------------------------------------
# B+ Tree
# ----------------------------------------------------------------------
class _Leaf:
    __slots__ = ("keys", "recs", "next")

    def __init__(self):
        self.keys = []      # sorted (date, city) keys
        self.recs = []      # records, parallel to keys
        self.next = None    # link to the next leaf


class _Internal:
    __slots__ = ("keys", "children")

    def __init__(self):
        self.keys = []      # routing keys only
        self.children = []  # len(children) == len(keys) + 1


class BPlusTree:
    def __init__(self, max_keys=64):
        if max_keys < 3:
            raise ValueError("max_keys must be at least 3")
        self.max_keys = max_keys
        self.min_keys = max_keys // 2
        self.root = _Leaf()
        self.count = 0

    # ---------------- helpers ----------------
    def _find_leaf(self, key):
        node = self.root
        while isinstance(node, _Internal):
            node = node.children[bisect_right(node.keys, key)]
        return node

    # ---------------- search ----------------
    def search(self, key):
        leaf = self._find_leaf(key)
        i = bisect_left(leaf.keys, key)
        if i < len(leaf.keys) and leaf.keys[i] == key:
            return leaf.recs[i]
        return None

    # ---------------- insert ----------------
    def insert(self, key, rec):
        """Returns True if inserted, False if the key already exists."""
        inserted, split = self._insert(self.root, key, rec)
        if split is not None:                       # root split: tree grows one level
            up_key, right = split
            new_root = _Internal()
            new_root.keys = [up_key]
            new_root.children = [self.root, right]
            self.root = new_root
        if inserted:
            self.count += 1
        return inserted

    def _insert(self, node, key, rec):
        if isinstance(node, _Leaf):
            i = bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                return False, None                  # duplicate record
            node.keys.insert(i, key)
            node.recs.insert(i, rec)
            if len(node.keys) <= self.max_keys:
                return True, None
            mid = len(node.keys) // 2               # leaf split: key is COPIED up
            right = _Leaf()
            right.keys, right.recs = node.keys[mid:], node.recs[mid:]
            del node.keys[mid:], node.recs[mid:]
            right.next, node.next = node.next, right
            return True, (right.keys[0], right)

        i = bisect_right(node.keys, key)
        inserted, split = self._insert(node.children[i], key, rec)
        if split is None:
            return inserted, None
        up_key, right = split
        node.keys.insert(i, up_key)
        node.children.insert(i + 1, right)
        if len(node.keys) <= self.max_keys:
            return inserted, None
        mid = len(node.keys) // 2                   # internal split: middle key MOVES up
        up = node.keys[mid]
        new = _Internal()
        new.keys, new.children = node.keys[mid + 1:], node.children[mid + 1:]
        del node.keys[mid:], node.children[mid + 1:]
        return inserted, (up, new)

    # ---------------- delete ----------------
    def delete(self, key):
        """Returns True if a record was removed."""
        found = self._delete(self.root, key)
        if not found:
            return False
        self.count -= 1
        if isinstance(self.root, _Internal) and not self.root.keys:
            self.root = self.root.children[0]       # root lost its last key: tree shrinks
        return True

    def _delete(self, node, key):
        if isinstance(node, _Leaf):
            i = bisect_left(node.keys, key)
            if i >= len(node.keys) or node.keys[i] != key:
                return False
            del node.keys[i], node.recs[i]
            return True
        i = bisect_right(node.keys, key)
        child = node.children[i]
        found = self._delete(child, key)
        if found and len(child.keys) < self.min_keys:
            self._fix_child(node, i)
        return found

    def _fix_child(self, p, i):
        left = p.children[i - 1] if i > 0 else None
        right = p.children[i + 1] if i < len(p.keys) else None
        if left is not None and len(left.keys) > self.min_keys:
            self._borrow_from_left(p, i)
        elif right is not None and len(right.keys) > self.min_keys:
            self._borrow_from_right(p, i)
        elif left is not None:
            self._merge(p, i - 1)
        else:
            self._merge(p, i)

    @staticmethod
    def _borrow_from_left(p, i):
        c, left = p.children[i], p.children[i - 1]
        if isinstance(c, _Leaf):
            c.keys.insert(0, left.keys.pop())
            c.recs.insert(0, left.recs.pop())
            p.keys[i - 1] = c.keys[0]
        else:                                       # rotate through the parent
            c.keys.insert(0, p.keys[i - 1])
            c.children.insert(0, left.children.pop())
            p.keys[i - 1] = left.keys.pop()

    @staticmethod
    def _borrow_from_right(p, i):
        c, right = p.children[i], p.children[i + 1]
        if isinstance(c, _Leaf):
            c.keys.append(right.keys.pop(0))
            c.recs.append(right.recs.pop(0))
            p.keys[i] = right.keys[0]
        else:
            c.keys.append(p.keys[i])
            c.children.append(right.children.pop(0))
            p.keys[i] = right.keys.pop(0)

    @staticmethod
    def _merge(p, j):                               # merge child j+1 into child j
        a, b = p.children[j], p.children[j + 1]
        if isinstance(a, _Leaf):
            a.keys += b.keys
            a.recs += b.recs
            a.next = b.next                         # unlink b from the leaf chain
        else:
            a.keys.append(p.keys[j])                # separator comes down
            a.keys += b.keys
            a.children += b.children
        del p.keys[j], p.children[j + 1]

    # ---------------- range query ----------------
    def range_query(self, start, end):
        """All records with start <= date <= end (dates as YYYYMMDD ints).
        Returns (list of (key, record), stats)."""
        lo = (start, "")                            # "" sorts before every city name
        leaf = self._find_leaf(lo)                  # O(log n) descent
        i = bisect_left(leaf.keys, lo)
        out, leaves, examined = [], 0, 0
        while leaf is not None:                     # O(k): walk the leaf chain
            leaves += 1
            keys = leaf.keys
            while i < len(keys):
                examined += 1
                if keys[i][0] > end:                # window closed: early exit
                    return out, self._stats(leaves, examined)
                out.append((keys[i], leaf.recs[i]))
                i += 1
            leaf, i = leaf.next, 0
        return out, self._stats(leaves, examined)

    def _stats(self, leaves, examined):
        return {"height": self.height(), "leaves_visited": leaves, "records_examined": examined}

    # ---------------- inspection ----------------
    def height(self):
        h, node = 1, self.root
        while isinstance(node, _Internal):
            node, h = node.children[0], h + 1
        return h

    def leaf_count(self):
        node = self.root
        while isinstance(node, _Internal):
            node = node.children[0]
        n = 0
        while node is not None:
            n, node = n + 1, node.next
        return n

    def first_items(self, n):
        """First n (key, record) pairs in time order, read along the leaf chain."""
        node = self.root
        while isinstance(node, _Internal):
            node = node.children[0]
        out = []
        while node is not None and len(out) < n:
            out.extend(zip(node.keys, node.recs))
            node = node.next
        return out[:n]

    def last_item(self):
        """Newest (key, record) in the tree: follow the right-most child to the last leaf. O(log n)."""
        node = self.root
        while isinstance(node, _Internal):
            node = node.children[-1]
        return (node.keys[-1], node.recs[-1]) if node.keys else None

    def dump(self):
        """Text drawing of the tree (use only on small trees)."""
        lines = []

        def walk(node, depth):
            pad = "    " * depth
            if isinstance(node, _Leaf):
                body = ", ".join(fmt_key(k) for k in node.keys)
                lines.append(f"{pad}Leaf [{body}]" + (" -> next leaf" if node.next else " -> NULL"))
            else:
                lines.append(f"{pad}Node | " + " | ".join(fmt_key(k) for k in node.keys) + " |")
                for c in node.children:
                    walk(c, depth + 1)

        walk(self.root, 0)
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Hash table: region name -> BPlusTree (separate chaining)
# ----------------------------------------------------------------------
class RegionIndex:
    def __init__(self, buckets=7, max_keys=64):
        self.nbuckets = buckets
        self.max_keys = max_keys
        self.buckets = [[] for _ in range(buckets)]   # each bucket: list of [name, tree]

    def bucket_of(self, region):                      # djb2 hash
        h = 5381
        for ch in region:
            h = (h * 33 + ord(ch)) & 0xFFFFFFFF
        return h % self.nbuckets

    def tree(self, region, create=False):
        chain = self.buckets[self.bucket_of(region)]
        for name, t in chain:
            if name == region:
                return t
        if not create:
            return None
        t = BPlusTree(self.max_keys)
        chain.append([region, t])
        return t

    def insert(self, region, key, rec):
        return self.tree(region, create=True).insert(key, rec)

    def delete(self, region, key):
        t = self.tree(region)
        return t.delete(key) if t else False

    def search(self, region, key):
        t = self.tree(region)
        return t.search(key) if t else None

    def range_query(self, region, start, end):
        t = self.tree(region)
        return t.range_query(start, end) if t else ([], None)

    def season_query(self, region, season, year_from, year_to):
        """Region + season over several years: one range query per year, each O(log n + k).
        Returns one dict per year: {"year", "records" [(key, record)...], "stats"}."""
        first_month, last_month = SEASONS[season]
        out = []
        for year in range(year_from, year_to + 1):
            start = year * 10000 + first_month * 100 + 1
            end = year * 10000 + last_month * 100 + 31     # day 31 is only an upper bound
            records, stats = self.range_query(region, start, end)
            out.append({"year": year, "records": records, "stats": stats})
        return out

    def regions(self):
        return sorted(name for chain in self.buckets for name, _ in chain)

    def bucket_table(self):
        return [(i, [name for name, _ in chain]) for i, chain in enumerate(self.buckets)]


# ----------------------------------------------------------------------
# Glue between the DataFrame and the index
# ----------------------------------------------------------------------
def fmt_key(key):
    d, city = key
    return f"{d // 10000}-{d // 100 % 100:02d}-{d % 100:02d} {city}"


def date_ints(dates):
    """pandas datetime Series -> YYYYMMDD ints."""
    return dates.dt.year * 10000 + dates.dt.month * 100 + dates.dt.day


def city_labels(df):
    """City name if the data has a city/station column, otherwise 'lat_lon'."""
    col = next((c for c in df.columns if str(c).lower() in ("city", "station")), None)
    fallback = [f"{a:.2f}_{b:.2f}" for a, b in zip(df["lat"], df["lon"])]
    if col is None:
        return fallback
    names = df[col].astype("string").fillna("").str.strip().tolist()
    return [n if n else f for n, f in zip(names, fallback)]


def build_index(df, max_keys=64):
    """Insert every row into the RegionIndex. Returns (index, inserted, duplicates)."""
    n = len(df)
    dates = date_ints(df["date"]).tolist()
    cities = city_labels(df)
    regions = df["region"].tolist()
    lat, lon, tmax = df["lat"].tolist(), df["lon"].tolist(), df["tmax"].tolist()
    tmin = df["tmin"].tolist() if "tmin" in df.columns else [nan] * n
    rain = df["Rain"].tolist() if "Rain" in df.columns else [nan] * n

    index = RegionIndex(max_keys=max_keys)
    inserted = 0
    for i in range(n):
        rec = (tmax[i], tmin[i], rain[i], lat[i], lon[i])
        if index.insert(regions[i], (dates[i], cities[i]), rec):
            inserted += 1
    return index, inserted, n - inserted
