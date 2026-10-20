import random, bisect, sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from bplus_index import BPlusTree, _Leaf, _Internal, RegionIndex

def check(t):
    # invariants: sorted keys, uniform depth, node sizes, leaf chain == in-order leaves, count
    depths=set(); leaves=[]
    def walk(n,d,lo,hi,is_root):
        if isinstance(n,_Leaf):
            depths.add(d); leaves.append(n)
            assert n.keys==sorted(n.keys) and len(set(n.keys))==len(n.keys)
            assert len(n.keys)==len(n.recs)
            if not is_root: assert len(n.keys)>=t.min_keys,("leaf underfull",len(n.keys))
            assert len(n.keys)<=t.max_keys
            for k in n.keys:
                assert (lo is None or k>=lo) and (hi is None or k<hi),("leaf range",k,lo,hi)
        else:
            assert len(n.children)==len(n.keys)+1
            assert n.keys==sorted(n.keys)
            if not is_root: assert len(n.keys)>=t.min_keys,("internal underfull",len(n.keys))
            else: assert len(n.keys)>=1
            assert len(n.keys)<=t.max_keys
            for i,c in enumerate(n.children):
                l=lo if i==0 else n.keys[i-1]; h=hi if i==len(n.keys) else n.keys[i]
                walk(c,d+1,l,h,False)
    walk(t.root,0,None,None,True)
    assert len(depths)==1,("unbalanced",depths)
    # chain
    chain=[]; n=t.root
    while isinstance(n,_Internal): n=n.children[0]
    while n: chain.append(n); n=n.next
    assert chain==leaves,"leaf chain mismatch"
    assert sum(len(l.keys) for l in leaves)==t.count,("count",t.count)

random.seed(7)
bad=0
for mk in (3,4,5,8,64):
    t=BPlusTree(mk); ref={}
    cities=['A','B','C','D']
    for step in range(6000):
        op=random.random()
        k=(random.randint(20000101,20000101+400), random.choice(cities))
        if op<0.55:
            ins=t.insert(k,step); assert ins==(k not in ref)
            if k not in ref: ref[k]=step
        elif op<0.85:
            d=t.delete(k); assert d==(k in ref)
            ref.pop(k,None)
        else:
            assert t.search(k)==ref.get(k)
        if step%250==0:
            check(t)
            a=random.randint(20000101,20000500); b=a+random.randint(0,60)
            out,_=t.range_query(a,b)
            exp=sorted((kk,v) for kk,v in ref.items() if a<=kk[0]<=b)
            assert [(k,r) for k,r in out]==exp,("range mismatch",mk,a,b)
    check(t)
    # delete everything
    for k in list(ref): assert t.delete(k)
    check(t); assert t.count==0 and t.height()==1
    print("max_keys",mk,"OK")
# sequential (time-series) load then range
t=BPlusTree(64)
for d in range(100000): t.insert((d,'x'),d)
check(t); print('sequential 100k height',t.height(),'leaves',t.leaf_count())
out,s=t.range_query(500,529); print(len(out),s)
# RegionIndex
ri=RegionIndex()
for r in ['NW India','NE India','Central India','East Coast','West Coast','South Peninsula','Himalayan Region']:
    ri.insert(r,(20200101,'c'),1)
print(ri.bucket_table())
