"""Pure Python native category-equality splits; never order nominal codes.

Numeric operations delegate unchanged to the qualified reference backend.
Missing and unseen nominal values have separate explicit codes. Each node tests
one category versus the remainder; arbitrary subsets/target ordering are absent.
"""
from array import array
from bisect import bisect_right
import math
from .backend import PythonBackend


class Column(array):
    def __new__(cls,kind,values,nominal=False):return super().__new__(cls,kind,values)
    def __init__(self,kind,values,nominal=False):self.nominal=nominal


class Hist(list):
    def __init__(self,values,nominal=False):super().__init__(values);self.nominal=nominal


class ScalableBackend(PythonBackend):
    name='private-python-columnar-nominal'
    def __init__(self,nominal_sizes=None):self.nominal_sizes=nominal_sizes or {}
    def bin(self,binner,X):
        if not hasattr(X,'_compact_columns'):return super().bin(binner,X)
        out=[]
        for j,cuts in enumerate(binner.thresholds):
            if j in self.nominal_sizes:out.append(Column('I',X.column(j),True))
            else:out.append(Column('B' if binner.max_bins<=255 else 'H',(0 if math.isnan(v) else bisect_right(cuts,v)+1 for v in X.column(j))))
        return out
    def node_statistics(self,Xb,ids,g,h,u,features,need_missing,need_uncertainty):
        if not need_missing or not self.nominal_sizes:
            return super().node_statistics(Xb,ids,g,h,u,features,need_missing,need_uncertainty)
        # Numeric missing is bin0; native nominal missing is fitted k+1.
        # Unseen nominal k+2 is separate and must not trigger this local gate.
        columns=tuple((Xb[f],self.nominal_sizes[f]+1 if f in self.nominal_sizes else 0) for f in features)
        missing=sum(any(column[i]==code for column,code in columns) for i in ids)/max(len(ids),1)
        G,H,uncertainty,_=super().node_statistics(Xb,ids,g,h,u,features,False,need_uncertainty)
        return G,H,uncertainty,missing
    def histogram(self,column,*args):return Hist(super().histogram(column,*args),getattr(column,'nominal',False))
    def subtract(self,parent,child):return Hist(super().subtract(parent,child),getattr(parent,'nominal',False))
    def fast_split(self,hist,total,depth,genome,params):
        if not getattr(hist,'nominal',False):return super().fast_split(hist,total,depth,genome,params)
        if genome.split_program not in (('OBJECTIVE_GAIN',),('OBJECTIVE_GAIN','BALANCE','MUL')):raise ValueError('Unsupported private nominal split program')
        n,G,H=total[:3];counts,gs,hs=hist[:3];l1=params.l1
        parent_l2=params.l2*(1+(params.depth_reg_strength*depth if genome.regularization_policy=='depth_scaled' else 0))
        child_l2=params.l2*(1+(params.depth_reg_strength*(depth+1) if genome.regularization_policy=='depth_scaled' else 0))
        def score(g,h,l2):return max(abs(g)-l1,0.)**2/max(h+l2,1e-12)
        parent=score(G,H,parent_l2);best=None
        for code in range(1,len(counts)):
            ln,lg,lh=counts[code],gs[code],hs[code];rn,rg,rh=n-ln,G-lg,H-lh
            if min(ln,rn)<params.min_samples_leaf or min(lh,rh)<params.min_hessian:continue
            gain=.5*(score(lg,lh,child_l2)+score(rg,rh,child_l2)-parent)
            if len(genome.split_program)>1:gain*=2*min(ln,rn)/n
            if math.isfinite(gain) and gain>params.min_gain and (best is None or round(gain,12)>round(best[0],12)):best=(gain,code,True)
        return best
    def partition(self,column,ids,cut,missing_left):
        if not getattr(column,'nominal',False):return super().partition(column,ids,cut,missing_left)
        left=[];right=[]
        for i in ids:(left if column[i]==cut else right).append(i)
        return left,right
    def predict_tree(self,nodes,Xb):
        out=[]
        for i in range(len(Xb[0])):
            k=0
            for _ in range(len(nodes)+1):
                node=nodes[k]
                if node.is_leaf:out.append(node.value);break
                col=Xb[node.feature];b=col[i]
                left=(b==node.threshold_bin) if getattr(col,'nominal',False) else node.missing_left if b==0 else b<=node.threshold_bin
                k=node.left if left else node.right
            else:raise ValueError('Invalid cyclic native tree')
        return out
