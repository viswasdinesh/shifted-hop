"""Pure LTV test: no DDP, no physical system. Random time-varying A_k,B_k,Q_k.
Sweep lambda_min(Q_k) from PD -> singular PSD -> indefinite.
Four methods against a per-horizon backward-Riccati reference."""
import numpy as np, sys
sys.path.insert(0,'.')
from lqr import hop_horizon_search
from shifted_chart import _sym, shifted_horizon_search
from verify_cartpole import riccati_reference

rng=np.random.default_rng(7)

def make_Q(n, lam_min, rank_def=0):
    ev=np.abs(rng.normal(size=n))*3+1.0
    ev[0]=lam_min
    for i in range(1,rank_def+1): ev[i]=0.0
    U,_=np.linalg.qr(rng.normal(size=(n,n)))
    return _sym(U@np.diag(ev)@U.T)

def reg_then_hop(A,B,Q,R,z0,QT,T,eps0=1e-6):
    """Algorithm-2 style: add mu*I, escalating x10, until PD. Then stock HOP."""
    Qr=[]
    for M in Q:
        mu=0.0; M2=_sym(M)
        while np.linalg.eigvalsh(M2+mu*np.eye(M.shape[0]))[0]<=1e-9:
            mu=eps0 if mu==0 else mu*10
        Qr.append(_sym(M)+mu*np.eye(M.shape[0]))
    QTr=[]
    for M in QT:
        mu=0.0; 
        while np.linalg.eigvalsh(_sym(M)+mu*np.eye(M.shape[0]))[0]<=1e-9:
            mu=eps0 if mu==0 else mu*10
        QTr.append(_sym(M)+mu*np.eye(M.shape[0]))
    return hop_horizon_search(A,B,Qr,R,z0,QTr,T)

n,m,T=6,3,60
A=[np.eye(n)+0.15*rng.normal(size=(n,n)) for _ in range(T)]
B=[rng.normal(size=(n,m)) for _ in range(T)]
R=_sym(rng.normal(size=(m,m))); R=R@R.T+np.eye(m)
Qf=_sym(rng.normal(size=(n,n))); Qf=Qf@Qf.T+5*np.eye(n)
QT=[Qf]*T; z0=rng.normal(size=n)

print(f"LTV: n={n} m={m} T={T}, random A_k,B_k. Ref = per-horizon backward Riccati.\n")
hdr=f"{'lam_min(Q)':>12} {'rank def':>9} | {'HOP':>11} {'T*':>4} | {'reg+HOP':>11} {'T*':>4} | {'shifted':>11} {'T*':>4}"
print(hdr); print("-"*len(hdr))
cases=[(1.0,0),(1e-3,0),(1e-6,0),(0.0,0),(0.0,2),(0.0,4),(-1e-3,0),(-1.0,0),(-10.0,0)]
for lam,rd in cases:
    Q=[make_Q(n,lam,rd) for _ in range(T)]
    Jr=riccati_reference(A,B,Q,R,z0,QT,T)
    row=f"{lam:>12.3g} {rd:>9d} |"
    for name,fn in [("hop",lambda: hop_horizon_search(A,B,Q,R,z0,QT,T)),
                    ("reg",lambda: reg_then_hop(A,B,Q,R,z0,QT,T)),
                    ("shf",lambda: shifted_horizon_search(A,B,Q,R,z0,QT,T)[0])]:
        try:
            J=fn(); e=np.abs(J-Jr)/np.maximum(np.abs(Jr),1e-12)
            row+=f" {e.max():>11.2e} {int(np.argmin(J))+1:>4} |"
        except Exception as ex:
            row+=f" {type(ex).__name__[:11]:>11} {'--':>4} |"
    print(row)
print(f"\nreference T* = {int(np.argmin(riccati_reference(A,B,[make_Q(n,1.0,0) for _ in range(T)],R,z0,QT,T)))+1} (PD case)")
