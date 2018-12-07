import numpy as np
N = int(1e6)

m = [0 if i < 0.5 else 1 for i in np.random.random(N)]
def p(n):
    shift = [-1, 1, -1, -1, 1, -1, 1]
    code = [0 if shift[i%7]==-1 else 1 for i in range(n)]
    return code

n = p(N)

Pm1 = sum(m)/N
Pm0 = 1-Pm1
Pn1 = sum(n)/N
Pn0 = 1-Pn1
P00 = sum([i==0 and j==0 for i,j in zip(m,n)])/N
P01 = sum([i==0 and j==1 for i,j in zip(m,n)])/N
P10 = sum([i==1 and j==0 for i,j in zip(m,n)])/N
P11 = sum([i==1 and j==1 for i,j in zip(m,n)])/N

dis = (P00-Pm0*Pn0)**2/(Pm0*Pn0) + (P01-Pm0*Pn1)**2/(Pm0*Pn1) + (P10-Pm1*Pn0)**2/(Pm1*Pn0) + (P11-Pm1*Pn1)**2/(Pm1*Pn1)
dis1 = dis*N
print('Pm0={},Pm1={},Pn0={},Pn1={}'.format(Pm0,Pm1,Pn0,Pn1))
print('P00={},P01={},P10={},P11={}'.format(P00,P01,P10,P11))
print('dis={},dis1={}'.format(dis,dis1))
print('DONE')
