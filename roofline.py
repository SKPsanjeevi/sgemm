import matplotlib.pyplot as plt
import numpy as np


# NVIDIA GeForce RTX 3050 Mobile - specs
# https://www.techpowerup.com/gpu-specs/geforce-rtx-3050-mobile.c3788
fp32Flops = 5.501e12                    # FLOPs
bwBps = 192e9                           # Bytes/s

# Different kernels and measured time in ms
data = np.array([
    ('Naive',               71),
    ('Coalesced',            8.04),
    ('Shared Mem',           6.54),
    ('1D register tiling',   2.064),
    ('2D register tiling',   1.394),
    ('CUBLAS',               0.802)
], dtype=[('Version', 'U30'), ('Time(ms)', 'f4')])

# Draw roofline with Performance (FLOPs) and Bandwidth (Bytes/s) ceiling
exponent = 4
x = np.logspace(0.0, exponent, num=4)   # FLOPs/Byte
xmin = 0
xmax = 10**exponent
yBW = x * bwBps

fig = plt.figure(figsize=(8, 5))
plt.hlines(y=fp32Flops, xmin=0, xmax=xmax, colors='r')          # Compute bound line
plt.text((xmin+xmax)/2, fp32Flops, "Compute Bound", ha = 'right', va = 'bottom')
plt.loglog(x, yBW)                                              # Memory bound line

plt.xlim(0, xmax)
plt.ylim(0, 1e13)
plt.xlabel('Arithmetic Intensity [FLOPs/Byte]')
plt.ylabel('Performance [FLOPs]')


# Plot different kernel performance
fp32Size    = 4 # Bytes
M = N = P = 1024
AmatrixSize = M * P
BmatrixSize = P * N
CmatrixSize = M * N
sizeTotal   = AmatrixSize + BmatrixSize + CmatrixSize
totalMem    = sizeTotal * fp32Size      # total size of matrices in Bytes
totalFLOP   = M * N * (2*P)

FLOPs       = totalFLOP / (data['Time(ms)'] * 1e-3)
# print(FLOPs)
FLOPperByte = FLOPs / bwBps
# print(FLOPperByte)
plt.plot(FLOPperByte, FLOPs, 'r+')

for i in range(len(FLOPs)):
    plt.text(FLOPperByte[i], FLOPs[i], f"  {data['Version'][i]}", ha='left', va='center_baseline')

plt.savefig('roofline.png')
plt.show()
