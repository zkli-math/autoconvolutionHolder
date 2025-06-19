# Script that CB used on Hazel 2 to calculate the value of the expression for many random step functions

# CB would run this code for about 48 hours on the supercomputer to get some initial data:

import numpy as np
import cupy as cp
import time

# Computes a custom metric over a vector v using NumPy
def Big_F(v):
    v = np.array(v)
    # Initialize a boolean mask of shape (len(v), 2*len(v)) all True
    why = np.resize(np.array(True), (np.size(v), np.size(v)*2))
    # Generate y matrix (broadcasted column indices)
    y = np.arange(0, np.size(v), 1)
    y = np.resize(y, (np.size(v)*2, np.size(v))).transpose()
    # Generate x matrix (broadcasted row indices)
    x = np.arange(0, np.size(v)*2, 1)
    x = np.resize(x, (np.size(v), np.size(v)*2))
    # Set mask to False where x-y <= 0 or x-y > len(v)
    why = np.where(x-y<=0,False,why)
    why = np.where(x-y>np.size(v),False,why)
    # Compute indices to look up in vector v
    ind = np.where(why, x-y-1, 0)
    # Compute values using broadcasting and the mask
    val = v[y.astype(np.int64)] * np.where(why, v[ind.astype(np.int64)], 0)
    # Sum across rows
    l = np.sum(val, axis=0)
    # Append 0 to the end for interval-based calculation
    l = np.concatenate((l, np.array([0])))
    i = np.arange(0, np.size(l)-1, 1)
    v1 = l[(i+1).astype(np.int64)] - l[i.astype(np.int64)]
    v2 = l[i.astype(np.int64)]
    # Return the custom normalized metric
    return 2*np.sum(((v1**2)/3) + v1*v2 + v2**2)/(np.max(l) * np.sum(l[i.astype(np.int64)] + l[(i+1).astype(np.int64)]))

size = 100
att = 40

# GPU-accelerated version of Big_F that evaluates multiple step functions at once
def NBig_F(v, a, b, y, ind, why):
    v = cp.array(v)
    # Multiply v[y] * v[ind] where mask is True
    val = v[(a.astype(cp.int64), y.astype(cp.int64))] * cp.where(why, v[(a.astype(cp.int64), ind.astype(cp.int64))], 0)
    # Sum across axis 1 (equivalent to rows in this setup)
    l = cp.sum(val, axis=1)
    # Define intervals
    i = cp.arange(0, cp.shape(l)[1], 1)
    i1 = cp.where(i+1 < cp.shape(l)[1], i+1, 0)
    # Compute interval differences and values
    v1 = l[(b.astype(cp.int64), (i1).astype(cp.int64))] - l[(b.astype(cp.int64), i.astype(cp.int64))]
    v2 = l[(b.astype(cp.int64), i.astype(cp.int64))]
    # Return the custom normalized metric for each row
    return 2 * cp.sum(((v1**2)/3) + v1*v2 + v2**2,axis=1)/(cp.max(l, axis=1) * cp.sum(l[(b.astype(cp.int64), i.astype(cp.int64))] + l[(b.astype(cp.int64), (i1).astype(cp.int64))], axis=1))

# Precompute GPU tensors for reuse
why1 = cp.resize(cp.array(True), (size, size*2))
y1 = cp.arange(0, size, 1)
y1 = cp.resize(y1, (size*2, size)).transpose()
x1 = cp.arange(0, size*2, 1)
x1 = cp.resize(x1, (size, size*2))
# Apply same logic as in Big_F to generate the boolean mask
why1 = cp.where(x1-y1<=0,False,why1)
why1 = cp.where(x1-y1>size,False,why1)
# Compute index matrix
ind1 = cp.resize(cp.where(why1, x1-y1-1, 0), (att, size, size*2))
# Resize tensors for batch processing
why1 = cp.resize(why1, (att, size, size*2))
y1 = cp.resize(y1, (att, size, size*2))
# Create a and b indices for batch access
a1 = cp.arange(0, att, 1)
a1 = cp.resize(a1, (size*2, size, att)).transpose()
b1 = cp.arange(0, att, 1)
b1 = cp.resize(b1, (size*2, att)).transpose()

re = 0
while True:
    Start = time.time()
    # Initialize a random vector and normalize it
    V = np.random.rand(size)
    V = V/cp.sum(V)
    # Evaluate its score
    rec = Big_F(V)
    print(rec)

    # First optimization phase (coarse adjustment)
    S = 25
    start = time.time()
    for i in range(1000):
        S = S * 0.999
        # Generate new batch of noisy step functions around V
        v = cp.resize(V, (att, size)) + S * (cp.resize(cp.random.rand(att * size), (att, size)) - 0.5)
        v = cp.abs(v)  # Keep values positive
        # Evaluate the new batch
        score = NBig_F(v, a1, b1, y1, ind1, why1).get()
        ind = np.argmax(score)
        # If a better result is found, update
        if score[ind] > rec:
            rec = score[ind]
            V = v.get()[ind]
            V = V/cp.sum(V)
    end = time.time()
    print(str(-1) + ": " + str(end - start) + ": " + str(rec) + ": " + str(S))

    # Second optimization phase (finer adjustments)
    S = 0.05
    for j in range(400):
        start = time.time()
        for i in range(1000):
            S = S * 0.99998
            v = cp.resize(V, (att, size)) + S * (cp.resize(cp.random.rand(att * size), (att, size)) - 0.5)
            v = cp.abs(v)
            score = NBig_F(v, a1, b1, y1, ind1, why1).get()
            ind = np.argmax(score)
            if score[ind] > rec:
                rec = score[ind]
                V = v.get()[ind]
                V = V/cp.sum(V)
        end = time.time()
        print(str(j) + ": " + str(end - start) + ": " + str(rec) + ": " + str(S))

    # Save the best result found in this loop to file
    End = time.time()
    text = open(r"\usr\bin\share\autoconvolution\cwboyer\functions100\f" + str(re) + ".py", "w")
    text.write("v = " + str(V.tolist()))
    text.write(f"\n")
    text.write("value = " + str(rec))
    text.write(f"\n")
    text.write("time = " + str(End-Start))
    text.close()
    re += 1
