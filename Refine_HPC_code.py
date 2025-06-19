# This would take the best previous function and then split each step into 5 peices and then run a refinement process in order to get a better value

import numpy as np
import cupy as cp
import random
import time

# Set the resolution/size of the step function
size = 500

# Initialize a base step function and downsample it by averaging every 5 steps
V = [0.04771739818193131, 0.05881805942016414, 0.06535585523561711, 0.06837645420840573, 0.07457204572205384, 0.07113704557332491, 0.07778115289373695, 0.07189991307822083, 0.07102862114492596, 0.06631421103641831, 0.06731447014841395, 0.05369840378351274, 0.05948844251892857, 0.05605458037963489, 0.05503879750230377, 0.049492758049705696, 0.05451396832447118, 0.05044886706523308, 0.0529542303789405, 0.048276450790399425, 0.056417100018719184, 0.05475475115397932, 0.04925973599034347, 0.05743779602777774, 0.05771982710314894, 0.052835026628888136, 0.057999841130758574, 0.05670612117829514, 0.05347059615477266, 0.05732507791446663, 0.05498148283130224, 0.05465583410369003, 0.0568227421430412, 0.05799708838027596, 0.056145318406758245, 0.060252811069254085, 0.059343739521247704, 0.062249311881643864, 0.06643046751828316, 0.06918359063287563, 0.06328261626442189, 0.06377316379545432, 0.06531438954368075, 0.06652596210267361, 0.06625736197088967, 0.0634331183453876, 0.06102337645570334, 0.06667625348413087, 0.06553505131646731, 0.06818560592065678, 0.07015589395266963, 0.07208781263082227, 0.07592231341495069, 0.0767309732348442, 0.07744025077132505, 0.07741569923507145, 0.07959304552197072, 0.08139029387674304, 0.08069434166463846, 0.0829757522354276, 0.08524768029530207, 0.08292927469010757, 0.0862178979238204, 0.08795755356579446, 0.09016356753019304, 0.09261999105989441, 0.09364022826884012, 0.09624211638655275, 0.0995824375232355, 0.09777017026648381, 0.10260566420321567, 0.10333239366723891, 0.104850190572525, 0.10657169882125242, 0.1102712960931007, 0.11252465324417041, 0.11396675747582845, 0.1199171798444524, 0.12205817292431517, 0.13066165073966998, 0.1348884957777485, 0.1424940762271107, 0.1488056667416703, 0.1541091944051, 0.15976275575911036, 0.16457615494316966, 0.17057455418243078, 0.17373611298690458, 0.17439898498869136, 0.17205127574364615, 0.1681718412714955, 0.15824404923607338, 0.14604311292460193, 0.12919721266745396, 0.11278154185993666, 0.0903068729323169, 0.06862708647918295, 0.04606158390800534, 0.028038877298072692, 0.014076730150655133]  # predefined list of values
V = np.array(V)
j = np.floor(np.arange(0,size,1)/5).astype(np.int64)
V = V[j]  # make V have 'size' entries by repeating each block of 5 values

# Precompute index and mask arrays on GPU to accelerate repeated computation
why1 = cp.resize(cp.array(True), (size, size*2))
y1 = cp.arange(0, size, 1)
y1 = cp.resize(y1, (size*2, size)).transpose()
x1 = cp.arange(0, size*2, 1)
x1 = cp.resize(x1, (size, size*2))
why1 = cp.where(x1-y1<=0,False,why1)
why1 = cp.where(x1-y1>size,False,why1)
ind1 = cp.resize(cp.where(why1, x1-y1-1, 0), (size, size, size*2))
why1 = cp.resize(why1, (size, size, size*2))
y1 = cp.resize(y1, (size, size, size*2))
a1 = cp.arange(0, size, 1)
a1 = cp.resize(a1, (size*2, size, size)).transpose()
b1 = cp.arange(0, size, 1)
b1 = cp.resize(b1, (size*2, size)).transpose()

# CPU version of the expression computation
def Big_F(v):
    v = np.array(v)
    why = np.resize(np.array(True), (np.size(v), np.size(v)*2))
    y = np.arange(0, np.size(v), 1)
    y = np.resize(y, (np.size(v)*2, np.size(v))).transpose()
    x = np.arange(0, np.size(v)*2, 1)
    x = np.resize(x, (np.size(v), np.size(v)*2))
    why = np.where(x-y<=0,False,why)
    why = np.where(x-y>np.size(v),False,why)
    ind = np.where(why, x-y-1, 0)
    val = v[y.astype(np.int64)] * np.where(why, v[ind.astype(np.int64)], 0)
    l = np.sum(val, axis=0)
    l = np.concatenate((l, np.array([0])))
    i = np.arange(0, np.size(l)-1, 1)
    v1 = l[(i+1).astype(np.int64)] - l[i.astype(np.int64)]
    v2 = l[i.astype(np.int64)]
    return 2*np.sum(((v1**2)/3) + v1*v2 + v2**2)/(np.max(l) * np.sum(l[i.astype(np.int64)] + l[(i+1).astype(np.int64)]))

# GPU-accelerated version to evaluate a batch of inputs
def NBig_F(v, a, b, y, ind, why):
    v = cp.array(v)
    val = v[(a.astype(cp.int64), y.astype(cp.int64))] * cp.where(why, v[(a.astype(cp.int64), ind.astype(cp.int64))], 0)
    l = cp.sum(val, axis=1)
    i = cp.arange(0, cp.shape(l)[1], 1)
    i1 = cp.where(i+1 < cp.shape(l)[1], i+1, 0)
    v1 = l[(b.astype(cp.int64), (i1).astype(cp.int64))] - l[(b.astype(cp.int64), i.astype(cp.int64))]
    v2 = l[(b.astype(cp.int64), i.astype(cp.int64))]
    return 2 * cp.sum(((v1**2)/3) + v1*v2 + v2**2,axis=1)/(cp.max(l, axis=1) * cp.sum(l[(b.astype(cp.int64), i.astype(cp.int64))] + l[(b.astype(cp.int64), (i1).astype(cp.int64))], axis=1))

# Main optimization loop
for scale in range(1000000):
    start = time.time()

    # Prepare gradient directions: each basis vector has a single 1e-7 at a different position
    vc = []
    for i in range(size):
        vc.append(np.array([0] * i + [0.0000001] + [0] * (size-1-i)))
    vc = cp.array(vc)

    # Evaluate initial score
    init = Big_F(V)

    # Gradient ascent loop
    for i in range(1000):
        print(str(i) + ": " + str(init))
        
        # Slightly perturb each dimension of V to estimate gradient
        Vs = cp.resize(cp.array(V), (size, size)) + vc

        # Compute function values for each perturbed vector
        grad = NBig_F(Vs, a1, b1, y1, ind1, why1).get() - init

        # Normalize the gradient and take a small step in its direction
        grad = grad/np.linalg.norm(grad)
        V += grad * (0.01/(((i+1)**(1/4)) * ((scale+1) ** (1/3)))) * (1 - init)
        V = np.where(V >=0, V, 0)  # Ensure V remains non-negative

        # Update score
        init = Big_F(V)

    end = time.time()

    # Save optimized function and metadata
    text = open(r"\usr\bin\share\autoconvolution\cwboyer\functions_final\f" + str(scale) + ".py", "w")
    text.write("v = " + str(V.tolist()))
    text.write(f"\n")
    text.write("value = " + str(init))
    text.write(f"\n")
    text.write("time = " + str(end-start))
    text.close()
