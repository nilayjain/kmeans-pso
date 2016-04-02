import csv
import copy
import math
import numpy as np
import random
import numpy.random as nprand
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import time
from profilehooks import profile
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from numpy import genfromtxt


def populate_essentials(fname, hyper_to_d):
    rows = genfromtxt(fname, delimiter=',')
    n = len(rows)
    hypercube_dim_float = math.log(n,2)
    #print "hypercube_dim_float " + str(hypercube_dim_float)
    hypercube_dim = math.ceil(hypercube_dim_float)
    #print "hypercube_dim " + str(hypercube_dim)
    n = 0
    for row in rows:
        row = row[:-1]
        hyper_to_d[n] = np.array(row)
        n = n+1
    return (hypercube_dim, n)

# particles is the indices of the current particles

def initialize(num_clusters, hyper_to_d, particles_h_to_d, particles, hypercube_dim, velocity):
    for i in range(num_clusters):
        x = random.choice(hyper_to_d.keys())
        particles_h_to_d[x] = hyper_to_d[x]
        particles[i] =  x
        x = hypercube_dim
        a = random.randrange(0,x)
        velocity[i] = int(a)

# calculate the global fitness function
def global_fitness(particles, hyper_to_d):
    arr = []
    for i in xrange(len(particles)):
        for j in xrange(i, len(particles)):
            arr.append((hyper_to_d[particles[i]]-hyper_to_d[particles[j]])**2)
    return np.sum(arr)

# calculate the local fitness function
def local_fitness(p, particles_h_to_d):
    #p is an integer ie. index of the required particle
    key = particles_h_to_d[p]
    arr = []
    for p1 in particles_h_to_d.keys():
        arr.append((particles_h_to_d[p1] - key)**2)
    return np.sum(arr)

def move(particle, vel, hyper_to_d, particles_h_to_d, dim):
    '''d = np.array([particle])
    bin = (((d[:,None] & (1 << np.arange(dim)))) > 0).astype(int)
    ones = np.where(bin>0)[1]'''
    particle = bin(particle)[2:]
    diff1 = (int)(dim - len(particle))
    lis = ''
    for i in range(diff1):
        lis+='0'
    lis+=particle
    particle = lis

    if vel < 0:
        vel = -vel

    diff = nprand.randint(dim, size = vel)
    out = particle
    for i in diff:
        if out[i] == '0':
            out = out[:i] + '1' + out[i+1:]
        else:
            out = out[:i] + '0' + out[i+1:]
    num = int(out, 2)
    ones = np.where
    while num not in hyper_to_d and num not in particles_h_to_d:
        if vel < 0:
            vel = -vel
        diff = nprand.randint(dim, size = vel)
        out = particle
        for i in diff:
            if out[i] == '0':
                out = out[:i] + '1' + out[i+1:]
            else:
                out = out[:i] + '0' + out[i+1:]
        num = int(out,2)
    return int(out, 2)

def edit_distance(p1, p2, hypercube_dim):
    #arr is an array
    pos1 = bin(p1)[2:]
    pos2 = bin(p2)[2:]
    dist = abs(len(pos1) - len(pos2))
    for i in range(min(len(pos1), len(pos2))):
        if pos1[i] != pos2[i]:
            dist+=1
    return dist

def pso(iterations, num_clusters, num_particles, hyper_to_d, particles_h_to_d, hypercube_dim, particles, velocity):
    i =0
    gbest = np.zeros(num_clusters, dtype = "int")
    pbest = np.zeros(num_clusters, dtype = "int")

    np.copyto(gbest, particles)
    np.copyto(pbest, particles)

    best_value =  global_fitness(gbest, hyper_to_d)
    best_local_value = []
    bests = []

    for j in range(num_clusters):
        best_local_value.append(local_fitness(particles[j] , particles_h_to_d))

    while i < iterations:

        for j in xrange(num_clusters):

            particles_h_to_d.pop(particles[j], None)
            #print "particles[j] = " + str(particles[j])
            val = hyper_to_d[particles[j]]

            velocity[j] = int(velocity[j] + nprand.uniform(-1,1) * edit_distance(pbest[j], particles[j], hypercube_dim) + nprand.uniform(-1,1) * edit_distance(gbest[j], particles[j], hypercube_dim))

            particles[j] = move(particles[j] , velocity[j], hyper_to_d, particles_h_to_d,hypercube_dim)

            particles_h_to_d[particles[j]] = hyper_to_d[particles[j]]
            gfit = global_fitness(particles, hyper_to_d)

            if(gfit > best_value):
                print "improved from " + str(best_value) + " to " + str(global_fitness(particles, hyper_to_d))
                best_value = gfit
                gbest[j] = particles[j]
                new = np.zeros(num_clusters, dtype = "int")
                np.copyto(new, gbest)
                bests.append(new)

            pfit = local_fitness(particles[j], particles_h_to_d)
            if pfit > best_local_value[j]:
                #print "improved pbest"
                best_local_value[j] = pfit
                pbest[j] = particles[j]
        i = i+1
    return (bests, pbest, best_value)


def driver(fname, cluster, iterations):
    num_clusters = cluster
    hyper_to_d = {}
    particles_h_to_d = {}
    particles = np.zeros(num_clusters, dtype = 'int')
    hypercube_dim, num_particles = populate_essentials(fname, hyper_to_d)
    k = 0
    #print hyper_to_d
    velocity = np.zeros(num_clusters, dtype = 'int')


    initialize(num_clusters, hyper_to_d, particles_h_to_d, particles, hypercube_dim, velocity)

    best_value = global_fitness(particles, particles_h_to_d)
    #print "particles "+ particles[0]
    #print  velocity[0]
    #print move('11111111', velocity[0], num_particles)
    #print particles

    start = time.time()
    (bests, pbest, best_value) = pso(iterations, num_clusters, num_particles, hyper_to_d,  particles_h_to_d, hypercube_dim, particles, velocity)
    end = time.time()
    print "time to initialize = "
    print end-start
    #print str(sum(best_local_value)/2) + " local value"

    f = open(fname)
    f.readline()  # skip the header
    X = np.genfromtxt(f, delimiter = ',')
    y = X[:,-1]
    y = y.astype(np.int)
    #print max(y)
    X = np.delete(X, -1, 1)

    # iris = datasets.load_iris()
    # X = iris.data
    # y = iris.target
    # print y

    estimators = {'kmeans++': KMeans(n_clusters=cluster, n_jobs = -1),
                  'k_means_bad_init': KMeans(n_clusters=cluster, n_init=10,
                                                  init='random', n_jobs = -1)
                }

    # print "bests[0] = " + str(bests[0])
    # print hyper_to_d[bests[0][0]]
    for i in range(len(bests)):
        lis = []
        for j in bests[i]:
            lis.append(hyper_to_d[j])
        estimators["kmeans_pso_"+str(i)] =  KMeans(n_clusters=cluster, n_init = 1, init=np.asarray(lis), n_jobs = -1)

    #estimators['k_means_pso_last' + str(i)] =  KMeans(n_clusters=9, n_init = 1, init=np.asarray(hyper_to_d[]), n_jobs = -1)


    #np.random.seed(5)
    fignum = 1
    for name, est in estimators.items():
        start = time.time()
        est.fit(X)
        labels = est.labels_
        print str(-est.score(X)) + " score of kmeans " + name
        end = time.time()
        print (end-start)

        #print "silhouette_score = "+str(metrics.silhouette_score(X, labels, metric='euclidean'))
        fig = plt.figure(name, figsize=(10,9))
        plt.clf()
        ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=40, azim=134)

        ax.scatter(X[:, 5], X[:, 6], X[:, 10], c=labels.astype(np.float))

        ax.w_xaxis.set_ticklabels([])
        ax.w_yaxis.set_ticklabels([])
        ax.w_zaxis.set_ticklabels([])
        ax.set_xlabel('dim1')
        ax.set_ylabel('dim2')
        ax.set_zlabel('dim3')
        fignum = fignum + 1

    # Plot the ground truth
    fig = plt.figure("ground truth", figsize=(4, 3))
    plt.clf()
    ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=40, azim=134)

    plt.cla()

    for name, label in [('0', 0),
                        ('1', 1)]:
        ax.text3D(X[y == label, 5].mean(),
                  X[y == label, 6].mean(),
                  X[y == label, 10].mean(), name,
                  horizontalalignment='center',
                  bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
    # Reorder the labels to have colors matching the cluster results
    #y = np.choose(y, [1, 2, 3, 4, 5, 6, 7, 8, 9]).astype(np.float)
    ax.scatter(X[:, 5], X[:, 6], X[:, 10], c=y)

    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('dim1')
    ax.set_ylabel('dim2')
    ax.set_zlabel('dim3')
    plt.show()

cluster = 2

iterations = 80
driver("wine.csv", cluster, iterations)
