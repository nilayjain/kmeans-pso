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

def populate_essentials(d_to_hyper,hyper_to_d):
	f = open('winequality-white.csv', 'rb')
	reader = csv.reader(f)
	n = 0
	for row in reader:
		n = n+1
	f.close()
	hypercube_dim_float = math.log(n,2)
	#print "hypercube_dim_float " + str(hypercube_dim_float)
	hypercube_dim = math.ceil(hypercube_dim_float)
	#print "hypercube_dim " + str(hypercube_dim)
	f = open('winequality-white.csv', 'rb')
	reader = csv.reader(f)
	n = 0

	for row in reader:
		row = row[:-1]
		bin_n = bin(n)
		bin_n = bin_n[2:]
		diff = (int)(hypercube_dim - len(bin_n))
		key = ' '.join(row)
		lis = ''
		for i in range(diff):
			lis+='0'
		lis+=bin_n
		d_to_hyper[key] = lis
		hyper_to_d[lis] = key
		#print row
		n = n+1
	return (hypercube_dim,hypercube_dim_float, n)

def initialize(num_clusters, d_to_hyper, hyper_to_d, particles_h_to_d, particles_d_to_h, hypercube_dim, hypercube_dim_float, pbest, gbest, position, velocity):
	for i in range(num_clusters):
		x = random.choice(d_to_hyper.keys())
		particles_d_to_h[x] = d_to_hyper[x]
		particles_h_to_d[particles_d_to_h[x]] = hyper_to_d[d_to_hyper[x]]
		pbest.append(d_to_hyper[x])
		gbest.append(d_to_hyper[x])
		position.append(d_to_hyper[x])
		if hypercube_dim == hypercube_dim_float: x = hypercube_dim+1
		else: x = hypercube_dim
		a = random.randrange(0,x)
		velocity.append(a)

# calculate the global fitness function
def global_fitness(particles_d_to_h):
	ans = 0
	for i in particles_d_to_h.keys():
		lis1 = i.split()
		for j in particles_d_to_h.keys():
			lis2 = j.split()
			#print lis1
			#print lis2
			a = 0
			for b in lis1:
				ans = ans + (float(lis1[a]) - float(lis2[a]))**2
				a = a+1
	return ans

# calculate the local fitness function
def local_fitness(particles_d_to_h, p, particles_h_to_d):
	#p is binary string
	key = particles_h_to_d[p]
	ans = 0
	for i in particles_d_to_h.keys():
		lis1 = i.split()
		lis2 = key.split()
		a = 0
		for b in lis1:
			ans = ans + (float(lis1[a]) - float(lis2[a]))**2
			a = a+1
	return ans

def move(particle, vel, hyper_to_d, particles_h_to_d, position):
	dim = len(particle)
	if vel < 0:
		vel = -vel
	#print vel
	#print "move"
	diff = nprand.randint(dim, size = vel)
	out = particle
	for i in diff:
		if out[i] == '0':
			out = out[:i] + '1' + out[i+1:]
		else:
			out = out[:i] + '0' + out[i+1:]
	while out not in hyper_to_d and out not in particles_h_to_d:
		dim = len(particle)
		if vel < 0:
			vel = -vel
		diff = nprand.randint(dim, size = vel)
		out = particle
		for i in diff:
			if out[i] == '0':
				out = out[:i] + '1' + out[i+1:]
			else:
				out = out[:i] + '0' + out[i+1:]
	return out

def edit_distance(pos1, pos2):
	dist = 0
	for i in range(len(pos1)):
		if pos1[i] == pos2[i]:
			dist+=1
	return dist

def pso(iterations, num_clusters, best_value, best_local_value, num_particles, d_to_hyper, hyper_to_d, particles_h_to_d, particles_d_to_h, hypercube_dim, hypercube_dim_float, pbest, gbest, position, velocity):
	i =0
	bests = []
	while i < iterations:
		'''if i%10==0 and i > 0:
			print "gbest"
			print gbest
			print "pbest"
			print pbest'''
		for j in range(num_clusters):
			#print particles_h_to_d
			#print position
			#print j
			particles_h_to_d.pop(position[j], None)
#			del particles_h_to_d[position[j]]
			val = hyper_to_d[position[j]]
#			del particles_d_to_h[val]
			particles_d_to_h.pop(val,None)
			velocity[j] = int(velocity[j] + nprand.uniform(-1,1) * edit_distance(pbest[j], position[j])\
									   + nprand.uniform(-1,1) * edit_distance(gbest[j], position[j]))
			p1 = position
			position[j] = move(position[j] , velocity[j], hyper_to_d, particles_h_to_d, position)
			particles_h_to_d[position[j]] = hyper_to_d[position[j]]
			particles_d_to_h[hyper_to_d[position[j]]] = position[j]

			gfit = global_fitness(particles_d_to_h)
			if(gfit > best_value):
				best_value = gfit
				##### add all good positions reached to a list
				test = copy.deepcopy(position)
				bests.append(test)
				gbest[j] = position[j]

			pfit = local_fitness(particles_d_to_h, position[j], particles_h_to_d)
			if pfit > best_local_value[j]:
				best_local_value[j] = pfit
				#bests.append(position)
				pbest[j] = copy.deepcopy(position[j])

		i = i+1
	print gbest
	print "global best = " + str(best_value)
	test = {}
	test[hyper_to_d[gbest[0]]] = gbest[0]
	test[hyper_to_d[gbest[1]]] = gbest[1]
	test[hyper_to_d[gbest[2]]] = gbest[2]
	#print test
	print "val using gbest = " + str(global_fitness(test))
	return (bests, pbest, best_value)


def driver(cluster):
	num_clusters = cluster
	d_to_hyper = {}
	hyper_to_d = {}
	particles_d_to_h = {}
	particles_h_to_d = {}
	hypercube_dim, hypercube_dim_float, num_particles = populate_essentials(d_to_hyper,hyper_to_d)
	k = 0
	#print hyper_to_d

	pbest = [] #best position of particle (k * lgn)
	gbest = [] #global best position of particle (k * lgn)
	best_value = 0 #best value of global fitness function
	best_local_value = []
	velocity = []
	bests = []
	position = [] #current position of particle
	initialize(num_clusters, d_to_hyper, hyper_to_d, particles_h_to_d, particles_d_to_h, hypercube_dim, hypercube_dim_float, pbest, gbest, position, velocity)
	best_value = global_fitness(particles_d_to_h)
	#print "position "+ position[0]
	#print  velocity[0]
	#print move('11111111', velocity[0], num_particles)
	#print position
	j = 0
	for j in range(num_clusters):
		best_local_value.append(local_fitness(particles_d_to_h, position[j] , particles_h_to_d))

	iterations = 300
	start = time.time()
	(bests, pbest, best_value) = pso(iterations, num_clusters, best_value, best_local_value, num_particles, d_to_hyper, hyper_to_d, particles_h_to_d, particles_d_to_h, hypercube_dim, hypercube_dim_float, pbest, gbest, position, velocity)
	end = time.time()
	print "time to initialize = "
	print end-start
	#print "gbest:"
	#print gbest
	#print "pbest:"
	#print pbest
	#print str(best_value) + " global value"
	#print str(sum(best_local_value)/2) + " local value"
	lis = []
	j = 0
	for i in gbest:
		lis.append(hyper_to_d[i])
		lis[j] = lis[j].split(' ')
		j = j+1

	#inarr = np.asarray(lis, np.float32)
	inarr = []
	for k in range(len(bests)):
		lis = []
		best = bests[k]
		j = 0
		for i in best:
			lis.append(hyper_to_d[i])
			lis[j] = lis[j].split(' ')
			j = j+1
		inarr.append(np.asarray(lis, np.float32))
	inarr = np.asarray(inarr, np.float32)
	#print inarr
	np.random.seed(5)
	#centers = [[1, 1], [-1, -1], [1, -1]]

	f = open("winenew.csv")
	f.readline()  # skip the header
	X = np.genfromtxt(f, delimiter = ',')
	estimators = {'k_means_iris_3': KMeans(n_clusters=9),
	              'k_means_bad_init' : KMeans(n_clusters=9, n_init=16, init='random')
	              }
	i = 0

	for i in range(len(bests)):
		#print inarr[i]
		### check if all seeds are equal!
		estimators['k_means_pso'+str(i)] = KMeans(n_clusters=9, n_init = 1, init=inarr[i])

	fignum = 1
	for name, est in estimators.items():
	    start = time.time()
	    est.fit(X)
	    labels = est.labels_
	    print str(-est.score(X)) + " score of kmeans " + name
	    end = time.time()
	    print (end-start)
	    #ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=labels.astype(np.float))

	#     ax.w_xaxis.set_ticklabels([])
	#     ax.w_yaxis.set_ticklabels([])
	#     ax.w_zaxis.set_ticklabels([])
	#     ax.set_xlabel('Petal width')
	#     ax.set_ylabel('Sepal length')
	#     ax.set_zlabel('Petal length')
	#     fignum = fignum + 1

	# # Plot the ground truth
	# fig = plt.figure(name, figsize=(4, 3))
	# plt.clf()
	# ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

	# plt.cla()

	# for name, label in [('Setosa', 0),
	#                     ('Versicolour', 1),
	#                     ('Virginica', 2)]:
	#     ax.text3D(X[y == label, 3].mean(),
	#               X[y == label, 0].mean() + 1.5,
	#               X[y == label, 2].mean(), name,
	#               horizontalalignment='center',
	#               bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
	# # Reorder the labels to have colors matching the cluster results
	# y = np.choose(y, [1, 2, 0]).astype(np.float)
	# ax.scatter(X[:, 3], X[:, 0], X[:, 2], c=y)

	# ax.w_xaxis.set_ticklabels([])
	# ax.w_yaxis.set_ticklabels([])
	# ax.w_zaxis.set_ticklabels([])
	# ax.set_xlabel('Petal width')
	# ax.set_ylabel('Sepal length')
	# ax.set_zlabel('Petal length')
	# plt.show()

cluster = 9
driver(cluster)
