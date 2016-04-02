import csv
import math
import numpy as np
import random
import numpy.random as nprand


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
	a=999
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
		lis = bin(a)
		lis = lis[2:]
		velocity.append(a)

#calculate the global fitness function
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

def move(particle, vel, hyper_to_d, particles_h_to_d, position, path):
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
	path = {}
	while i < iterations:
		if i%10==0 and i > 0:
			#print "gbest"
			#print gbest
			#print "pbest"
			#print pbest
			pass
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
			position[j] = move(position[j] , velocity[j], hyper_to_d, particles_h_to_d, position, path)
			#while moved not in position:
			#	moved = move(position[j] , velocity[j], hyper_to_d, particles_h_to_d, position, path)
#			print position[j]
			#position[j] = moved
			particles_h_to_d[position[j]] = hyper_to_d[position[j]]
			particles_d_to_h[hyper_to_d[position[j]]] = position[j]


			pfit = local_fitness(particles_d_to_h, position[j], particles_h_to_d)
			if pfit > best_local_value[j]:

				gfit = global_fitness(particles_d_to_h)
				if(gfit > best_value):
					best_value = gfit
					gbest[j] = position[j]
				#print "g updated" + str(i)

				best_local_value[j] = pfit
				pbest[j] = position[j]
				path[position[j]] = True
				#print "p updated" + str(i)
				print "pbest updated for " + str(j) + " particle" + " improved val: " + str(pfit) + "in iteration: " + str(i) + " position of rest is: "
				print "local value for this configuration: " + str(sum(best_local_value)/2)
				print "global value for this configuration: " + str(gfit)
				print particles_h_to_d

		i = i+1
	print path
	return (gbest, pbest, best_value, best_local_value)


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
	position = [] #current position of particle
	path = []
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
	(gbest, pbest, best_value, best_local_value) = pso(iterations, num_clusters, best_value, best_local_value, num_particles, d_to_hyper, hyper_to_d, particles_h_to_d, particles_d_to_h, hypercube_dim, hypercube_dim_float, pbest, gbest, position, velocity)
	print "gbest:"
	print gbest
	print "pbest:"
	print pbest
	print str(best_value) + " global value"
	print str(sum(best_local_value)/2) + " local value"

cluster = 11
driver(cluster)