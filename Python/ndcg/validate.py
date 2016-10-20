#!/usr/bin/env python

"""
compute, save and plot validation scores for a recommender
input is a CSV file with user,movie,rating
output is a Pickle file containing the dict { 'ndcgs': ndcgs }
keys in ndcgs are 1...L
values are list of NDCGs for each user
to get it working, you'll need to supply your recommender
see rec.get_ranked_ids() below
"""


import sys
import csv
import time
import cPickle as pickle
from collections import OrderedDict, defaultdict
from matplotlib import pyplot as plt
from random import shuffle

from recommender import *
from metrics import get_ndcg_2
from relevances import *

#

try:
	input_file = sys.argv[1]
except IndexError:
	input_file = '/path/to/your/input/file'
		
try:
	output_file = sys.argv[2]
	print "saving calculated NDCGs to ", output_file
except IndexError:
	output_file = 'ndcgs/output.pkl'
	
#

# relevance score = rating - relevance_bias
# e.g. 1...5 -> -2...2
relevance_bias = 3.0

ndcg_k = 50
min_items_for_ndcg = 1	# min. known ratings in the test set
min_users = 10			# to compute mean ndcg at the end
max_train_size = 200	# L

###

def validate( ratings ):
	#print "all:", ratings.keys()	
	
	global ndcgs
	
	user_ndcgs = []
	
	# this is L for a user
	max_split_i = min( max_train_size, len( ratings ) - min_items_for_ndcg + 1 )
	print " ", max_split_i
	
	# ratings in random order
	tmp = ratings.items()
	shuffle( tmp )
	ratings = OrderedDict( tmp )
	
	for split_i in range( 1, max_split_i ):	
		train = ratings.keys()[:split_i]
		test = ratings.keys()[split_i:]
		
		# a recommender gets some ratings and returns a list of recommended get_ranked_ids
		# the length of the list is ndcg_k + len( train )
		# in case the recommender returns items from the train set
		ranked_ids = rec.get_ranked_ids( train, ndcg_k + len( train ))
		
		# recommender might return ids in train - filter them out
		ranked_ids = [ x for x in ranked_ids if x not in train ]
		ranked_ids = ranked_ids[:ndcg_k]
		
		relevances = get_relevances( ratings, ranked_ids )	
		best_relevances = get_best_relevances( ratings, test, ndcg_k )

		# when given None as worst_relevances, 
		# get_ndcg_2() will assume that the worst possible DCG is zero
		worst_relevances = None
		
		ndcg = get_ndcg_2( relevances, best_relevances, worst_relevances, ndcg_k )
		
		# nans when all ratings in test are the same, e.g. (2, 2, 2)
		if not np.isnan( ndcg ):
			ndcgs[split_i].append( ndcg )
	
#

ndcgs = defaultdict( list )

reader = csv.reader( open( input_file, 'rb' ))
rec = Rec()

print "start:", time.strftime("%H:%M:%S", time.localtime())
start_time = time.time()

# skip headers
reader.next()

# first line

line = reader.next()
user = line[0]
movie = int( line[1] )
rating = float( line[2] )

current_user = user
current_ratings = OrderedDict()
current_ratings[movie] = rating

counter = 0
user_counter = 1

for line in reader:
	user = line[0]
	movie = int( line[1] )
	rating = float( line[2] )	
	
	if user == current_user:
		current_ratings[movie] = rating
	else:
		
		if len( current_ratings ) >= min_items_for_ndcg + 1:
			validate( current_ratings )
		
		# init new user
		current_ratings = OrderedDict()
		current_ratings[movie] = rating
		current_user = user
		user_counter += 1
		
	counter += 1
	if counter % 1000 == 0:
		print counter, user_counter
		
# the last user
if len( current_ratings ) >= min_items_for_ndcg + 1:
	validate( current_ratings )
	
#

print "end:", time.strftime("%H:%M:%S", time.localtime())
end_time = time.time()

duration = end_time - start_time
duration_minutes = int( duration ) / 60
duration_seconds = int( duration ) % 60
print "validation took {} minutes, {} seconds.".format( duration_minutes, duration_seconds )
	
if output_file:
	pickle.dump( { 'ndcgs': ndcgs }, open( output_file, 'wb' ))
	
# plot

mean_ndcgs = OrderedDict( { k: sum( v ) / len( v ) for k, v in ndcgs.items() if len( v ) >= min_users } )
for k, v in mean_ndcgs.items()[:20]:
	print k, v
	
plt.plot( mean_ndcgs.keys(), mean_ndcgs.values())

axes = plt.gca()
axes.set_ylim([0,0.5])

plt.show()