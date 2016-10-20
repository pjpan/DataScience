"plot average NDCGs"


import cPickle as pickle
from collections import OrderedDict
from matplotlib import pyplot as plt

from os import listdir
from os.path import isfile, join

# minimum number of users we have scores for to average
min_users = 10

input_dir = 'ndcgs/'

# a list of input files
input_files = [ join( input_dir, f ) for f in listdir( input_dir ) if isfile( join( input_dir, f )) and f.endswith( '.pkl' ) ]

for i_f in input_files:
	print i_f
	
#

ndcgs = [ pickle.load( open( i_f, 'rb' ))['ndcgs'] for i_f in input_files ]

for i in range( len( ndcgs )):
	assert( sorted( ndcgs[i].keys()) == ndcgs[i].keys())

mean_ndcgs = [ 
	OrderedDict( { k: sum( v ) / len( v ) for k, v in x.items() if len( v ) >= min_users } ) 
	for x in ndcgs ]

colors = [ 'g', 'b', 'r', 'k', 'y' ]

for i, n in enumerate( mean_ndcgs ):
	plt.plot( n.keys(), n.values(), colors[i] )

plt.show()




