import numpy as np

# change this if using K > 100
denominator_table = np.log2( np.arange( 2, 102 ))

def dcg_at_k( r, k, method = 1 ):
	"""Score is discounted cumulative gain (dcg)
	Relevance is positive real values.  Can use binary
	as the previous methods.
	Example from
	http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
	>>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
	>>> dcg_at_k(r, 1)
	3.0
	>>> dcg_at_k(r, 1, method=1)
	3.0
	>>> dcg_at_k(r, 2)
	5.0
	>>> dcg_at_k(r, 2, method=1)
	4.2618595071429155
	>>> dcg_at_k(r, 10)
	9.6051177391888114
	>>> dcg_at_k(r, 11)
	9.6051177391888114
	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider
		method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
				If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
	Returns:
		Discounted cumulative gain
	"""
	r = np.asfarray(r)[:k]
	if r.size:
		if method == 0:
			return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
		elif method == 1:
			# return np.sum(r / np.log2(np.arange(2, r.size + 2)))
			return np.sum(r / denominator_table[:r.shape[0]])
		else:
			raise ValueError('method must be 0 or 1.')
	return 0.
 
 
def get_ndcg( r, k, method = 1 ):
	"""Score is normalized discounted cumulative gain (ndcg)
	Relevance orignally was positive real values.  Can use binary
	as the previous methods.
	Example from
	http://www.stanford.edu/class/cs276/handouts/EvaluationNew-handout-6-per.pdf
	>>> r = [3, 2, 3, 0, 0, 1, 2, 2, 3, 0]
	>>> ndcg_at_k(r, 1)
	1.0
	>>> r = [2, 1, 2, 0]
	>>> ndcg_at_k(r, 4)
	0.9203032077642922
	>>> ndcg_at_k(r, 4, method=1)
	0.96519546960144276
	>>> ndcg_at_k([0], 1)
	0.0
	>>> ndcg_at_k([1], 2)
	1.0
	Args:
		r: Relevance scores (list or numpy) in rank order
			(first element is the first item)
		k: Number of results to consider
		method: If 0 then weights are [1.0, 1.0, 0.6309, 0.5, 0.4307, ...]
				If 1 then weights are [1.0, 0.6309, 0.5, 0.4307, ...]
	Returns:
		Normalized discounted cumulative gain
	"""
	dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
	dcg_min = dcg_at_k(sorted(r), k, method)
	#assert( dcg_max >= dcg_min )
	
	if not dcg_max:
		return 0.
	 
	dcg = dcg_at_k(r, k, method)
	
	#print dcg_min, dcg, dcg_max
	
	return (dcg - dcg_min) / (dcg_max - dcg_min)

# ndcg with explicitly given best and worst possible relevances
# for recommendations including unrated movies
def get_ndcg_2( r, best_r, worst_r, k, method = 1 ):

	dcg_max = dcg_at_k( sorted( best_r, reverse = True ), k, method )
	
	if worst_r == None:
		dcg_min = 0.
	else:
		dcg_min = dcg_at_k( sorted( worst_r ), k, method )
		
	# assert( dcg_max >= dcg_min )
	
	if not dcg_max:
		return 0.
	 
	dcg = dcg_at_k( r, k, method )
	
	#print dcg_min, dcg, dcg_max
	
	return ( dcg - dcg_min ) / ( dcg_max - dcg_min )
 
 
 
 
 