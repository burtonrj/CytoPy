from multiprocessing import Pool

def add_(x):
	print(x[0]+x[1])
pool = Pool(4)
pool.map(add_, [[1,2], [2,3], [2,3], [9,7]])

