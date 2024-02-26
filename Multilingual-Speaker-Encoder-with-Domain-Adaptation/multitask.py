'''
	Copyright (c) 2019-2020 Xin Detai@Beihang University

	Description:
		python multi-task processing
	Licence:
		MIT
	THE USER OF THIS CODE AGREES TO ASSUME ALL LIABILITY FOR THE USE OF THIS CODE.
	Any use of this code should display all the info above.
'''

import concurrent
from math import ceil
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def divide_to_chunk(data, chunksize = 10000):
	length = len(data)
	return (data[i:i+chunksize] for i in range(0, length, chunksize))

def pool_map(task, data, mode = 'Thread', max_workers = 10, chunksize = None):
	length = len(data)
	chunksize = chunksize if chunksize is not None else int(length / (2 * max_workers))
	chunkdata = divide_to_chunk(data, chunksize)
	Pool = ThreadPoolExecutor if mode == 'Thread' else ProcessPoolExecutor
	output = []
	final_chunk = 0
	print('Length: %d, max_workers: %d, chunksize: %d, #chunk: %d'
			% (length, max_workers, chunksize, ceil(length / chunksize)))
	with Pool(max_workers = max_workers) as executor:
		results = executor.map(task, chunkdata)
		try:
			for result in results:
				output.extend(result)
				final_chunk += 1
		except Exception as e:
			print (e)
	print('Final #chunk: %d' % (final_chunk))
	return output

if __name__ == '__main__':
	def square(x):
		return [item*item for item in x]
	from time import time
	task = square
	data = [i for i in range(100000)]
	st = time()
	out = pool_map(task, data, mode = 'Thread', max_workers = 10, chunksize = 5000)
	et = time()
	print(out[:100])
	print(len(out))
	print('total time: %.4f' % (et - st))

