import nbformat
from nbconvert.preprocessors import ExecutePreprocessor

def execute_notebook(notebook_filename):
	with open(notebook_filename) as ff:
		nb_in = nbformat.read(ff, nbformat.NO_CONVERT)
		
	ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
	nb_out = ep.preprocess(nb_in)
	return nb_out

def test_gnn_node_classification():
	nb_out = execute_notebook('node_classification_answer.ipynb')
	#print('len(nb_out):', len(nb_out))
	#print("len(nb_out[0]):", len(nb_out[0]))
	#for i, e in enumerate(nb_out[0]):
	#	print('nb_out[0] key ', i, ": ", e)
	#for i, e in enumerate(nb_out[0]['cells']):
	#	print("nb_out[0]['cells'] key ", i, ": ", e)
	#print("len(nb_out[0]['cells']):", len(nb_out[0]['cells']))
	##print("nb_out[0]['cells'][0]:", nb_out[0]['cells'][0])
	##print("nb_out[0]['cells'][1]:", nb_out[0]['cells'][1])
	#print("nb_out[0]['cells'][0]['outputs']:", nb_out[0]['cells'][0]['outputs'])
	#print("nb_out[0]['cells'][1]['outputs']:", nb_out[0]['cells'][1]['outputs'])
	#print("nb_out[0]['cells']['outputs'][0][-1]:", nb_out[0]['cells']['outputs'][0][-1])
	# load the last cell's output
	print(float(nb_out[0]['cells'][-1]['outputs'][-1]['text']))
	assert float(nb_out[0]['cells'][-1]['outputs'][-1]['text']) > 0.5

if __name__ == '__main__':
	test_gnn_node_classification()
