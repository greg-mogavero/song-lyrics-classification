import pickle

def load_preprocessed_data(fp="./preprocessed_data.pkl"):
	with open(fp, "rb") as f:
		X, y, index, reverse_index, le = pickle.load(f)
		
	return X, y, index, reverse_index, le