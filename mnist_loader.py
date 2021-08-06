import pickle
import gzip
import numpy as np

def load():
    # traning_data, validation_data, test_data = None, None, None
    with gzip.open("data/mnist.pkl.gz", "rb") as f:
        training_data, validation_data, test_data = pickle.load(f, encoding="latin1")
    return (training_data, validation_data, test_data)

def result_vector(j):
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e

def load_wrapped():
    tr_d, va_d, te_d = load()
    
    def reshape(data, size=(784, 1)):
        return [np.reshape(x, size) for x in data]
        
    training_inputs = reshape(tr_d[0])
    training_results = [result_vector(y) for y in tr_d[1]]
    training_data = list(zip(training_inputs, training_results))

    validation_inputs = reshape(va_d[0])
    validation_data = list(zip(validation_inputs, va_d[1]))

    test_inputs = reshape(te_d[0])
    test_data = list(zip(test_inputs, te_d[1]))
    
    return (training_data, validation_data, test_data)
