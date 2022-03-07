import _pickle as cPickle
import numpy as np


matrix = np.loadtxt("Dagger-All-Data/Dagger_D0_D1_s.txt", dtype=np.float32)

f = open('Dataset/D1_data.pkl', 'wb')


pickler = cPickle.Pickler(f)

pickler.dump(matrix)