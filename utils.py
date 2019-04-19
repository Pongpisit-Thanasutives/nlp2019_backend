import pickle

def savefile(obj, filename):
    pickle_out = open(filename, "wb")
    pickle.dump(obj, pickle_out)
    pickle_out.close()

def loadfile(filename):
    pickle_in = open(filename,"rb")
    return pickle.load(pickle_in)