from sklearn.utils import shuffle
import os, json
import pickle


def load_pkl(file):
    with open(file, "rb") as f:
        data = pickle.load(f)
    return data

def write_pkl(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)
    return

def read_res(file):
    with open(file, "r") as f:
        data = f.readlines()
    data = [int(v) for v in data]
    return data



def read_TREC():
    data = {}

    def read(mode):
        x, y = [], []

        with open("data/TREC/TREC_" + mode + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                if line[-1] == "\n":
                    line = line[:-1]
                y.append(line.split()[0].split(":")[0])
                x.append(line.split()[1:])

        x, y = shuffle(x, y)

        if mode == "train":
            dev_idx = len(x) // 10
            data["dev_x"], data["dev_y"] = x[:dev_idx], y[:dev_idx]
            data["train_x"], data["train_y"] = x[dev_idx:], y[dev_idx:]
        else:
            data["test_x"], data["test_y"] = x, y

    read("train")
    read("test")

    return data


def read_defect(path, train, valid, test):
    data = {}
    trainx, trainy = load_pkl(os.path.join(path, train))
    validx, validy = load_pkl(os.path.join(path, valid))
    testx, testy = load_pkl(os.path.join(path, test))
    trainx, trainy = shuffle(trainx, trainy)
    validx, validy = shuffle(validx, validy)
    # no need to shuffle test set
    # trainx = [i.split() for i in trainx]
    # validx = [i.split() for i in validx]
    # testx = [i.split() for i in testx]
    data["train_x"], data["train_y"] = trainx, trainy
    data["dev_x"], data["dev_y"] = validx, validy
    data["test_x"], data["test_y"] = testx, testy

    return data
    

def read_MR():
    data = {}
    x, y = [], []

    with open("data/MR/rt-polarity.pos", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(1)

    with open("data/MR/rt-polarity.neg", "r", encoding="utf-8") as f:
        for line in f:
            if line[-1] == "\n":
                line = line[:-1]
            x.append(line.split())
            y.append(0)

    x, y = shuffle(x, y)
    dev_idx = len(x) // 10 * 8
    test_idx = len(x) // 10 * 9

    data["train_x"], data["train_y"] = x[:dev_idx], y[:dev_idx]
    data["dev_x"], data["dev_y"] = x[dev_idx:test_idx], y[dev_idx:test_idx]
    data["test_x"], data["test_y"] = x[test_idx:], y[test_idx:]

    return data


def save_model(model, params):
    if "SAVE_PATH" in params:
        path = params["SAVE_PATH"]
    else:
        path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"
    pickle.dump(model, open(path, "wb"))
    print(f"A model is saved successfully as {path}!")


def load_model(params):
    if "path" in params:
        path = params["path"]
    else:
        path = f"saved_models/{params['DATASET']}_{params['MODEL']}_{params['EPOCH']}.pkl"

    try:
        model = pickle.load(open(path, "rb"))
        print(f"Model in {path} loaded successfully!")

        return model
    except:
        print(f"No available model such as {path}.")
        exit()
