from gensim.utils import pickle
from model import CNN
import utils
from tqdm import tqdm
import os, pickle

from torch.autograd import Variable
import torch
import torch.optim as optim
import torch.nn as nn

from sklearn.utils import shuffle
from sklearn.metrics import f1_score, classification_report
from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import argparse
import copy


def train(data, params):
    if params["MODEL"] != "rand":
        # load word2vec
        print("loading word2vec...")
        word_vectors = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

        wv_matrix = []
        for i in range(len(data["vocab"])):
            word = data["idx_to_word"][i]
            if word in word_vectors.vocab:
                wv_matrix.append(word_vectors.word_vec(word))
            else:
                wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))

        # one for UNK and one for zero padding
        wv_matrix.append(np.random.uniform(-0.01, 0.01, 300).astype("float32"))
        wv_matrix.append(np.zeros(300).astype("float32"))
        wv_matrix = np.array(wv_matrix)
        params["WV_MATRIX"] = wv_matrix

    model = CNN(**params).cuda(params["GPU"])

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adadelta(parameters, params["LEARNING_RATE"])
    criterion = nn.CrossEntropyLoss()

    pre_dev_f1 = 0
    max_dev_f1 = 0
    max_test_f1 = 0
    for e in range(params["EPOCH"]):
        data["train_x"], data["train_y"] = shuffle(data["train_x"], data["train_y"])

        for i in tqdm(range(0, len(data["train_x"]), params["BATCH_SIZE"])):
            batch_range = min(params["BATCH_SIZE"], len(data["train_x"]) - i)

            batch_x = data["train_x"][i:i + batch_range]
            batch_y = data["train_y"][i:i + batch_range]

            batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
            batch_y = Variable(torch.LongTensor(batch_y)).cuda(params["GPU"])

            optimizer.zero_grad()
            model.train()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            nn.utils.clip_grad_norm(parameters, max_norm=params["NORM_LIMIT"])
            optimizer.step()

        dev_f1 = test(data, model, params, mode="dev")
        test_f1 = test(data, model, params)
        print("epoch:", e + 1, "/ dev_f1:", dev_f1, "/ test_f1:", test_f1)
        params["SAVE_PATH"] = params["SAVE_NAME"] + str(e + 1) + ".pkl"
        utils.save_model(model, params)
        if params["EARLY_STOPPING"] and dev_f1 <= pre_dev_f1:
            print("early stopping by dev_f1!")
            break
        else:
            pre_dev_f1 = dev_f1

        if dev_f1 > max_dev_f1:
            max_dev_f1 = dev_f1
            max_test_f1 = test_f1
            best_model = copy.deepcopy(model)

    print("max dev f1:", max_dev_f1, "test f1:", max_test_f1)
    return best_model


def test(data, model, params, mode="test", output=False):
    model.eval()

    if mode == "dev":
        x, y = data["dev_x"], data["dev_y"]
    elif mode == "test":
        x, y = data["test_x"], data["test_y"]
    preds = []
    for i in range(0, len(x), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(x) - i)

        batch_x = x[i:i + batch_range]
        batch_y = y[i:i + batch_range]
        batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
        pred = np.argmax(model(batch_x).cpu().data.numpy(), axis=1)
        preds.extend(pred)
    # acc = sum([1 if p == y else 0 for p, y in zip(preds, y)]) / len(preds)
    f1 = f1_score(y, preds)
    if output:
        with open("gold.txt", "w") as f:
            f.writelines("\n".join([str(p) for p in y]))
        with open("pred.txt", "w") as f:
            f.writelines("\n".join([str(p) for p in preds]))
    print(classification_report(y, preds, digits=4))
    return f1


def process_data(data, params, save_dir):
    # if os.path.exists(save_dir):
    #     with open(save_dir, "rb") as f:
    #         data = pickle.load(f)
    #     return data

    print("Processing training data.")
    trainx = [[data["word_to_idx"][w] if w in data["word_to_idx"] else data["word_to_idx"]["<UNK>"] for w in sent] +
                [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                for sent in data["train_x"]]
    trainy = [data["classes"].index(c) for c in data["train_y"]]
    data["train_x"], data["train_y"] = trainx, trainy

    print("Processing dev data.")
    devx = [[data["word_to_idx"][w] if w in data["word_to_idx"] else data["word_to_idx"]["<UNK>"] for w in sent] +
                [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                for sent in data["dev_x"]]
    devy = [data["classes"].index(c) for c in data["dev_y"]]
    data["dev_x"], data["dev_y"] = devx, devy

    print("Processing test data.")
    testx = [[data["word_to_idx"][w] if w in data["word_to_idx"] else data["word_to_idx"]["<UNK>"] for w in sent] +
                [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                for sent in data["test_x"]]
    testy = [data["classes"].index(c) for c in data["test_y"]]
    data["test_x"], data["test_y"] = testx, testy

    # with open(save_dir, "wb") as f:
    #     pickle.dump(data, f)
    return data


def main():
    parser = argparse.ArgumentParser(description="-----[CNN-classifier]-----")
    parser.add_argument("--mode", default="train", help="train: train (with test) a model / test: test saved models")
    parser.add_argument("--model", default="rand", help="available models: rand, static, non-static, multichannel")
    parser.add_argument("--dataset", default="defect", help="available datasets: MR, TREC")
    parser.add_argument("--data_path", default=None, type=str)
    parser.add_argument("--train_name", default=None, type=str)
    parser.add_argument("--valid_name", default=None, type=str)
    parser.add_argument("--test_name", default=None, type=str)
    parser.add_argument("--load_model", default="None", type=str)
    parser.add_argument("--save_name", default="None", type=str)
    parser.add_argument("--output", action='store_true')
    parser.add_argument("--save_model", default=False, action='store_true', help="whether saving model or not")
    parser.add_argument("--early_stopping", default=False, action='store_true', help="whether to apply early stopping")
    parser.add_argument("--epoch", default=100, type=int, help="number of max epoch")
    parser.add_argument("--learning_rate", default=1.0, type=float, help="learning rate")
    parser.add_argument("--gpu", default=-1, type=int, help="the number of gpu to be used")

    options = parser.parse_args()
    # data = getattr(utils, f"read_{options.dataset}")()
    data = getattr(utils, f"read_{options.dataset}")(options.data_path, options.train_name, options.valid_name, "test.pkl")
    data2 = getattr(utils, f"read_{options.dataset}")(options.data_path, options.train_name, options.valid_name, options.test_name)

    data["vocab"] = sorted(list(set([w for sent in data["train_x"] + data["dev_x"] for w in sent]))) + ["<UNK>"]
    data["classes"] = sorted(list(set(data["train_y"])))
    data["word_to_idx"] = {w: i for i, w in enumerate(data["vocab"])}
    data["idx_to_word"] = {i: w for i, w in enumerate(data["vocab"])}

    data["test_x"] = data2["test_x"]
    data["test_y"] = data2["test_y"]

    params = {
        "MODEL": options.model,
        "DATASET": options.dataset,
        "SAVE_MODEL": options.save_model,
        "SAVE_NAME": options.save_name,
        "EARLY_STOPPING": options.early_stopping,
        "EPOCH": options.epoch,
        "LEARNING_RATE": options.learning_rate,
        "MAX_SENT_LEN": max([len(sent) for sent in data["train_x"] + data["dev_x"] + data["test_x"]]),
        "BATCH_SIZE": 50,
        "WORD_DIM": 300,
        "VOCAB_SIZE": len(data["vocab"]),
        "CLASS_SIZE": len(data["classes"]),
        "FILTERS": [3, 4, 5],
        "FILTER_NUM": [100, 100, 100],
        "DROPOUT_PROB": 0.5,
        "NORM_LIMIT": 3,
        "GPU": options.gpu
    }

    data = process_data(data, params, save_dir=os.path.join(options.data_path, "savedata.pkl"))

    print("=" * 20 + "INFORMATION" + "=" * 20)
    print("MODEL:", params["MODEL"])
    print("DATASET:", params["DATASET"])
    print("VOCAB_SIZE:", params["VOCAB_SIZE"])
    print("EPOCH:", params["EPOCH"])
    print("LEARNING_RATE:", params["LEARNING_RATE"])
    print("EARLY_STOPPING:", params["EARLY_STOPPING"])
    print("SAVE_MODEL:", params["SAVE_MODEL"])
    print("=" * 20 + "INFORMATION" + "=" * 20)

    if options.mode == "train":
        print("=" * 20 + "TRAINING STARTED" + "=" * 20)
        model = train(data, params)
        # if params["SAVE_MODEL"]:
        #     utils.save_model(model, params)
        print("=" * 20 + "TRAINING FINISHED" + "=" * 20)
    else:
        params["path"] = options.load_model
        model = utils.load_model(params).cuda(params["GPU"])

        test_f1 = test(data, model, params, output=options.output)
        print("test f1:", test_f1)


if __name__ == "__main__":
    main()
