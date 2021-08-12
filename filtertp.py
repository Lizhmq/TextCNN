import json, pickle
from utils import load_pkl, write_pkl, read_res
from sklearn.metrics import classification_report

def main():
    gold = read_res("gold.txt")
    pred = read_res("pred.txt")
    tx, ty = load_pkl("data/data/test.pkl")
    print(classification_report(gold, pred, digits=4))
    newtx, newty = [], []
    for i in range(len(gold)):
        if gold[i] == 1 and gold[i] == pred[i]:
            newtx.append(tx[i])
            newty.append(ty[i])
            print(tx[i])
            print(ty[i])
            print("\n\n")
    write_pkl("data/data/testtp.pkl", (newtx, newty))



if __name__ == "__main__":
    main()