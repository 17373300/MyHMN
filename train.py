import argparse
from model.model import HMN
import torch
import datetime
from util.dataset import make_data
import os
import json
from transformers import BertTokenizer
import numpy as np


def save(model, save_dir, epoch):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, str(epoch) + ".pt")
    torch.save(model.state_dict(), save_path)


def eval(model, iterator, law_token, criterion, result_path, tokenizer):
    model.eval()
    loss_all = 0
    right = 0
    wrong = 0
    for batch in iterator:
        text, text_lens, label = batch
        text_lens, label = text_lens.cuda(), label.cuda()
        predict, M = model(text, text_lens, law_token)
        loss = criterion(predict.float(), label.float())
        loss_all += loss.item()
        # print(predict)

        gap = 0.5
        predict = predict.cpu().detach()
        label = label.cpu().detach()
        tp = fp = fn = tn = 0
        for i in range(predict.size(0)):
            for j in range(predict.size(1)):
                if predict[i][j] < gap:
                    if label[i][j] < gap:
                        tn += 1
                    else:
                        fn += 1
                        print(predict[i])
                        print(label[i])
                        print("\n")
                else:
                    if label[i][j] < gap:
                        fp += 1
                        print(predict[i])
                        print(label[i])
                        print("\n")
                    else:
                        tp += 1
        precision = tp / (tp + fp + 0.1)
        recall = tp / (tp + fn + 0.1)
        f1 = 2 * precision * recall / (precision + recall)

        predict = np.argmax(predict, axis=1)
        label = np.argmax(label, axis=1)
        for i, j in enumerate(predict):
            if predict[i].item() == label[i].item():
                right += 1
            else:
                wrong += 1
    print(predict)
    print(label)
    print("Dev : precision {}, recall {}, f1 {}".format(precision, recall, f1))
    print("Dev : loss {}".format(loss_all))
    print("Dev : Accuracy {}".format(right / (right + wrong)))
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--bert_path', type=str, default='./xs/')
    parser.add_argument('--train_path',
                        type=str,
                        default='./data/train_property.json')
    parser.add_argument('--dev_path',
                        type=str,
                        default='./data/dev_property.json')
    parser.add_argument('--law_path',
                        type=str,
                        default='./data/law_details.json')
    parser.add_argument('--save_path', type=str, default='./save/')
    parser.add_argument('--gpu', default=1, type=int)
    parser.add_argument('--epoch', default=30, type=int)
    parser.add_argument('--learning_rate', default=0.0005, type=float)
    parser.add_argument('--model_path', default="", type=str)

    args = parser.parse_args()

    model = HMN()
    torch.cuda.set_device(args.gpu)
    model = model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    train_iter = make_data(args.train_path, args)
    dev_iter = make_data(args.dev_path, args)

    with open(args.law_path, "r") as f:
        law_token = json.load(f)
    tokenizer = BertTokenizer.from_pretrained(args.bert_path,
                                              do_lower_case=False)
    criterion = torch.nn.MSELoss()

    if len(args.model_path) != 0:
        model.load_state_dict(torch.load(args.model_path))
        eval(model, dev_iter, law_token, criterion,
             args.save_path + "result.txt", tokenizer)
        return

    for epoch in range(1, args.epoch + 1):
        start_test_time = datetime.datetime.now()
        loss_all = 0
        right = 0
        wrong = 0
        print("============= epoch:{} =============".format(epoch))
        for batch in train_iter:
            text, text_lens, label = batch
            text_lens, label = text_lens.cuda(), label.cuda()
            predict, M = model(text, text_lens, law_token)
            loss = criterion(predict.float(), label.float())
            loss_all += loss.item()
            loss.backward()
            optimizer.step()

            predict = np.argmax(predict.cpu().detach(), axis=1)
            label = np.argmax(label.cpu().detach(), axis=1)
            for i, j in enumerate(predict):
                if predict[i].item() == label[i].item():
                    right += 1
                else:
                    wrong += 1

        end_test_time = datetime.datetime.now()
        print("Train : epoch {}, time cost {}".format(
            epoch + 1, end_test_time - start_test_time))
        print("Train : loss {}".format(loss_all))
        print("Train : Accuracy {}".format(right / (right + wrong)))
        save(model, args.save_path, epoch)
        eval(model, dev_iter, law_token, criterion,
             args.save_path + str(epoch) + ".txt", tokenizer)


if __name__ == '__main__':
    main()
