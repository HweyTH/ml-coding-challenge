import pandas as pd
import random
import csv

RANDOM_STATE = 1
N_TRAIN = 1000
N_VALID = 200
EQUAL_CLASS_SPLIT = True

clean_dataset = open("trying_stuff/clean_dataset.csv")
trainfile = open("trying_stuff/trainfile.csv", "w+")
validfile = open("trying_stuff/validfile.csv", "w+")
testfile = open("trying_stuff/testfile.csv", "w+")

data = list(csv.reader(clean_dataset))
train = csv.writer(trainfile)
valid = csv.writer(validfile)
test = csv.writer(testfile)

header = data.pop(0)
train.writerow(header)
valid.writerow(header)
test.writerow(header)

random.seed(RANDOM_STATE)

if EQUAL_CLASS_SPLIT:

    random.shuffle(data[:len(data) // 4])
    random.shuffle(data[len(data) // 4:(len(data) // 4) * 2])
    random.shuffle(data[(len(data) // 4) * 2:(len(data) // 4) * 3])
    random.shuffle(data[(len(data) // 4) * 3:])

    for i in range(len(data) // 4):
        if i < N_TRAIN // 4: 
            train.writerow(data[i])
            train.writerow(data[(len(data) // 4) + i])
            train.writerow(data[2 * (len(data) // 4) + i])
            train.writerow(data[3 * (len(data) // 4) + i])
        elif i < (N_TRAIN + N_VALID) // 4: 
            valid.writerow(data[i])
            valid.writerow(data[(len(data) // 4) + i])
            valid.writerow(data[2 * (len(data) // 4) + i])
            valid.writerow(data[3 * (len(data) // 4) + i])
        else: 
            test.writerow(data[i])
            test.writerow(data[(len(data) // 4) + i])
            test.writerow(data[2 * (len(data) // 4) + i])
            test.writerow(data[3 * (len(data) // 4) + i])

else:

    random.shuffle(data)

    for i in range(len(data)):
        if i < N_TRAIN: 
            train.writerow(data[i])
        elif i < N_TRAIN + N_VALID: 
            valid.writerow(data[i])
        else: 
            test.writerow(data[i])


clean_dataset.close()
trainfile.close()
testfile.close()
