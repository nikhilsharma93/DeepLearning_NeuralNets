import matplotlib.pyplot as plt

fileName = r"results/train.log"

with open(fileName) as f:
    ar = []
    for line in f:
        line = line.split()
        if not line[0].find('%') == 0:
            ar.append(float(line[0]))

plt.plot(ar, '-+')
plt.xlabel('Iterations')
plt.ylabel('Mean Accuracy')
plt.title('Accuracy of the training data (averaged over the two classes)')
plt.show()

fileName = r"results/test.log"

with open(fileName) as f:
    ar = []
    for line in f:
        line = line.split()
        if not line[0].find('%') == 0:
            ar.append(float(line[0]))

plt.plot(ar, '-+')
plt.xlabel('Iterations')
plt.ylabel('Mean Accuracy')
plt.title('Accuracy of the test data (averaged over the two classes)')
plt.show()
