# Thomas Page
# McGill University
# October 2017

import csv
import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


def main():
    training_x = '../Data/train_set_x.csv'
    training_y = '../Data/train_set_y.csv'
    testing_x = '../Data/test_set_x.csv'

    dataset_x, dataset_y, testset_x = loadCsv(training_x, training_y, training_x)
    # print type(dataset_y[1])

    probability_of_classes = probability_class(dataset_y)
    # print(probability_of_classes)

    slovak, french, spanish, german, polish = probability_languages(dataset_x, dataset_y)
    probability_of_languages = [slovak, french, spanish, german, polish]

    predictions = classifier(testset_x, probability_of_languages, probability_of_classes)

    # Percent correct on trainging set (ignore for testing):

    truth = [el[1] for el in dataset_y[1:]]
    print(truth[:10])
    print(predictions[:10])
    confusion = confusion_matrix(truth, predictions)

    plt.figure()
    plot_confusion_matrix(confusion, classes=['Slovak', 'French', 'Spanish', 'German', 'Polish'], normalize=True, title='Confusion Matrix')
    print(confusion)
    correct = [i for i, j in zip(predictions, truth) if i == j]
    percent_correct = float(len(correct)) / float(len(predictions))
    print(percent_correct)
    plt.show()


def loadCsv(training_x, training_y, testing_x):
    with open(training_x) as data_x:
        reader_x = csv.reader(data_x)
        dataset_x = list(reader_x)
        for i in range(1, len(dataset_x)):
            dataset_x[i][1] = dataset_x[i][1].lower()

    with open(training_y) as data_y:
        reader_y = csv.reader(data_y)
        dataset_y = list(reader_y)

    with open(testing_x) as test_x:
        reader_test_x = csv.reader(test_x)
        testset_x = list(reader_test_x)

    return dataset_x, dataset_y, testset_x


def count_words(words):
    # Changing this line affects output:
    wc = {'not_present': 0.0}
    for word in words:
        wc[word] = wc.get(word, 0.0) + 1.0

    offset = 0.0
    if ' ' in wc:
        offset = wc[' ']
        del wc[' ']

    if len(words) > 0:
        for word in wc:
            wc[word] = wc[word] / (len(words) - offset + 1.0)

    return wc


def probability_class(dataset_y):
    result = {}
    for i in range(1, len(dataset_y)):
        result[dataset_y[(i)][1]] = result.get(dataset_y[(i)][1], 0.0) + 1.0
    for k in result:
        result[k] = result[k] / (len(dataset_y))
    return result


def probability_languages(dataset_x, dataset_y):
    slovak = []     # 0
    french = []     # 1
    spanish = []    # 2
    german = []     # 3
    polish = []     # 4

    for i in range(1, len(dataset_x)):
        if dataset_y[i][1] == '0':
            slovak += (dataset_x[i][1])
        elif dataset_y[i][1] == '1':
            french += (dataset_x[i][1])
        elif dataset_y[i][1] == '2':
            spanish += (dataset_x[i][1])
        elif dataset_y[i][1] == '3':
            german += (dataset_x[i][1])
        elif dataset_y[i][1] == '4':
            polish += (dataset_x[i][1])

    slovak_ = (count_words(slovak))
    french_ = (count_words(french))
    spanish_ = (count_words(spanish))
    german_ = (count_words(german))
    polish_ = (count_words(polish))

    return slovak_, french_, spanish_, german_, polish_


def print_probabilities(slovak_, french_, spanish_, german_, polish_):
    print("\nSlovak:")
    print(slovak_)

    print("\nFrench:")
    print(french_)

    print("\nSpanish:")
    print(spanish_)

    print("\nGerman:")
    print(german_)

    print("\nPolish:")
    print(polish_)

    return


def classifier(testset_x, probability_of_languages, probability_of_classes):
    okay = [[] for i in range(len(testset_x))]

    for i in range(1, len(testset_x)):
        okay[i] += testset_x[i][1].lower()
        okay[i] = [x for x in okay[i] if x != ' ']

    guesses = []

    for m in range(1, len(testset_x)):
        temp_probability = probability_of_classes.copy()
        for l in range(0, len(probability_of_languages)):
            for i in okay[m]:
                if i in probability_of_languages[l]:
                    temp_probability[str(l)] *= probability_of_languages[l][i]
                else:
                    temp_probability[str(l)] *= probability_of_languages[l]['not_present']

        guesses.append(max(temp_probability, key=temp_probability.get))

    with open('predictions.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        writer.writerow(['Id'] + ['Category'])
        for i in range(0, len(guesses)):
            writer.writerow([str(i)] + [guesses[i]])

    return guesses


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == '__main__':
    main()
