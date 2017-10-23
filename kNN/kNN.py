import csv

def main():
	# load data sources
	training_x = '../Data/train_set_x.csv'
    training_y = '../Data/train_set_y.csv'
    testing_x = '../Data/test_set_x.csv'

    dataset_x, dataset_y, testset_x = loadCsv(training_x, training_y, testing_x)
    # print type(dataset_y[1])

    # compute class probailities
    probability_of_classes = probability_class(dataset_y)
    # print(probability_of_classes)

    # compute language probabilties
    slovak, french, spanish, german, polish = probability_languages(dataset_x, dataset_y)
    probability_of_languages = [slovak, french, spanish, german, polish]

    # compute predictions
    predictions = classifier(testset_x, probability_of_languages, probability_of_classes)

    # Percent correct on trainging set (ignore for testing):
    truth = [el[1] for el in dataset_y[1:]]
    correct = [i for i, j in zip(predictions, truth) if i == j]
    percent_correct = float(len(correct)) / float(len(predictions))
    print(percent_correct)

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

# count words in language corpus
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
    slovak = []    # 0
    french = []    # 1
    spanish = []   # 2
    german = []    # 3
    polish = []    # 4

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
