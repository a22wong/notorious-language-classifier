# Alex Wong
# 260602944
# COMP 551 Project 2
import csv
import itertools
import numpy as np
import codecs
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def main():
    training_x = '../Data/train_set_x.csv'
    training_y = '../Data/train_set_y.csv'
    testing_x = '../Data/test_set_x.csv'

    # pre-process data
    dataset_x, dataset_y, testset_x = loadCsv(training_x, training_y, testing_x)
    probability_of_classes = probability_class(dataset_y)
    special_chars = getSpecialChars(training_x, dataset_y)
    classified_sc = classifySpecialChars(special_chars)
    slovak, french, spanish, german, polish = probability_languages(dataset_x, dataset_y)
    probability_of_languages = [slovak, french, spanish, german, polish]
    
    # make predicitons!
    predictions = classify(testset_x, classified_sc, probability_of_languages, probability_of_classes)

    truth = [el[1] for el in dataset_y[1:]]
    # print(truth[:10])
    # print(predictions[:10])
    # confusion = confusion_matrix(truth, predictions)

    # plt.figure()
    # plot_confusion_matrix(confusion, classes=['Slovak', 'French', 'Spanish', 'German', 'Polish'], normalize=True, title='Confusion Matrix')
    # print(confusion)
    correct = [i for i, j in zip(predictions, truth) if i == j]
    percent_correct = float(len(correct)) / float(len(predictions))
    print(percent_correct)
    # plt.show()

def loadCsv(training_x, training_y, testing_x):
    print "Loading from csv..."
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

def getSpecialChars(training_x, dataset_y):
    print "Getting special characters..."
	# open training set with utf-8 encoding to indentify special chars
	# special_chars: dictionary {key=special_char, value=list of associated languages}
    unicode_regex = re.compile('[^\x00-\x7F]', re.IGNORECASE)
    special_chars = {}
    with codecs.open(training_x, mode='r', encoding='utf-8') as data_x:
        line_nb = -1
        for line in data_x:
            line_nb += 1
            sys.stdout.write("\rProgress: %d" % (line_nb))
            sys.stdout.flush()
            for i in range(len(line)):
                if re.match(unicode_regex, line[i]):
                    if not line[i] in special_chars:
                        special_chars[line[i]] = [dataset_y[line_nb][1]]
                    else:
                        if not dataset_y[line_nb][1] in special_chars[line[i]]:
                            special_chars[line[i]].append(dataset_y[line_nb][1])
    data_x.close()

    return special_chars

def classifySpecialChars(special_chars):
    print "Classifying special characters..."
    # {'1': 462, '0': 41, '3': 34, '2': 240, '4': 41}
    slovak = []    # 0
    french = []    # 1
    spanish = []   # 2
    german = []    # 3
    polish = []    # 4

    for c in special_chars:
		if '0' in special_chars[c]:
			slovak.append(c)
		if '1' in special_chars[c]:
			french.append(c)
		if '2' in special_chars[c]:
			spanish.append(c)
		if '3' in special_chars[c]:
			german.append(c)
		if '4' in special_chars[c]:
			polish.append(c)

    return [slovak, french, spanish, german, polish]

def count_words(words):
    print "Counting words..."
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
    print "Calculating class probability..."
    result = {}
    for i in range(1, len(dataset_y)):
        result[dataset_y[(i)][1]] = result.get(dataset_y[(i)][1], 0.0) + 1.0
    for k in result:
        result[k] = result[k] / (len(dataset_y))
    return result

def probability_languages(dataset_x, dataset_y):
    print "Calculating language probability"
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

def classify(testset_x, classified_sc, probability_of_languages, probability_of_classes):
    print "Classifying..."
    guesses = []

    testset = [[] for i in range(len(testset_x))]

    for i in range(1, len(testset_x)):
        testset[i] += testset_x[i][1]#.lower()
        testset[i] = [x for x in testset[i] if x != ' ']

    for test_line in range(1, len(testset_x)):
        sys.stdout.write("\rProgress: %d" % (test_line))
        sys.stdout.flush()
        temp_probability = probability_of_classes.copy()
        classified_sc_counts = {'0':0, '1':0,'2':0,'3':0,'4':0}
        naive = 1
    	for test_char in testset[test_line]:
    		# TODO: initialize list for each lang's special chars for branching
    		# or use ~smarter~ lists of special chars for brancing
    		# if test_char appears in list, branch until leaf

    		# at least filter in order of which lang has largest coverage



            # decision tree approach

    		# slovak
    		if test_char in classified_sc['0']:
                classified_sc_counts['1'] += 1
                naive = 0
    			# guesses.append('1')
    		# french
    		elif test_char in classified_sc['1']:
                classified_sc_counts['2'] += 1
                naive = 0
    			# guesses.append('2')
    		# spanish
    		elif test_char in classified_sc['2']:
                classified_sc_counts['0'] += 1
                naive = 0
    			# guesses.append('0')
    		# german
    		elif test_char in classified_sc['3']:
                classified_sc_counts['4'] += 1
                naive = 0
    			# guesses.append('4')
    		# polish
    		elif test_char in classified_sc['4']:
                classified_sc_counts['3'] += 1
                naive = 0
    			# guesses.append('3')
    		default: naive bayes
            else:
                for l in range(len(probability_of_languages)):
                    # pre-naive bayes filtering approach
                    # if test_char in classified_sc[l]:
                    #     classified_sc_counts[l] += 1
                    #     naive = 0
                    if i in probability_of_languages[l]:
                         temp_probability[str(l)] *= probability_of_languages[l][i]
                    else:
                        temp_probability[str(l)] *= probability_of_languages[l]['not_present']

        # update prediction
        if naive:
            guesses.append(max(temp_probability, key=temp_probability.get))
        else:
            guesses.append(max(classified_sc_counts, key=classified_sc_counts.get))
            # needs tie-breaker with temp_probability --> separate function

    # write out
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