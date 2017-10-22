# Alex Wong
# 260602944
# COMP 551 Project 2


def main():
	training_x = '../Data/train_set_x.csv'
    training_y = '../Data/train_set_y.csv'
    testing_x = '../Data/test_set_x.csv'

    dataset_x, dataset_y, testset_x = loadCsv(training_x, training_y, testing_x)

    special_chars = getSpecialChars(training_x, dataset_y)

    # slovak_spec_chars, french_spec_chars, spanish_spec_chars, german_spec_chars, polish_spec_chars = getSpecCharsForLangs(special_chars)

    classified_sc = classifySpecialChars(special_chars)

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

def classify(testset_x, classified_sc, probability_of_classes):
    print "Classifying..."
	guesses = []

	testset = [[] for i in range(len(testset_x))]

    for i in range(1, len(testset_x)):
        testset[i] += testset_x[i][1]#.lower()
        testset[i] = [x for x in testset[i] if x != ' ']

    for test_line in range(1, len(testset_x)):
        temp_probability = probability_of_classes.copy()
        special_count_by_class = {'0':0, '1':0,'2':0,'3':0,'4':0}
        naive = 1
    	for test_char in testset[test_line]:
    		# TODO: initialize list for each lang's special chars for branching
    		# or use ~smarter~ lists of special chars for brancing
    		# if test_char appears in list, branch until leaf

    		# at least filter in order of which lang has largest coverage



            # decision tree approach
    		# slovak
    		# if test_char in special_chars_langs[0]:
      #           special_count_by_class['0'] += 1
      #           naive = 0
    		# 	# guesses.append(0)
    		# # french
    		# elif test_char in special_chars_langs[1]:
      #           special_count_by_class['1'] += 1
      #           naive = 0
    		# 	# guesses.append(1)
    		# # spanish
    		# elif test_char in special_chars_langs[2]:
      #           special_count_by_class['2'] += 1
      #           naive = 0
    		# 	# guesses.append(2)
    		# # german
    		# elif test_char in special_chars_langs[3]:
      #           special_count_by_class['3'] += 1
      #           naive = 0
    		# 	# guesses.append(3)
    		# # polish
    		# elif test_char in special_chars_langs[4]:
      #           special_count_by_class['4'] += 1
      #           naive = 0
    		# 	# guesses.append(4)
    		# default: naive bayes
    		else:
                for l in range(len(probability_languages)):
                    # pre-naive bayes filtering approach
                    if test_char in special_chars_langs[l]:
                        special_count_by_class[l] += 1
                        naive = 0
                    if i in probability_languages[l]:
                         temp_probability[str(l)] *= probability_of_languages[l][i]
                    else:
                        temp_probability[str(l)] *= probability_of_languages[l]['not_present']

        # update prediction
        if naive:
            guesses.append(max(temp_probability, key=temp_probability.get))
        else:
            guesses.append(max(special_count_by_class, key=special_count_by_class.get))
            # needs tie-breaker with temp_probability --> separate function

    # write out
    with open('predictions.csv', 'w') as csvoutput:
        writer = csv.writer(csvoutput)
        writer.writerow(['Id'] + ['Category'])
        for i in range(0, len(guesses)):
            writer.writerow([str(i)] + [guesses[i]])

    return guesses

if __name__ == '__main__':
    main()