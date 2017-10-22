import csv
import codecs
import re
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def main():
    training_x = '../Data/train_set_x.csv'
    training_y = '../Data/train_set_y.csv'
    testing_x = '../Data/test_set_x.csv'

    dataset_x, dataset_y, testset_x= loadCsv(training_x, training_y, testing_x)
    special_chars = getSpecialChars(training_x, dataset_y)
    # print special_chars
    # regex for non-ascii chars
    unicode_regex = re.compile('[^\x00-\x7F]', re.IGNORECASE)

    printCsv(special_chars)


def loadCsv(training_x, training_y, testing_x):
    with open(training_x) as data_x:
        reader_x = csv.reader(data_x)
        dataset_x = list(reader_x)
        for i in range(1, len(dataset_x)):
            dataset_x[i][1] = dataset_x[i][1].lower()
    data_x.close()

    with open(training_y) as data_y:
        reader_y = csv.reader(data_y)
        dataset_y = list(reader_y)
    data_y.close()

    with open(testing_x) as test_x:
        reader_test_x = csv.reader(test_x)
        testset_x = list(reader_test_x)
    test_x.close()

    return dataset_x, dataset_y, testset_x

def printCsv(data):
    with open('special_chars.csv', 'wb') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Char', 'Slovak', 'French', 'Spanish', 'German', 'Polish'])
        for key, value in data.items():
            if len(value) == 1:
                writer.writerow([key, value[0]])
            else:
                if len(value) == 2:
                    writer.writerow([key, value[0], value[1]])
                elif len(value) == 3:
                    writer.writerow([key, value[0], value[1], value[2]])
                elif len(value) == 4:
                    writer.writerow([key, value[0], value[1], value[2], value[3]])
                elif len(value) == 5:
                    writer.writerow([key, value[0], value[1], value[2], value[3], value[4]])
    csv_file.close()

def getSpecialChars(training_x, dataset_y):
    # open training set with utf-8 encoding to indentify special chars
    # special_chars: dictionary {key=special_char, value=list of associated languages}
    unicode_regex = re.compile('[^\x00-\x7F]', re.IGNORECASE)
    special_chars = {}
    with codecs.open(training_x, mode='r', encoding='utf-8') as data_x:
        line_nb = -1
        for line in data_x:
            line_nb += 1
            # if line_nb > 1000:
            #     break
            # print line_nb
            for i in range(len(line)):
                if re.match(unicode_regex, line[i]):
                    if not line[i] in special_chars:
                        special_chars[line[i]] = [dataset_y[line_nb][1]]
                    else:
                        if not dataset_y[line_nb][1] in special_chars[line[i]]:
                            special_chars[line[i]].append(dataset_y[line_nb][1])
    data_x.close()

    return special_chars

if __name__ == '__main__':
    main()




#####OLD METHODS######
    # special_chars_short = {}
    # for i in range(len(dataset_x)):
    # # for i in range(100):
    #     for j in range(len(dataset_x[i][1])):
    #         if re.match(unicode_regex, dataset_x[i][1][j]):
    #             if dataset_x[i][1][j] in special_chars_short:
    #                 if dataset_y[i][1] in special_chars_short[dataset_x[i][1][j]]:
    #                     pass
    #                 else:
    #                     special_chars_short[dataset_x[i][1][j]].append(dataset_y[i][1])
    #             else:
    #                 special_chars_short[dataset_x[i][1][j]] = [dataset_y[i][1]]
    #                 print dataset_x[i][1][j]#.decode('utf-8')

    # print len(special_chars_short)



    # special_chars2 = []
    # with codecs.open(training_x, mode='r', encoding='utf-8') as f:
    #     text = f.read()
    #     for i in range(len(text)):
    #     # for i in range(300):
    #         if re.match(unicode_regex, text[i]):
    #             if text[i] in special_chars2:
    #                 # print text[i].decode('utf-8')
    #                 pass
    #             else:
    #                 special_chars2.append(text[i])
    #                 # print f.readline
    #                 # print str(i)+": "+text[i].decode('utf-8')
    #     f.close()

    # print len(special_chars2)