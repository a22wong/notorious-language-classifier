import csv
import codecs
import re

def main():
    training_x = '../Data/train_set_x.csv'
    training_y = '../Data/train_set_y.csv'
    testing_x = '../Data/test_set_x.csv'

    encodeMyFile(training_x,'train_set_x_utf8.csv')
    encodeMyFile(testing_x, 'test_x_utf8.csv')



def encodeMyFile(input, output):
    print "Encoding..."
    with codecs.open(input, mode='r', encoding='utf-8') as sourcefile:
        with codecs.open(output, mode='w', encoding='utf-8') as tartgetfile:
            while True:
                contents = sourcefile.readline()
                if not contents:
                    break
                tartgetfile.write(contents)

if __name__ == '__main__':
	main()