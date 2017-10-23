import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def main():
    
    # # # test code -- uncomment in production
    # training_data = read_csv('../Data/train_set_x.csv')
    # training_labels = read_csv('../Data/train_set_y.csv', labels=True)

    # Testing data to record predictions 
    test_data = read_csv('/Users/Dave/Documents/Uni_Work/COMP_551/notorious-language-classifier/Data/test_set_x.csv')
    
   
    print("Reading training data...")
    
    # read in & normalize data
    training_data = read_csv('/Users/Dave/Documents/Uni_Work/COMP_551/notorious-language-classifier/Data/train_set_x.csv')
    
  
    print("Reading training labels...")
    
    # read in the labels
    training_labels = read_csv('/Users/Dave/Documents/Uni_Work/COMP_551/notorious-language-classifier/Data/train_set_y.csv', labels=True)


    print("Constructing features...")

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2), strip_accents='unicode')
    vectorizer.fit(training_data)

    training_X = vectorizer.transform(training_data)
    test_X = vectorizer.transform(test_data)


    print("Running cross validation ...")

    pipe = Pipeline([
        ('reduce_dim', PCA()),
        ('classify', ExtraTreesClassifier())
    ])

    n_estimators = [20, 40, 60, 80, 100]
    n_features = [1, 5, 10, 15, 20]

    # Number of estimators to try in cross validation parameters search
    search_parameters = [
        {
            'n_estimators': n_estimators,
            'reduce_dim': [PCA(iterated_power='7')]
        }
    ]

    # cross validation search for best number of estimators hyperparameter in Extra Randomized Trees
    ert = GridSearchCV(ExtraTreesClassifier(), search_parameters, cv=3, scoring=make_scorer(accuracy_score), n_jobs=-1)
    ert.fit(training_X, training_labels)

    print("The best estimator was:")
    print(ert.best_estimator_)
    
    print("With accuracy: {}".format(ert.best_score_))

    print("Making test set predictions and writing to file...")
    best_predictions = ert.predict(test_X)

    write_predictions_to_file(best_predictions)

    print("Finished writing to file.")

    
def write_predictions_to_file(predictions):
    
    with open("test_set_y.csv", 'w') as f:

        f.write('Id,Category\n')
    
        for idx, prediction in enumerate(predictions):
            line = '{},{}\n'.format(idx, prediction)
            f.write(line)
    
        f.close()

# read the csv file 
def read_csv(filepath, labels=False):
    csv_data = []

    counter = 0;

    with open(filepath) as f:

        # skip the first line given that it is the header
        next(f)

        for line in f:
            
            # counter += 1
            
            # if (counter == 1000):
            #     break

            # split the row 
            row_data = line.split(',')

            # string format the row
            row_data = int(row_data[1]) if labels else preprocess_data(row_data[1])
            csv_data.append(row_data)

    return csv_data

# do some early preprocessing to the data
def preprocess_data(row_data):

    emoji_pattern = re.compile(
        u"(\ud83d[\ude00-\ude4f])|"  # emoticons
        u"(\ud83c[\udf00-\uffff])|"  # symbols & pictographs (1 of 2)
        u"(\ud83d[\u0000-\uddff])|"  # symbols & pictographs (2 of 2)
        u"(\ud83d[\ude80-\udeff])|"  # transport & map symbols
        u"(\ud83c[\udde0-\uddff])"  # flags (iOS)
        "+", flags=re.UNICODE)
    
    # Remove emoji and other symbols from text
    emoji_pattern.sub('', row_data);

    # lowercase, remove newline character, and filter out any words smaller than 4 characters 
    return row_data


if __name__ == '__main__':
    main()
