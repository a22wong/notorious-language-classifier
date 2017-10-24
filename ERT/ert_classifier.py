import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn
import itertools

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_extraction.text import CountVectorizer 

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import make_scorer
from sklearn.metrics import confusion_matrix

def main():
    



    print("Reading testing data...")

    # Testing data to record predictions 
    test_data = read_csv('../Data/test_set_x.csv')
    
   
    print("Reading training data...")

        # # test code -- uncomment in production
    training_data = read_csv('../Data/train_set_x.csv')
    training_labels = read_csv('../Data/train_set_y.csv', labels=True)
    
    # # read in & normalize data
    # training_data = read_csv('/Users/Dave/Documents/Uni_Work/COMP_551/notorious-language-classifier/Data/train_set_x.csv')
    
  
    # print("Reading training labels...")
    
    # # read in the labels
    # training_labels = read_csv('/Users/Dave/Documents/Uni_Work/COMP_551/notorious-language-classifier/Data/train_set_y.csv', labels=True)


    print("Constructing features...")

    vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 2), strip_accents='unicode')
    vectorizer.fit(training_data)

    training_X = vectorizer.transform(training_data)
    test_X = vectorizer.transform(test_data)


    print("Running cross validation ...")

    n_estimators = [20, 60, 100, 140, 180]

    # Number of estimators to try in cross validation parameters search
    param_grid = [
        {
            # 'reduce_dim__n_components': n_components,
            'n_estimators': n_estimators
        }
    ]

    # cross validation search for best number of estimators hyperparameter in Extra Randomized Trees
    grid = GridSearchCV(ExtraTreesClassifier(), param_grid=param_grid, cv=3, scoring=make_scorer(accuracy_score), n_jobs=-1)
    grid.fit(training_X, training_labels)

    print("The best estimator was:")
    
    print(grid.best_estimator_)
    
    print("With accuracy: {}".format(grid.best_score_))


    print("Printing graph data...")

    print_graphs(grid, training_X, training_labels, n_estimators)


    print("Making test set predictions and writing to file...")

    best_predictions = grid.predict(test_X)

    write_predictions_to_file(best_predictions)


    print("Finished writing to file.")


def print_confusion_matrix(y_actual, y_pred):
    
    # Print the confusion matrix 
    cnf_matrix = confusion_matrix(y_actual, y_pred)

    
    np.set_printoptions(precision=2)  

    plot_confusion_matrix(cnf_matrix, classes=['Slovak', 'French', 'Spanish', 'German', 'Polish'], normalize=True, title='Confusion Matrix')
    plt.show()
    plt.close(plt.figure())
    

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

def print_estimator_graph(validation_scores, training_scores, estimators):
    
    plt.figure(figsize=(10,6))
    plt.plot(estimators, validation_scores, 'o-', "g")
    plt.plot(estimators, training_scores, 'o-', color="r") 

    plt.title("The Effect that the Quantity of Estimators in an Extremely Randomized Tree has on Accuracy")
    plt.yscale('linear')

    plt.xlabel("Number of Estimators")
    plt.ylabel("Model Accuracy")
    plt.legend(["Validation Accuracy", "Training Accuracy"])
    plt.show()


        
def print_graphs(grid, training_data, training_labels, n_estimators):

    results = grid.cv_results_

    class_names = [ "Slovak", "French", "Spanish", "Polish" ]

    # Print the confusion matrix
    print_confusion_matrix(training_labels, grid.predict(training_data))

    # Print the graphs comparing the results of different numbers of estimators
    print_estimator_graph(results['mean_test_score'], results['mean_train_score'], n_estimators)






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
