import os
import pandas as pd
import re
import numpy as np
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import precision_recall_fscore_support, accuracy_score,classification_report
from sklearn import tree
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from nltk.stem.porter import *


def sentence_filter(sentence):
    '''1. replace the following with spaces --, ~, ...'''

    sentence = re.sub('\-{2,}|\.{3,}|\~',' ', sentence)
    
    '''2. remove tags <words> and all other symbols'''
    
    #remove tags
    sentence = re.sub('<(.*?)+>', '', sentence)
    #deleting all other symbols
    sentence = re.sub('[^A-Za-z0-9 \/\-\$\%]+','', sentence)
    
    '''3. filter each word to be more than 2 letters,numbers,specific symbols (/-$%)'''
    output = sentence.split()
    output = [x for x in output if len(x) >= 2]
    
    return ' '.join(output)


def special_sentence_filter(sentence):
    sentence = re.sub('\-{2,}|\.{3,}|\~',' ', sentence)
    
    #remove tags
    sentence = re.sub('<(.*?)+>', '', sentence)
    #deleting all other symbols
    sentence = re.sub('[^A-Za-z \/\-\$\%]+',' ', sentence)
    
    output = sentence.split()
    # print(output)
    output = [x for x in output if len(x) >= 3]
    
    return ' '.join(output)
    

def predict_and_test(model, X_test_bag_of_words, y_test, output = 'report'):
    num_dec_point = 3
    predicted_y = model.predict(X_test_bag_of_words)
    
    if output == 'report':
        a_mic = accuracy_score(y_test, predicted_y)
        p_mic, r_mic, f1_mic, _ = precision_recall_fscore_support(y_test, 
                            predicted_y,
                            average='micro',
                            warn_for=())
        p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(y_test, 
                            predicted_y,
                            average='macro',
                            warn_for=())
        print('micro acc,prec,rec,f1: ',round(a_mic,num_dec_point), round(p_mic,num_dec_point), round(r_mic,num_dec_point), round(f1_mic,num_dec_point),sep="\t")
        print('macro prec,rec,f1: ',round(p_mac,num_dec_point), round(r_mac,num_dec_point), round(f1_mac,num_dec_point),sep="\t")
        
        try:
            target_names = [str(x) for x in [1,2,3,4,5]]
            print(classification_report(y_test, predicted_y, target_names=target_names))
        except:
            target_names = ['negative','neutral','positive']
            print(classification_report(y_test, predicted_y, target_names=target_names))
        
        plot_classification_report(y_test, predicted_y)
        return [[round(a_mic,num_dec_point), round(p_mic,num_dec_point), round(r_mic,num_dec_point), round(f1_mic,num_dec_point)],[round(p_mac,num_dec_point), round(r_mac,num_dec_point), round(f1_mac,num_dec_point)]]
    
    elif output == 'result':
        return predicted_y

#printing output
def print_output(index_list, result_list):
    for x in range(len(result_list)):
        print(f'{index_list[x]} {result_list[x]}')

def bnb(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----bnb standard")
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(lowercase = False, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = BernoulliNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def bnb_top_1000(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----bnb top 1000 words")
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(max_features = 1000, lowercase = False, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = BernoulliNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def bnb_lowercase(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----bnb lowercase")
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(lowercase = True, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = BernoulliNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def bnb_porterstem(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----bnb porterstem")

    stemmer = PorterStemmer()
    stemmed_X_train = [stemmer.stem(plural) for plural in X_train]
    
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(stop_words = 'english', lowercase = False, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(stemmed_X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = BernoulliNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def mnb(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----mnb standard")
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(lowercase = False, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = MultinomialNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def mnb_top_1000(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----mnb top 1000 words")
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(max_features = 1000, lowercase = False, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = MultinomialNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def mnb_lowercase(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----mnb lowercase")
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(lowercase = True, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = MultinomialNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def mnb_porterstem(X_train, X_test, y_train, y_test, output = 'report'):
    if output == 'report':
            print("----mnb porter stem")

    stemmer = PorterStemmer()
    stemmed_X_train = [stemmer.stem(plural) for plural in X_train]
    
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(stop_words = 'english', lowercase = False, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(stemmed_X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = MultinomialNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def dt(X_train, X_test, y_train, y_test, output = 'report'):
    # if random_state id not set. the feaures are randomised, therefore tree may be different each time
    if output == 'report':
        print("----dt standard")
    # create count vectorizer and fit it with training data
    count = CountVectorizer(lowercase = False, max_features=1000, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    # clf = tree.DecisionTreeClassifier(min_samples_leaf=1,criterion='entropy',random_state=0)
    # print('min_sample_leaf:', int(0.01 * len(X_train)))
    # print(type(X_train))
    clf = tree.DecisionTreeClassifier(min_samples_leaf=0.01,criterion='entropy',random_state=0)
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def dt_without_1(X_train, X_test, y_train, y_test, output = 'report'):
    # if random_state id not set. the feaures are randomised, therefore tree may be different each time
    print("----dt without 1% criterion")
    # create count vectorizer and fit it with training data
    count = CountVectorizer(lowercase = False, max_features=1000, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def dt_lowercase(X_train, X_test, y_train, y_test, output = 'report'):
    # if random_state id not set. the feaures are randomised, therefore tree may be different each time
    print("----dt lowercase")
    # create count vectorizer and fit it with training data
    count = CountVectorizer(lowercase = True, max_features=1000, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    X_train_bag_of_words = count.fit_transform(X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = tree.DecisionTreeClassifier(criterion='entropy',random_state=0)
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def dt_porterstem(X_train, X_test, y_train, y_test, output = 'report'):
    # if random_state id not set. the feaures are randomised, therefore tree may be different each time
    if output == 'report':
        print("----dt porterstem")
    stemmer = PorterStemmer()
    stemmed_X_train = [stemmer.stem(plural) for plural in X_train]
    
    # create count vectorizer and fit it with training data, and transform X_test
    count = CountVectorizer(stop_words = 'english', lowercase = False, max_features = 1000, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    # print(count.get_)
    X_train_bag_of_words = count.fit_transform(stemmed_X_train)
    X_test_bag_of_words = count.transform(X_test)
    clf = tree.DecisionTreeClassifier(min_samples_leaf=0.01,criterion='entropy',random_state=0)
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    
    return result

def my_model(X_train, X_test, y_train, y_test, output = 'report', clf = MultinomialNB()):
    print("----My model")
    
    stemmer = PorterStemmer()
    stemmed_X_train = [stemmer.stem(plural) for plural in X_train]
    
    count = CountVectorizer(stop_words = 'english', lowercase = True, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    # count = CountVectorizer(stop_words = 'english', lowercase = True, token_pattern = r'(?u)[a-zA-Z]{3,}')
    X_train_bag_of_words = count.fit_transform(stemmed_X_train)# transform the test data into bag of words created with fit_transform
    X_test_bag_of_words = count.transform(X_test)
    
    # clf = MultinomialNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    return result

def my_model_modified(X_train, X_test, y_train, y_test, output = 'report', clf = MultinomialNB()):
    print("----My model")
    
    stemmer = PorterStemmer()
    stemmed_X_train = [stemmer.stem(plural) for plural in X_train]
    
    # count = CountVectorizer(stop_words = 'english', lowercase = True, token_pattern = r'(?u)[a-zA-Z0-9-/$%]{2,}')
    count = CountVectorizer(stop_words = 'english', lowercase = True, token_pattern = r'(?u)[a-zA-Z]{3,}')
    X_train_bag_of_words = count.fit_transform(stemmed_X_train)# transform the test data into bag of words created with fit_transform
    X_test_bag_of_words = count.transform(X_test)
    
    # clf = MultinomialNB()
    model = clf.fit(X_train_bag_of_words, y_train)
    result = predict_and_test(model, X_test_bag_of_words, y_test, output)
    return result

def sentiment_transform(df):
    def new_class_distribution(number):
        if number in [1,2,3]:
            return -1
        elif number == 4:
            return 0
        elif number == 5:
            return 1
    
    df['sentiment'] = df['rating'].apply(lambda x: new_class_distribution(x)) #apply lambda filter to column


def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

    plt.figure(figsize=figsize)

    xticks = ['precision', 'recall', 'f1-score', 'support']
    yticks = list(np.unique(y_tru))
    yticks += ['avg']

    rep = np.array(precision_recall_fscore_support(y_tru, y_prd)).T
    avg = np.mean(rep, axis=0)
    avg[-1] = np.sum(rep[:, -1])
    rep = np.insert(rep, rep.shape[0], avg, axis=0)

    res = sns.heatmap(rep,
                cmap = "Blues",
                vmin = 0,
                vmax = 1,
                annot=True, 
                annot_kws = {"size": 30},
                cbar=True, 
                xticklabels=xticks, 
                yticklabels=yticks,
                ax=ax)
    res.set_xticklabels(res.get_xmajorticklabels(), fontsize = 26)
    res.set_yticklabels(res.get_ymajorticklabels(), fontsize = 26, rotation = 45)

def plot_double_result(rA, rB, title, legend_1, legend_2):
    result_A = rA[0] + rA[1]
    result_A = [100*x for x in result_A]
    
    result_B = rB[0] + rB[1]
    result_B = [100*x for x in result_B]
    
    # set width of bar
    barWidth = 0.4
    fig = plt.subplots(figsize =(12, 8))
    
    # Set position of bar on X axis
    br1 = np.arange(len(result_A))
    br2 = [x + barWidth for x in br1]
    
    # Make the plot
    plt.bar(br1, result_A, color ='g', width = barWidth,
            edgecolor ='grey', label = legend_1)
    plt.bar(br2, result_B, color ='r', width = barWidth,
            edgecolor ='grey', label = legend_2)
    
    # Adding Xticks
    plt.xlabel('Micro and Macro', fontweight ='bold', fontsize = 20)
    plt.ylabel('Percentage (%)', fontweight ='bold', fontsize = 20)
    plt.xticks([r + barWidth/2 for r in range(len(result_A))],
            ['Micro:Accuracy','Micro:Precision','Micro:Recall','Micro:F1','Macro:Precision','Macro:Recall','Macro:F1'])
    x1 = list(range(len(result_A)))
    x2 = list(range(len(result_B)))
    plt.legend(fontsize = 18, loc = 4)
    import seaborn as sns
    sns.set(style="whitegrid")
    axes = plt.gca()
    axes.set_ylim([0,100])
    axes.tick_params(axis='x', labelsize=15)
    axes.tick_params(axis='y', labelsize=20)
    plt.title(title, fontweight ='bold', fontsize = 30)
    # frame_scores.plot.bar(ax = ax, cmap = 'RdYlBu', edgecolor = "black")
    for a, b in zip(x1, result_A):
        # print(a,b)
        plt.text(a, b+0.05, '%.1f' % b, ha='center', va='bottom', fontsize=18)
    for a, b in zip(x2, result_B):
        # print(a,b)
        plt.text(a+barWidth, b+0.05, '%.1f' % b, ha='center', va='bottom', fontsize=18)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(y_test):
    unique, count = np.unique(y_test,return_counts = True)
    plt.figure()
    if len(unique) == 3:
        plt.bar(['neg','neut','pos'], count)
    else:
        plt.bar(unique, count)
    plt.title('Class Distribution', fontweight ='bold', fontsize = 20)
    plt.xlabel('Class', fontweight ='bold', fontsize = 20)
    plt.ylabel('Frequency', fontweight ='bold', fontsize = 20)

#%%

'''Report analysis'''

if __name__ == "__main__":
    sklearn_site_joblib=True
    hard_file = True
    model = 'PorterStem'
    plot_class_distribution = False
    
    
    if hard_file:
        base_location = 'F:/UNSW/2022/T2_2022/COMP9414/Assignment_2/'
        os.chdir(base_location)
        excel_file = base_location + 'reviews.tsv'
        
        #read from tsv and get X and Y
        df = pd.read_csv(excel_file, sep='\t', header=None)
    
    else:
        df = pd.read_csv(sys.stdin, sep='\t', header=None)
    
    df = df.rename(columns={0: 'index', 1: 'rating', 2: 'sentence'} )
    
    if model != 'my_model':    
        df['sentence'] = df['sentence'].apply(lambda x: sentence_filter(x)) #apply lambda filter to column
    elif model == 'my_model':
        df['sentence'] = df['sentence'].apply(lambda x: special_sentence_filter(x))

    
    sentiment_transform(df)
    X = df['sentence'].to_numpy()
    indices = df['index'].to_numpy()
    
    for i in ['rating','sentiment']:
        print('\n' + i + '=============================================')
        y = df[i].to_numpy()
    
        # split into train and test
        split_percentage = 0.8
        split_point = int(len(X) * split_percentage)
        X_train = X[:split_point]
        X_test = X[split_point:]
        y_train = y[:split_point]
        y_test = y[split_point:]
        indices_test = indices[split_point:]
        
        if plot_class_distribution:
            plot_class_distribution(y_test) #plots the class distribution
        
        '''models'''
        
        if model == 'DT':
            result_1 = dt(X_train, X_test, y_train, y_test, 'report')
            result_2 = dt_without_1(X_train, X_test, y_train, y_test , 'report')
            
            plot_double_result(result_1, result_2, 
                               'Classification Report - Decision Tree (' + i + ')',
                               'Standard Model',
                               'Without 1% Criterion')
        elif model == 'BNB':
            result_1 = bnb(X_train, X_test, y_train, y_test, 'report')
            result_2 = bnb_top_1000(X_train, X_test, y_train, y_test , 'report')
            
            plot_double_result(result_1, result_2,
                               'Classification Report - ' + model + ' (' + i + ')',
                               'Standard Model',
                               'Top 1000 words')
        elif model == 'MNB':
            result_1 = mnb(X_train, X_test, y_train, y_test, 'report')
            result_2 = mnb_top_1000(X_train, X_test, y_train, y_test , 'report')
            
            plot_double_result(result_1, result_2, 
                               'Classification Report - ' + model + ' (' + i + ')',
                               'Standard Model',
                               'Top 1000 words')
            
        elif model == 'PorterStem':
            '''question 3'''       
            from nltk.stem.porter import *
            result_1 = dt(X_train, X_test, y_train, y_test, 'report')
            result_2 = bnb(X_train, X_test, y_train, y_test, 'report')
            result_3 = mnb(X_train, X_test, y_train, y_test, 'report')
            
            result_4 = dt_porterstem(X_train, X_test, y_train, y_test, 'report')
            result_5 = bnb_porterstem(X_train, X_test, y_train, y_test, 'report')
            result_6 = mnb_porterstem(X_train, X_test, y_train, y_test, 'report')
            
            print('standard')
            print('dt', result_1[0] + result_1[1])
            print('bnb', result_2[0] + result_2[1])
            print('mnb', result_3[0] + result_3[1])
            
            print('porterstem')
            print('dt', result_4[0] + result_4[1])
            print('bnb', result_5[0] + result_5[1])
            print('mnb', result_6[0] + result_6[1])
            
        
        elif model == 'lowercase':
            '''question 4'''
            result_1 = dt(X_train, X_test, y_train, y_test, 'report')
            result_2 = bnb(X_train, X_test, y_train, y_test, 'report')
            result_3 = mnb(X_train, X_test, y_train, y_test, 'report')
            
            result_4 = dt_lowercase(X_train, X_test, y_train, y_test, 'report')
            result_5 = bnb_lowercase(X_train, X_test, y_train, y_test, 'report')
            result_6 = mnb_lowercase(X_train, X_test, y_train, y_test, 'report')
            
            print('standard')
            print('dt', result_1[0] + result_1[1])
            print('bnb', result_2[0] + result_2[1])
            print('mnb', result_3[0] + result_3[1])
            
            print('lowercase')
            print('dt', result_4[0] + result_4[1])
            print('bnb', result_5[0] + result_5[1])
            print('mnb', result_6[0] + result_6[1])
        
        elif model == 'my_model':
            '''question 5'''
            result_1 = mnb(X_train, X_test, y_train, y_test, 'report')
            result_2 = my_model_modified(X_train, X_test, y_train, y_test, 'report', clf = MultinomialNB())

            print('my_model')
            print('mnb', result_1[0] + result_1[1])
            print('mnb_modified', result_2[0] + result_2[1])

            plot_double_result(result_1, result_2, 
                    'Classification Report - ' + model + ' (' + i + ')',
                    'Standard MNB Model',
                    'My Model')
    
