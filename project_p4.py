from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np
import pickle as pkl
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import string
import re
import csv
import nltk

EMBEDDING_FILE = "w2v.pkl"

def load_w2v(filepath):
    with open(filepath, 'rb') as fin:
        return pkl.load(fin)

def load_as_list(fname):
    df = pd.read_csv(fname)
    documents = df['review'].values.tolist()
    labels = df['label'].values.tolist()
    return documents, labels

def extract_user_info(user_input):
    name = ""
    pattern = re.compile(r"(\s|^)([A-Z][A-Za-z\.\-&']*(\s|$)){2,4}")
    match = re.search(pattern, user_input)
    if match:
        name = match[0].strip()
    return name

def get_tokens(inp_str):
    return inp_str.split()

def preprocessing(user_input):
    modified_input = ""
    punctuation = re.compile(r"^\s*(!|\"|#|\$|%|&|'|\(|\)|\*|\+|,|-|\.|\/|:|;|<|=|>|\?|@|\[|\]|\^|_|`|{|\||}|~|'|\\)\s*$")
    tokens = get_tokens(user_input)
    valid = []
    for i in range(0, len(tokens)):
        t = tokens[i]
        if not re.match(punctuation, t):
            valid.append(t.lower())
    for i in range(0, len(valid)):
        t = valid[i]
        modified_input += t
        if i < len(valid) - 1:
            modified_input += " "
    return modified_input

def vectorize_train(training_documents):
    # Initialize the TfidfVectorizer model and document-term matrix
    vectorizer = TfidfVectorizer()
    tfidf_train = None
    for i in range(len(training_documents)):
        training_documents[i] = (preprocessing(training_documents[i])) # preprocess training docs
    tfidf_train = vectorizer.fit_transform(training_documents) # train vectorizer
    return vectorizer, tfidf_train


def vectorize_test(vectorizer, user_input):

    tfidf_test = None
    tfidf_test = vectorizer.transform([preprocessing(user_input)])
    return tfidf_test

def train_nb_model(training_data, training_labels):
    # Initialize the GaussianNB model and the output label
    nb_model = GaussianNB()

    dense_rep = training_data.toarray()
    nb_model.fit(dense_rep,training_labels)

    return nb_model
def get_model_prediction(nb_model, tfidf_test):
    label = 0


    dense = tfidf_test.toarray()
    label = nb_model.predict(dense)

    return label


def w2v(word2vec, token):

    word_vector = np.zeros(300,)
    try:
        word_vector = word2vec[token]
        return word_vector
    except KeyError:
        return word_vector


def string2vec(word2vec, user_input):

    embedding = np.zeros(300,)


    pp_input=preprocessing(user_input)
    pp_tokens = get_tokens(pp_input)

    embeddings=np.empty(shape=(len(pp_tokens),300))
    
    for i, token in enumerate(pp_tokens):
        embeddings[i] = w2v(word2vec,token)
    embedding = np.mean(embeddings, axis=0)
    return embedding


def instantiate_models():

    logistic = None
    svm = None
    mlp = None
    logistic = LogisticRegression(random_state=100)
    svm = LinearSVC(random_state=100)
    mlp = MLPClassifier(random_state=100)
    return logistic, svm, mlp


def train_model(model, word2vec, training_documents, training_labels):

    # Write your code here:
    embeddings = [string2vec(word2vec,doc) for doc in training_documents]
    model.fit(embeddings, training_labels)

    return model


def test_model(model, word2vec, test_documents, test_labels):

    precision = None
    recall = None
    f1 = None
    accuracy = None

    embeddings = [string2vec(word2vec,doc) for doc in test_documents]
    predictions = model.predict(embeddings)
    precision = precision_score(test_labels, predictions)
    recall = recall_score(test_labels, predictions)
    f1 = f1_score(test_labels, predictions)
    accuracy = accuracy_score(test_labels, predictions)

    return precision, recall, f1, accuracy


def count_words(user_input):

    num_words = 0
    tokens = nltk.tokenize.word_tokenize(user_input)
    tokens = [t for t in tokens if t not in string.punctuation]
    num_words = len(tokens)
    return num_words

def words_per_sentence(user_input):

    wps = 0.0
    sentences = nltk.tokenize.sent_tokenize(user_input)
    total_words = sum([count_words(s) for s in sentences])
    wps = total_words / len(sentences)
    return wps

def get_pos_tags(user_input):

    tagged_input = []
    tokens = get_tokens(user_input)
    tagged_input = nltk.pos_tag(tokens)
    return tagged_input

def get_pos_categories(tagged_input):

    num_pronouns = 0
    num_prp = 0
    num_articles = 0
    num_past = 0
    num_future = 0
    num_prep = 0
    for tag_pair in tagged_input:
        tag = tag_pair[1]
        if tag in ['PRP','PRP$','WP','WP$']:
            num_pronouns+=1
            if tag not in ['PRP$','WP','WP$']:
                num_prp+=1
        elif tag == 'DT':
            num_articles+=1
        elif tag in ['VBD','VBN']:
            num_past+=1
        elif tag == 'MD':
           num_future+=1 
        elif tag == 'IN':
            num_prep+=1
    return num_pronouns, num_prp, num_articles, num_past, num_future, num_prep

def count_negations(user_input):

    num_negations = 0
    # [r"(\s|^)not", r"(\s|^)no(\s|$)", r"n\'t(\s|$)", r"(\s|^)never(\s|$)"]
    for n in [r"(\s|^)not(\s|$)",r"(\s|^)no(\s|$)", r"\w*n\'t", r"never"]:
        num_negations+= len(re.findall(n,user_input))
    return num_negations

def summarize_analysis(num_words, wps, num_pronouns, num_prp, num_articles, num_past, num_future, num_prep, num_negations):
    informative_correlates = []

    psychological_correlates = {}
    psychological_correlates["num_words"] = "Talkativeness, verbal fluency"
    psychological_correlates["wps"] = "Verbal fluency, cognitive complexity"
    psychological_correlates["num_pronouns"] = "Informal, personal"
    psychological_correlates["num_prp"] = "Personal, social"
    psychological_correlates["num_articles"] = "Use of concrete nouns, interest in objects/things"
    psychological_correlates["num_past"] = "Focused on the past"
    psychological_correlates["num_future"] = "Future and goal-oriented"
    psychological_correlates["num_prep"] = "Education, concern with precision"
    psychological_correlates["num_negations"] = "Inhibition"

    # Set thresholds
    features_keys = []
    features = [("num_pronouns",num_pronouns,6), 
                ("num_prp",num_prp,5), 
                ("num_articles",num_articles,4), 
                ("num_past",num_past,3), 
                ("num_future",num_future,2),
                ("num_prep",num_prep,1),
                ("num_negations",num_negations,0)
                ]
    num_words_threshold = 100
    wps_threshold = 20
    if num_words > num_words_threshold:
        features_keys.append("num_words")
    if wps > wps_threshold:
        features_keys.append("wps")
    features.sort(key= lambda x:(x[1], x[2]),reverse=True)
    top_features = features[0:(3-len(features_keys))]
    for f in top_features:
        features_keys.append(f[0])
    for k in features_keys:
        informative_correlates.append(psychological_correlates[k])
    return informative_correlates


def welcome_state():
    print("Hello and welcome to the CS 421 chatbot!\n")

    return "get_info_state"

def get_name_state():
    user_input = input("What is your name?\n")

    name = extract_user_info(user_input)

    user_input = print(f"Thanks {name}!")

    return "sentiment_analysis"

def sentiment_analysis_state(model, word2vec, first_time=False):
    user_input = input("What do you want to talk about today?\n")

    w2v_test = string2vec(word2vec, user_input)

    label = None
    label = mlp.predict(w2v_test.reshape(1, -1))

    if label == 0:
        print("Hmm, it seems like you're feeling a bit down.")
    elif label == 1:
        print("It sounds like you're in a positive mood!")
    else:
        print("Hmm, that's weird.  My classifier predicted a value of: {0}".format(label))
    if first_time:
        return "stylistic_analysis"
    else:
        return "check_next_state"

def stylistic_analysis_state():
    user_input = input("I'd also like to do a quick stylistic analysis. What's on your mind today?\n")

    num_words = count_words(user_input)
    wps = words_per_sentence(user_input)
    pos_tags = get_pos_tags(user_input)
    num_pronouns, num_prp, num_articles, num_past, num_future, num_prep = get_pos_categories(pos_tags)
    num_negations = count_negations(user_input)


    informative_correlates = summarize_analysis(num_words, wps, num_pronouns,
                                                num_prp, num_articles, num_past,
                                                num_future, num_prep, num_negations)
    print("Thanks!  Based on my stylistic analysis, I've identified the following psychological correlates in your response:")
    for correlate in informative_correlates:
        print("- {0}".format(correlate))


    return "check_next_state"
    
def check_next_state():
    next_state = ""
    options = [r'sentiment_analysis', r'stylistic_analysis', r'quit']
    next_state = input("Would you like to keep talking or redo the stylistic or sentiment analysis? (Enter sentiment_analysis, stylistic_analysis, or quit)\n")
    for regex in options:
        if re.search(regex,next_state):
            return str(regex)
    return "quit"

def run_chatbot(model, word2vec):
    first_time=True
    state = "welcome_state"
    while state != "quit":
        if state == "welcome_state":
            state = welcome_state()
        elif state == "get_info_state":
            state = get_name_state()
        elif state == "check_next_state":
            state = check_next_state()
        elif state == "sentiment_analysis":
            state = sentiment_analysis_state(model,word2vec,first_time)
            first_time=False
        elif state == "stylistic_analysis":
            state = stylistic_analysis_state()
    print("Goodbye!")


# ----------------------------------------------------------------------------




if __name__ == "__main__":

    documents, labels = load_as_list("dataset.csv")

    word2vec = load_w2v(EMBEDDING_FILE)

    logistic, svm, mlp = instantiate_models()
    logistic = train_model(logistic, word2vec, documents, labels)
    svm = train_model(svm, word2vec, documents, labels)
    mlp = train_model(mlp, word2vec, documents, labels)

    print("Word2Vec embedding for {0}:\t{1}".format("vaccine", w2v(word2vec, "vaccine")))

    test_documents, test_labels = load_as_list("test.csv")
    models = [logistic, svm, mlp]
    model_names = ["Logistic Regression", "SVM", "Multilayer Perceptron"]
    outfile = open("classification_report.csv", "w", newline='\n')
    outfile_writer = csv.writer(outfile)
    outfile_writer.writerow(["Name", "Precision", "Recall", "F1", "Accuracy"]) # Header row
    i = 0
    while i < len(models): # Loop through other results
        p, r, f, a = test_model(models[i], word2vec, test_documents, test_labels)
        if models[i] == None: # Models will be None if functions have not yet been implemented
            outfile_writer.writerow([model_names[i],"N/A", "N/A", "N/A", "N/A"])
        else:
            outfile_writer.writerow([model_names[i], p, r, f, a])
        i += 1
    outfile.close()

    vectorizer, tfidf_train = vectorize_train(documents)
    lexicon = [preprocessing(d) for d in test_documents]
    tfidf_test = vectorizer.transform(lexicon)
    naive = train_nb_model(tfidf_train, labels)
    predictions = naive.predict(tfidf_test.toarray())
    acc = np.sum(np.array(test_labels) == predictions) / len(test_labels)
    print("Naive Bayes Accuracy:", acc)

    run_chatbot(mlp, word2vec)