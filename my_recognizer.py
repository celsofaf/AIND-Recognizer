import warnings
from asl_data import SinglesData


def recognize(models: dict, test_set: SinglesData):
    """ Recognize test word sequences from word models set

   :param models: dict of trained models
       {'SOMEWORD': GaussianHMM model object, 'SOMEOTHERWORD': GaussianHMM model object, ...}
   :param test_set: SinglesData object
   :return: (list, list)  as probabilities, guesses
       both lists are ordered by the test set word_id
       probabilities is a list of dictionaries where each key a word and value is Log Liklihood
           [{SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            {SOMEWORD': LogLvalue, 'SOMEOTHERWORD' LogLvalue, ... },
            ]
       guesses is a list of the best guess words ordered by the test set word_id
           ['WORDGUESS0', 'WORDGUESS1', 'WORDGUESS2',...]
   """
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # TODO implement the recognizer
    def argmax(d: dict):
        '''
        Given a dictionary of numeric values, returns the key corresponding to the max value
        '''
        return max(zip(d.values(), d.keys()))[1]
    
    probabilities = []
    guesses = []
    for word_id in sorted(list(test_set.get_all_Xlengths().keys())):
        X, lengths = test_set.get_item_Xlengths(word_id)
        scores = {}
        for word, model in models.items():
            try:
                scores[word] = model.score(X, lengths)
            except:  # if invalid model, or can't calculate score
                scores[word] = float('-inf')
        probabilities.append(scores)
        guesses.append(argmax(scores))
    
    # return probabilities, guesses
    return probabilities, guesses
