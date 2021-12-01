import gensim.downloader
import pandas as pd
import numpy as np
from gensim.models import keyedvectors
import fasttext

model_name = '''word2vec-google-news-300'''

# doing this so if i accidentally rerun it i dont waste time
try:
    model
except:
    if model_name == 'conceptnet-numberbatch-17-06-300':
        # good but words are so specific i haven't heard of many of them
        # too big brain, if you know the encyclopedia this is good. 
        # Maybe can use this if can sort by frequency properly
        model = gensim.downloader.load(model_name)
        en_filter = np.where(pd.Series(model.index_to_key).str.contains('/c/en'))
        model.vectors = np.array(model.vectors)[en_filter]
        model.index_to_key = np.array(model.index_to_key)[en_filter]
        model.index_to_key = [i[6:] for i in model.index_to_key]
        model.key_to_index = {key:i for i, key in enumerate(model.index_to_key)}
    elif model_name == 'word2vec-google-news-300':
        model = gensim.downloader.load(model_name)
    elif model_name == 'glove-twitter-200':
        # really good if you know basketball and other things twitter is talking about
        model = gensim.downloader.load(model_name)

# glove = gensim.downloader.load('glove-twitter-200')
        
lang_model = fasttext.load_model('lid.176.bin')

def get_clue(board, 
             picked_words, 
             model=model,
             primary_strategy='max_total_similarity', 
             secondary_strategy='max_total_similarity', 
             minimum_similarity=.315, 
             min_diff_from_neutral=0.02, 
             min_diff_from_negative=0.04,
             min_diff_from_assassin=0.07,
             restrict_vocab=10000,
             return_alternatives=False):
    '''
        board: {
                    'positive': 'bar drawing collar notre+dame nut thunder hammer sahara rust'.split(),
                    'negative': 'air kick scientist kung+fu fog pea aztec book'.split(),
                    'neutral': 'rodeo vet war bed bread port russia'.split(),
                    'assassin': 'shot'.split()
                }, 
        picked_words: ['thunder', 'drawing', 'collar'], 
        primary_strategy: ['max_total_similarity', 'max_n'], 
        secondary_strategy: ['max_total_similarity', 'max_min_adjusted_diff_from_last_positive', 'max_min_similarity', 'max_max_similarity'],
        minimum_similarity: 0 (minimum similarity of most similar word, domain (0,1),
        min_diff_from_neutral=0 (recommended 0-.02), 
        min_diff_from_negative=.05 (recommended (0-.02),
        min_diff_from_assassin=.1 (recommended 0-.05),
        restrict_vocab: 10000 (limits words that can be looked at),
        return_alternatives: False
    '''

    type_thresholds = {
        'positive': 0,
        'neutral': min_diff_from_neutral,
        'negative': min_diff_from_negative,
        'assassin': min_diff_from_assassin
    }
    
    for kind in board:
        board[kind] = [i for i in board[kind] if i not in picked_words]

    board_words = {j:i for i in board for j in board[i]}

    # add compound words if they dont exist
    calls = 0
    def get_mean_vector(words):
        return np.mean([model.get_vector(i) for i in words], axis=0)    

    for word in board_words.keys():
        if ' ' in word:
            if word in model.key_to_index:
                pass
            elif word.replace(' ','') in model.key_to_index:
                model.add_vector(word, model.get_vector(word.replace(' ','')))
            elif word.replace(' ','_') in model.key_to_index:
                model.add_vector(word, model.get_vector(word.replace(' ','_')))
            elif all([i in model.key_to_index for i in word.split(' ')]):
                model.add_vector(word, get_mean_vector(word.split(' ')))
            else:
                raise Exception(f"Don't know how to add word {word} to model")

    distances = None
    best_n = 0
    best_words = []

    just_board_words = list(board_words.keys())
    just_board_types = np.array(list(board_words.values()))

    board_thresholds = np.array([type_thresholds[i] for i in board_words.values()])

    max_total_word_similarity = -np.inf

    
    
    for i, word in enumerate(model.index_to_key[:restrict_vocab]):
        
        # check if english recognized word
        # https://stackoverflow.com/questions/39142778/python-how-to-determine-the-language
        special_characters = '''"!@#$%^&*()-+?_=,<>/"'''

        if any([i in word for i in special_characters]):
            continue
        if lang_model.predict(word, k=1)[0][0]!='__label__en':
            continue
        if any([word.lower() in i.lower() for i in just_board_words]):
            continue
        if any([i.lower() in word.lower() for i in just_board_words]):
            continue
        
        similarities = model.cosine_similarities(model.get_vector(word), [model.get_vector(i) for i in just_board_words])
        
        if similarities.max()<minimum_similarity:
            continue

        # need to change this to look at 
        # distance to next and stop 
        # based on dist to next bad less than threshold

        sorted_similarity_indicies = np.argsort(similarities)[::-1]
        types = just_board_types[sorted_similarity_indicies]
        
        if types[0]!='positive':
            continue
        
        sorted_similarities = similarities[sorted_similarity_indicies]
        
        if sorted_similarities[0]<minimum_similarity:
            continue
        
        thresholds = board_thresholds[sorted_similarity_indicies]

        first_neutral = np.where(types=='neutral')[0][0]
        first_negative = np.where(types=='negative')[0][0]
        first_assassin = np.where(types=='assassin')[0][0]

        first_neutral_similarity = sorted_similarities[first_neutral]
        first_negative_similarity = sorted_similarities[first_negative]
        first_assassin_similarity = sorted_similarities[first_assassin]

        min_positive_similarity = np.max([
                                        first_neutral_similarity + type_thresholds['neutral'],
                                        first_negative_similarity + type_thresholds['negative'],
                                        first_assassin_similarity + type_thresholds['assassin']
                                        ])

        n = sorted_similarities[(sorted_similarities>min_positive_similarity) & (sorted_similarities>minimum_similarity)].shape[0]
    
        similarity_sum = sorted_similarities[:n].sum()
        if primary_strategy=='max_total_similarity':
            if similarity_sum>max_total_word_similarity:
                max_total_word_similarity = similarity_sum
                best_n = n
                best_word = word
                best_words = [word]
        elif primary_strategy=='max_n':
            if n > best_n:
                calls += 1
                best_n = n
                best_words = [word]
            elif n == best_n:
                calls += 1 
                best_words.append(word)
        else:
            raise ValueError('primary_strategy invalid')

    best_word = None
    max_min_adjusted_diff_from_last_positive = 0
    max_min_similarity = 0
    max_max_similarity = 0
    max_total_similarity = 0
    for word in best_words:
        temp = pd.DataFrame({
                            'word': just_board_words,
                            'similarity': [model.similarity(word, i) for i in just_board_words],
                            'type': just_board_types,
                            'diff_thresholds': [type_thresholds[i] for i in just_board_types]
                    }).sort_values('similarity', ascending=False).reset_index(drop=True)

        first_positive_similarity = temp.iloc[0].similarity
        last_positive_similarity = temp.iloc[best_n-1].similarity
        total_similarity = temp.iloc[:best_n].similarity.sum()# - temp.iloc[best_n].similarity

        temp['diff_from_last_positive'] = last_positive_similarity - temp.similarity
        temp['diff_from_last_positive_adjusted'] = temp.diff_from_last_positive - temp.diff_thresholds
        min_adjusted_diff_from_last_positive = temp[temp.type!='positive'].diff_from_last_positive_adjusted.min()

        if secondary_strategy=='max_min_adjusted_diff_from_last_positive':
            if min_adjusted_diff_from_last_positive > max_min_adjusted_diff_from_last_positive:
                max_min_adjusted_diff_from_last_positive = min_adjusted_diff_from_last_positive
                best_word = word
        elif secondary_strategy=='max_min_similarity':
            if last_positive_similarity>max_min_similarity:
                max_min_similarity = last_positive_similarity
                best_word = word
        elif secondary_strategy=='max_max_similarity':
            if first_positive_similarity>max_max_similarity:
                max_max_similarity = first_positive_similarity
                best_word = word
        elif secondary_strategy=='max_total_similarity':
            if total_similarity>max_total_similarity:
                max_total_similarity = total_similarity
                best_word = word
        else:
            raise ValueError('secondary_strategy invalid')
    if return_alternatives:
        return best_word, best_n, [i for i in best_words if best_word!= i]
    else:
        return best_word, best_n

# might be good to condition on similarity > .2 OR distance < 15
# similarity / distance?

def get_analysis(word, board, picked_words, model=model):
    just_board_words = [j for i in board.values() for j in i]
    just_board_types = [key for key in board for value in board[key]]
    return pd.DataFrame({
                    'word': just_board_words,
                    'cosine_similarity': [model.similarity(word, i) for i in just_board_words],
                    'euclidean_distance': [np.sum(np.square(model.get_vector(word) - model.get_vector(i))) for i in just_board_words],
                    'type': just_board_types
            }).sort_values('cosine_similarity', ascending=False).reset_index(drop=True)
