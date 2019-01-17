import numpy as np
import pandas as pd
import multiprocessing as mp
from time import time
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
import re
import gc
import os
from pandas.api.types import is_numeric_dtype, is_categorical_dtype
from functools import partial
import wordbatch
from wordbatch.extractors import WordBag
from wordbatch.models import FM_FTRL

INPUT_PATH = r'../input'

os.environ['MKL_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['JOBLIB_START_METHOD'] = 'forkserver'

def dameraulevenshtein(seq1, seq2):
    oneago = None
    thisrow = list(range(1, len(seq2) + 1)) + [0]
    for x in range(len(seq1)):
        twoago, oneago, thisrow = (oneago, thisrow, [0] * len(seq2) + [x + 1])
        for y in range(len(seq2)):
            delcost = oneago[y] + 1
            addcost = thisrow[y - 1] + 1
            subcost = oneago[y - 1] + (seq1[x] != seq2[y])
            thisrow[y] = min(delcost, addcost, subcost)
            if (x > 0 and y > 0 and seq1[x] == seq2[y - 1]
                and seq1[x - 1] == seq2[y] and seq1[x] != seq2[y]):
                thisrow[y] = min(thisrow[y], twoago[y - 2] + 1)
    return thisrow[len(seq2) - 1]


class SymSpell:
    def __init__(self, max_edit_distance=3, verbose=0):
        self.max_edit_distance = max_edit_distance
        self.verbose = verbose
        self.dictionary = {}
        self.longest_word_length = 0

    def get_deletes_list(self, w):
        deletes = []
        queue = [w]
        for d in range(self.max_edit_distance):
            temp_queue = []
            for word in queue:
                if len(word) > 1:
                    for c in range(len(word)):  # character index
                        word_minus_c = word[:c] + word[c + 1:]
                        if word_minus_c not in deletes:
                            deletes.append(word_minus_c)
                        if word_minus_c not in temp_queue:
                            temp_queue.append(word_minus_c)
            queue = temp_queue

        return deletes

    def create_dictionary_entry(self, w):
        new_real_word_added = False
        if w in self.dictionary:
            self.dictionary[w] = (self.dictionary[w][0], self.dictionary[w][1] + 1)
        else:
            self.dictionary[w] = ([], 1)
            self.longest_word_length = max(self.longest_word_length, len(w))

        if self.dictionary[w][1] == 1:
            new_real_word_added = True
            deletes = self.get_deletes_list(w)
            for item in deletes:
                if item in self.dictionary:
                    self.dictionary[item][0].append(w)
                else:
                    self.dictionary[item] = ([w], 0)

        return new_real_word_added

    def create_dictionary_from_arr(self, arr, token_pattern=r'[a-z]+'):
        total_word_count = 0
        unique_word_count = 0

        for line in arr:
            words = re.findall(token_pattern, line.lower())
            for word in words:
                total_word_count += 1
                if self.create_dictionary_entry(word):
                    unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def create_dictionary(self, fname):
        total_word_count = 0
        unique_word_count = 0

        with open(fname) as file:
            for line in file:
                # separate by words by non-alphabetical characters
                words = re.findall('[a-z]+', line.lower())
                for word in words:
                    total_word_count += 1
                    if self.create_dictionary_entry(word):
                        unique_word_count += 1

        print("total words processed: %i" % total_word_count)
        print("total unique words in corpus: %i" % unique_word_count)
        print("total items in dictionary (corpus words and deletions): %i" % len(self.dictionary))
        print("  edit distance for deletions: %i" % self.max_edit_distance)
        print("  length of longest word in corpus: %i" % self.longest_word_length)
        return self.dictionary

    def get_suggestions(self, string, silent=False):
        if (len(string) - self.longest_word_length) > self.max_edit_distance:
            if not silent:
                print("no items in dictionary within maximum edit distance")
            return []

        suggest_dict = {}
        min_suggest_len = float('inf')

        queue = [string]
        q_dictionary = {} 

        while len(queue) > 0:
            q_item = queue[0] 
            queue = queue[1:]

            if ((self.verbose < 2) and (len(suggest_dict) > 0) and
                    ((len(string) - len(q_item)) > min_suggest_len)):
                break

            if (q_item in self.dictionary) and (q_item not in suggest_dict):
                if self.dictionary[q_item][1] > 0:
                    assert len(string) >= len(q_item)
                    suggest_dict[q_item] = (self.dictionary[q_item][1],
                                            len(string) - len(q_item))

                    if (self.verbose < 2) and (len(string) == len(q_item)):
                        break
                    elif (len(string) - len(q_item)) < min_suggest_len:
                        min_suggest_len = len(string) - len(q_item)

                for sc_item in self.dictionary[q_item][0]:
                    if sc_item not in suggest_dict:
                        assert len(sc_item) > len(q_item)
                        assert len(q_item) <= len(string)
                        if len(q_item) == len(string):
                            assert q_item == string
                            item_dist = len(sc_item) - len(q_item)

                        assert sc_item != string
                        item_dist = dameraulevenshtein(sc_item, string)

                        if (self.verbose < 2) and (item_dist > min_suggest_len):
                            pass
                        elif item_dist <= self.max_edit_distance:
                            assert sc_item in self.dictionary  # should already be in dictionary if in suggestion list
                            suggest_dict[sc_item] = (self.dictionary[sc_item][1], item_dist)
                            if item_dist < min_suggest_len:
                                min_suggest_len = item_dist

                        if self.verbose < 2:
                            suggest_dict = {k: v for k, v in suggest_dict.items() if v[1] <= min_suggest_len}

            assert len(string) >= len(q_item)

            if (self.verbose < 2) and ((len(string) - len(q_item)) > min_suggest_len):
                pass
            elif (len(string) - len(q_item)) < self.max_edit_distance and len(q_item) > 1:
                for c in range(len(q_item)):  # character index
                    word_minus_c = q_item[:c] + q_item[c + 1:]
                    if word_minus_c not in q_dictionary:
                        queue.append(word_minus_c)
                        q_dictionary[word_minus_c] = None

        if not silent and self.verbose != 0:
            print("number of possible corrections: %i" % len(suggest_dict))
            print("  edit distance for deletions: %i" % self.max_edit_distance)

        as_list = suggest_dict.items()
        outlist = sorted(as_list, key=lambda x: (x[1][1], -x[1][0]))

        if self.verbose == 0:
            return outlist[0]
        else:
            return outlist

    def best_word(self, s, silent=False):
        try:
            return self.get_suggestions(s, silent)[0]
        except:
            return None


class ItemSelector(BaseEstimator, TransformerMixin):
    def __init__(self, field):
        self.field = field

    def fit(self, x, y=None):
        return self

    def transform(self, dataframe):
        dt = dataframe[self.field].dtype
        if is_categorical_dtype(dt):
            return dataframe[self.field].cat.codes[:, None]
        elif is_numeric_dtype(dt):
            return dataframe[self.field][:, None]
        else:
            return dataframe[self.field]


class DropColumnsByDf(BaseEstimator, TransformerMixin):
    def __init__(self, min_df=1, max_df=1.0):
        self.min_df = min_df
        self.max_df = max_df

    def fit(self, X, y=None):
        m = X.tocsc()
        self.nnz_cols = ((m != 0).sum(axis=0) >= self.min_df).A1
        if self.max_df < 1.0:
            max_df = m.shape[0] * self.max_df
            self.nnz_cols = self.nnz_cols & ((m != 0).sum(axis=0) <= max_df).A1
        return self

    def transform(self, X, y=None):
        m = X.tocsc()
        return m[:, self.nnz_cols]


def split_cat(text):
    try:
        cats = text.split("/")
        return cats[0], cats[1], cats[2], cats[0] + '/' + cats[1]
    except:
        print("no category")
        return 'other', 'other', 'other', 'other/other'


def brands_filling(dataset):
    vc = dataset['brand_name'].value_counts()
    brands = vc[vc > 0].index
    brand_word = r"[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+"

    many_w_brands = brands[brands.str.contains(' ')]
    one_w_brands = brands[~brands.str.contains(' ')]

    ss2 = SymSpell(max_edit_distance=0)
    ss2.create_dictionary_from_arr(many_w_brands, token_pattern=r'.+')

    ss1 = SymSpell(max_edit_distance=0)
    ss1.create_dictionary_from_arr(one_w_brands, token_pattern=r'.+')

    two_words_re = re.compile(r"(?=(\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+))")

    def find_in_str_ss2(row):
        for doc_word in two_words_re.finditer(row):
            print(doc_word)
            suggestion = ss2.best_word(doc_word.group(1), silent=True)
            if suggestion is not None:
                return doc_word.group(1)
        return ''

    def find_in_list_ss1(list):
        for doc_word in list:
            suggestion = ss1.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    def find_in_list_ss2(list):
        for doc_word in list:
            suggestion = ss2.best_word(doc_word, silent=True)
            if suggestion is not None:
                return doc_word
        return ''

    print(f"Before empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_name]

    n_desc = dataset[dataset['brand_name'] == '']['item_description'].str.findall(
        pat=r"^[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+\s[a-z0-9*/+\-'’?!.,|&%®™ôèéü]+")
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss2(row) for row in n_desc]

    n_name = dataset[dataset['brand_name'] == '']['name'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in n_name]

    desc_lower = dataset[dataset['brand_name'] == '']['item_description'].str.findall(pat=brand_word)
    dataset.loc[dataset['brand_name'] == '', 'brand_name'] = [find_in_list_ss1(row) for row in desc_lower]

    print(f"After empty brand_name: {len(dataset[dataset['brand_name'] == ''].index)}")

    del ss1, ss2
    gc.collect()


replace_dict1 = {'h&m': 'handm',
                 'at&t': 'atandt',
                 "don't": 'dont',
                 r'[:;][)]': 'funsmile',
                 r'[:;][(]': 'sadsmile',
                 'ii&iilovez': 'iiandiilovez'}

replace_dict2 = {'t shirt': 'tshirt',
                 'hatchimals': 'hatchimal',
                 '100% authentic': '100%authentic',
                 'lulu bags': 'lululemonbags',
                 'lululemon bags': 'lululemonbags',
                 'victoria secret': 'victoriasecret',
                 'iphone 6': 'iphone6',
                 'iphone 6s': 'iphone6s'}


def tokenize(text):
    text = text.lower()  # 1. Нижний регистр

    for old_token, new_token in replace_dict1.items():  # 2. Замена важных токенов, содержащих "плохие" символы
        text = re.sub(old_token, new_token, text)

    text = re.sub('[^A-Za-z0-9%$]+', ' ', text.lower().strip())  # 3. Удаление "плохих" символов

    karats_regex = r'(\d)([\s-]?)(karat|karats|carat|carats|kt)([^\w])'  # 4. Караты
    karats_repl = r'\1k\4'
    text = re.sub(karats_regex, karats_repl, text)

    unit_regex = r'(\d+)[\s-]([a-z]{2}|ddd|lbs|%|$)(\s)'  # 5. Всякие штуки типа 2.5 oz, 100 %
    unit_repl = r'\1\2\3'
    text = re.sub(unit_regex, unit_repl, text)

    for old_token, new_token in replace_dict2.items():  # 5. Замена токенов, не содержащих "плохих" символов
        text = re.sub(old_token, new_token, text)

    tokens = [w for w in text.split() if len(w) > 1 or not w.isalpha()]  # 6. Разделение на токены и удаление
    # однобуквенных токенов

    return ' '.join(tokens)


def tokenize_naive(text):
    return text


def mapper(s, func, **kwargs):
    return s.apply(func, **kwargs)


def parallelize(data, func, n_jobs=4, **kwargs):
    data_split = np.array_split(data, n_jobs)
    f = partial(mapper, func=func, **kwargs)
    p = mp.Pool()
    feat = pd.concat(p.map(f, data_split))
    p.close()
    p.join()
    return feat


def intersect_drop_columns(X_train, X_test, min_df=1):
    nnz_train = ((X_train != 0).sum(axis=0) >= min_df).A1
    nnz_test = ((X_test != 0).sum(axis=0) >= min_df).A1
    nnz_cols = nnz_train & nnz_test
    return X_train[:, nnz_cols], X_test[:, nnz_cols]
    
#########################################

def make_common_data(X_train, X_test):
    """
    X_train, X_test - исходные данные
    """
    X_train = X_train[X_train['price'] >= 3].reset_index(drop=True)
    y = np.log1p(X_train['price'])
    del X_train['price']
    X = pd.concat([X_train, X_test])

    X['is_category'] = (X['category_name'].notnull() * 1).astype('category')

    X['category_name'] = X['category_name'] \
        .fillna('other/other/other') \
        .str.lower()

    X['main_cat'], X['subcat_1'], X['subcat_2'], X['maincat_subcat1'] = \
        zip(*X['category_name'].apply(lambda x: split_cat(x)))

    X['is_brand'] = (X['brand_name'].notnull() * 1).astype('category')

    X['maincat_cond'] = X['main_cat'].map(str) + '_' + X['item_condition_id'].astype(str)
    X['subcat_1_cond'] = X['subcat_1'].map(str) + '_' + X['item_condition_id'].astype(str)
    X['subcat_2_cond'] = X['subcat_2'].map(str) + '_' + X['item_condition_id'].astype(str)

    X['name'] = X['name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    X['brand_name'] = X['brand_name'] \
        .fillna('') \
        .str.lower() \
        .astype(str)
    X['item_description'] = X['item_description'] \
        .fillna('') \
        .str.lower() \
        .replace('No description yet', '')

    for col in ['name', 'item_description', 'brand_name']:
        print(col)
        X[col] = parallelize(X[col], tokenize)

    brands_filling(X)

    X['name_expanded'] = X['name'] + ' ' + X['brand_name']
    X['item_description_expanded'] = X['item_description'] \
                            + ' ' + X['name'] \
                            + ' ' + X['brand_name'] \
                            + ' ' + X['main_cat'] \
                            + ' ' + X['subcat_1'] \
                            + ' ' + X['subcat_2'] \

    return X, y


def make_data_for_fmftrl(X):
    data_list = []
    cols = ['brand_name', 'item_condition_id', 'shipping', 'main_cat',
            'subcat_1', 'subcat_2', 'maincat_subcat1',
            'maincat_cond', 'subcat_1_cond', 'subcat_2_cond',
            'is_brand', 'is_category']
    for col in cols:
        print(col)
        lb = LabelBinarizer(sparse_output=True)
        data_list.append(lb.fit_transform(X[col]))
    
    wb = wordbatch.WordBatch(tokenize_naive, extractor=(WordBag, {'hash_ngrams': 2,
                                                              'hash_ngrams_weights': [2.0, 1.0],
                                                              'hash_size': 2 ** 22,
                                                              'norm': None,
                                                              'tf': 'binary',
                                                              'idf': None}))
    wb.dictionary_freeze = True
    X_name = wb.fit_transform(X['name'])

    wb = wordbatch.WordBatch(tokenize_naive, extractor=(WordBag, {'hash_ngrams': 2,
                                                                  'hash_ngrams_weights': [1.0, 1.0],
                                                                  'hash_size': 2 ** 22,
                                                                  'norm': 'l2',
                                                                  'tf': 1.0,
                                                                  'idf': None}))
    wb.dictionary_freeze = True
    X_description = wb.fit_transform(X['item_description'])
    
    data_list += [X_name, X_description]
    
    X_sparse = hstack(data_list).tocsr()
    X_train, X_test = intersect_drop_columns(X_sparse[:n_train], X_sparse[n_train:])
    
    return X_train, X_test

def make_fmftrl_predictions(X_train, X_test, y):
    model = FM_FTRL(alpha=0.01,
                    beta=0.01,
                    L1=0.00001,
                    L2=0.1,
                    D=X_train.shape[1],
                    alpha_fm=0.01,
                    L2_fm=0.0,
                    init_fm=0.01,
                    D_fm=200,
                    e_noise=0.0001,
                    iters=17,
                    inv_link='identity',
                    threads=4)
    
    model.fit(X_train, y, verbose=1)
    y_pred = model.predict(X_test)
    return y_pred
    

def make_data_for_ridge(X):
    stopwords = frozenset(['the', 'a', 'an', 'is', 'it', 'this'])

    vectorizer = FeatureUnion([
        ('name_expanded', Pipeline([
            ('select', ItemSelector('name_expanded')),
            ('transform', HashingVectorizer(
                ngram_range=(1, 2),
                n_features=2 ** 27,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('category_name', Pipeline([
            ('select', ItemSelector('category_name')),
            ('transform', HashingVectorizer(
                ngram_range=(1, 1),
                token_pattern='.+',
                tokenizer=split_cat,
                n_features=2 ** 27,
                norm='l2',
                lowercase=False
            )),
            ('drop_cols', DropColumnsByDf(min_df=2))
        ])),
        ('brand_name', Pipeline([
            ('select', ItemSelector('brand_name')),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('maincat_cond', Pipeline([
            ('select', ItemSelector('maincat_cond')),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_1_cond', Pipeline([
            ('select', ItemSelector('subcat_1_cond')),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('subcat_2_cond', Pipeline([
            ('select', ItemSelector('subcat_2_cond')),
            ('transform', CountVectorizer(
                token_pattern='.+',
                min_df=2,
                lowercase=False
            )),
        ])),
        ('is_brand', Pipeline([
            ('select', ItemSelector('is_brand')),
            ('ohe', OneHotEncoder())
        ])),
        ('is_category', Pipeline([
            ('select', ItemSelector('is_category')),
            ('ohe', OneHotEncoder())
        ])),
        ('shipping', Pipeline([
            ('select', ItemSelector('shipping')),
            ('ohe', OneHotEncoder())
        ])),
        ('item_condition_id', Pipeline([
            ('select', ItemSelector('item_condition_id')),
            ('ohe', OneHotEncoder())
        ])),
        ('item_description_expanded', Pipeline([
            ('select', ItemSelector('item_description_expanded')),
            ('hash', HashingVectorizer(
                ngram_range=(1, 3),
                n_features=2 ** 27,
                dtype=np.float32,
                norm='l2',
                lowercase=False,
                stop_words=stopwords
            )),
            ('drop_cols', DropColumnsByDf(min_df=2)),
        ]))
    ], n_jobs=1)
    
    print('making sparse data')
    X_sparse = vectorizer.fit_transform(X)
    
    print('tf-idf transfrm')
    tfidf_transformer = TfidfTransformer()
    X_sparse = tfidf_transformer.fit_transform(X_sparse)

    X_train, X_test = intersect_drop_columns(X_sparse[:n_train], X_sparse[n_train:])
    return X_train, X_test
    

def make_ridge_predictions(X_train, X_test, y):
    model = Ridge(solver='auto',
                  fit_intercept=True,
                  alpha=0.4,
                  max_iter=200,
                  normalize=False,
                  tol=0.01)
                  
    model.fit(X_train, y)
    y_pred = model.predict(X_test)
    return y_pred
    

if __name__ == '__main__':
    mp.set_start_method('forkserver', True)
    
    print('loading data...')
    X_train = pd.read_table(os.path.join(INPUT_PATH, 'train.tsv'),
                            engine='c',
                            dtype={'item_condition_id': 'category',
                                    'shipping': 'category'})
    X_test = pd.read_table(os.path.join(INPUT_PATH, 'test.tsv'),
                            engine='c',
                            dtype={'item_condition_id': 'category',
                                  'shipping': 'category'})
    
    # N1, N2 = 500000, 10000
    # X_train, X_test = X_train.loc[:N1], X_test.loc[:N2]
    n_train = (X_train['price'] >= 3).sum()
    submission = X_test[['test_id']]
    del X_test['test_id'], X_train['train_id']
    gc.collect()
    
    print('making common data...')
    X, y = make_common_data(X_train, X_test)
    
    print('making data for ridge...')
    X_train, X_test = make_data_for_ridge(X)
    
    print('fitting ridge...')
    y_pred_ridge = make_ridge_predictions(X_train, X_test, y)
    
    print('making data for fmftrl...')
    X_train, X_test = make_data_for_fmftrl(X)
    
    print('fitting fmftrl...')
    y_pred_fmftrl = make_fmftrl_predictions(X_train, X_test, y)
    
    print('making submission...')
    submission.loc[:, 'price'] = np.expm1(0.5*y_pred_fmftrl + 0.5*y_pred_ridge)
    submission.loc[submission['price'] < 0, 'price'] = 0
    submission.to_csv('submission.csv', index=False)