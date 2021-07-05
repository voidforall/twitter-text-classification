import argparse
import re
import string

def duplicate_clean(dataset):
    """to clean the duplicate samples"""
    from collections import OrderedDict
    return list(OrderedDict.fromkeys(dataset))

class Tokenize(object):
        
	"""
    Tweet preporcessor for glove.twitter.27B
	
		:param lowercase: Transform tweets to lowercase?, defaults to True
		:type lowercase: Bool
		
    Ref: https://github.com/AntiDeprime/GloVeTwitterPreprocPy/blob/master/GloVePreprocessor.py
	"""
	
	def __init__(self, lowercase = True):   
		self.lowercase = lowercase
	
	def __hashtags__ (self, hashtag):    
		hashtag_body = hashtag.group()[1:]
		if( hashtag_body.upper == hashtag_body):
			result = f' <hashtag> {hashtag_body} <allcaps> '
		else:
			result = ' <hashtag> ' + ' '.join(hashtag_body.split('/(?=[A-Z])/)'))
		return (result)       

	def preprocess (self, tweet):
		"""
        Preprocessor function
		
		:param tweet: Tweet string
		:type tweet: str
		:returns: Preprocessed Tweet 
		:rtype: str
		
		"""
		self.tweet = tweet

		# Different regex parts for smiley faces
		eyes = '[8:=;]'
		nose = "['`\-]?"
		
		# Mark allcaps words
		self.tweet = re.sub(r'\b[A-Z][A-Z0-9]+\b', 
					lambda x: f'{x.group().lower()} <allcaps> ', 
					self.tweet)
		
		# Mark urls
		self.tweet = re.sub('https?:\/\/\S+\b|www\.(\w+\.)+\S*', " <url> ", self.tweet)
		# Force splitting words appended with slashes (once we tokenized the URLs, of course)
		self.tweet = re.sub('/',' / ', self.tweet) 
		# Mark @users
		self.tweet = re.sub('@\w+', ' <user> ', self.tweet)

		# Mark smileys
		self.tweet = re.sub(f'{eyes}\s*{nose}\s*[)dD]+|[)dD]+\s*{nose}\s*{eyes}', ' <smile> ', self.tweet)
		self.tweet = re.sub(f'{eyes}\s*{nose}\s*[pP]+', '<lolface>', self.tweet)
		self.tweet = re.sub(f'{eyes}\s*{nose}\s*\(+|\)+\s*{nose}\s*{eyes}', ' <sadface> ', self.tweet)
		self.tweet = re.sub(f'{eyes}\s*{nose}\s*[\/|l*]', ' <neutralface> ', self.tweet)
		self.tweet = re.sub('<\s*3',' <heart> ', self.tweet)
		self.tweet = re.sub(r'\blol\b', ' <lolface> ', self.tweet)
		
		# Mark numbers 
		self.tweet = re.sub(r'[-+]?[.\d]*[\d]+[:,.\d]*', ' <number> ', self.tweet)

		# Mark hashtags 
		self.tweet = re.sub('#\S+', self.__hashtags__, self.tweet)

		# Mark punctuation repetitions (eg. "!!!" => "! <REPEAT>")
		self.tweet = re.sub('([!?.]\s*){2,}', 
					lambda x: f'{x.group()[0]} <repeat> ', 
					self.tweet)

		# Mark elongated words like heyyy -> hey <elong>
		self.tweet = re.sub(r'\b(\S*?)(.)\2{2,}\b', lambda x: f'{x.group(1)}{x.group(2)} <elong> ', self.tweet)
		
		# To lowercase 
		if self.lowercase:
			self.tweet = self.tweet.lower()
		
		# Trim whitespaces 
		self.tweet = ' '.join(self.tweet.split())+'\n'

		return (self.tweet)

def tokenizer():
    # test the tokenizer
    word_list = [":)",
                ";-("
                "HELLO",
                ":-(",
                "www.google.com",
                "123",
                "!!?",
                "moon",
                "moonnnn"]

    token = Tokenize()
    for word in word_list:
        print(word, token.preprocess(word))


if __name__ == '__main__':
    # arguments
    parser = argparse.ArgumentParser(description="data preprocess: remove the duplicate and tokenize special expressions")
    parser.add_argument('--dataset', dest = "data",type=str,default=None, help="process which dataset: test/subset/fullset", required=True)
    args = parser.parse_args()

    # tokenize special expressions for submission data
    if args.data == 'test':
        # load the data 
        data_path = r"../data/test_data.txt"
        with open(data_path, "r", encoding="utf-8") as f:
            data = f.readlines()

        # start the tokenizer
        cleaned_data = []
        token = Tokenize()
        for rec_i in data:
            for idx, char_i in enumerate(rec_i):
                if char_i == ',':
                    break
            cleaned_data.append(token.preprocess(rec_i[idx+1:]))
        # save the preprocessed data
        clean_path = r"../data/test_data_clean.txt"
        with open(clean_path, "w", encoding="utf-8") as data_file_pos:
            data_file_pos.writelines(cleaned_data)

    # remove the duplicates and tokenize special expression for training sets
    else:
        if args.data == 'subset':
            pos_path = r"../data/train_pos.txt"
            neg_path = r"../data/train_neg.txt"
            pos_data_path = r"../data/train_pos_clean.txt"
            neg_data_path = r"../data/train_neg_clean.txt"
        elif args.data == 'fullset':
            pos_path = r"../data/train_pos_full.txt"
            neg_path = r"../data/train_neg_full.txt"
            pos_data_path = r"../data/train_pos_full_clean.txt"
            neg_data_path = r"../data/train_neg_full_clean.txt"
        else:
            print('no such dataset')

        # load the data 
        with open(pos_path, "r", encoding="utf-8") as data_pos:
            pos_X = data_pos.readlines()
        with open(neg_path, "r", encoding="utf-8") as data_neg:
            neg_X = data_neg.readlines()

        # remove the duplicate samples (sequence randomly shuffled)
        pos_X_clean = duplicate_clean(pos_X)
        neg_X_clean = duplicate_clean(neg_X)

        print("(pos) file length before duplicate removal: %d; after: %d"%(len(pos_X), len(pos_X_clean)))
        print("(neg) file length before duplicate removal: %d; after: %d"%(len(neg_X), len(neg_X_clean)))

        # Tokenize
        token = Tokenize()
        pos_X_tokenize = [token.preprocess(senc) for senc in pos_X_clean]
        neg_X_tokenize = [token.preprocess(senc) for senc in neg_X_clean]

        # save the preprocessed data
        with open(pos_data_path, "w", encoding="utf-8") as data_file_pos:
            data_file_pos.writelines(pos_X_tokenize)
        with open(neg_data_path, "w", encoding="utf-8") as data_file_neg:
            data_file_neg.writelines(neg_X_tokenize)

    print("End of the data preprocessing")
