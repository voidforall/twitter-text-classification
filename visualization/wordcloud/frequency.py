# adapted from wordcloud website, example: frequenc.py
import multidict as multidict

import numpy as np

import os
import re
from PIL import Image
from os import path
from wordcloud import WordCloud
import matplotlib.pyplot as plt

from wordcloud import STOPWORDS

# getFrequencyDictForText:
# generate frequency dictionary for visualization
def getFrequencyDictForText(sentence):
    fullTermsDict = multidict.MultiDict()
    tmpDict = {}

    # making dict for counting frequencies
    for text in sentence.split(" "):
        if re.match("a|the|an|the|to|in|for|of|or|by|with|is|on|that|be", text):
            continue
        if text in STOPWORDS:
            continue
        # reconstruct faces
        if text == "<smile>":
            text = ":)"
        elif text == "<lolface>":
            text = ":p"
        elif text == "<sadface>":
            text = ":("
        elif text == "<neutralface>":
            text = ":|"
        elif re.match("<[^>]*>", text): # check if it is other <...> than faces
            continue
        elif re.match("[^\w]+", text): # check if it has not alphabet, like "!""
            continue
        elif "<user>" in text:
            continue
        val = tmpDict.get(text, 0)
        tmpDict[text.lower()] = val + 1
    for key in tmpDict:
        fullTermsDict.add(key, tmpDict[key])
    return fullTermsDict

# makeImage:
# to make an image
# use down parameter to get disapprove
def makeImage(text, mask_path="upvote.png", down=False):
    mask = Image.open(mask_path)
    mask = mask.resize((900, int(900 / mask.size[0] * mask.size[1])))
    mask = np.array(mask)

    if down == True:
        mask = mask[::-1,:,:]

    wc = WordCloud(background_color="white", max_words=1000, mask=mask)
    # generate word cloud
    wc.generate_from_frequencies(text)

    # show
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    # positive
    # text = open(path.join(d, '../deep-pipeline/data/train_pos_full.txt'), encoding='utf-8')
    # text = text.read()
    # makeImage(getFrequencyDictForText(text))

    text = open('../../deep-pipeline/data/train_neg_full.txt', encoding='utf-8')
    text = text.read()
    makeImage(getFrequencyDictForText(text), down=True)
