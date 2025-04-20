import string
import math

# Example input text
text = input("Enter a text: ")

# List of stopwords
stopwords = ['i', 'me', 'my', 'you', 'he', 'she', 'it', 'we', 'they', 'a', 'an', 'the', 'and', 'or', 'but', 'so', 'is', 'are', 'was', 'were']

# Simple Stemmer
def stem(word): return word[:-3] if word.endswith("ing") else word

# Simple Lemmatizer
def lemmatize(word): return word.rstrip('s') if word.endswith('s') else word

# Remove punctuations
def remove_punctuation(text): return text.translate(str.maketrans('', '', string.punctuation))

# Process the text: Remove stopwords, lemmatization, and stemming
words = [lemmatize(stem(word.strip(string.punctuation).lower())) for word in remove_punctuation(text).split() if word.lower() not in stopwords]

# 1. Print processed words after stopwords, lemmatization, and stemming
print("Processed Words:", words)

# 5. Bag of Words (BoW) Representation: Count word occurrences
bow = {word: words.count(word) for word in set(words)}

# 2. Print Bag of Words representation
print("Bag of Words (BoW):", bow)

# TF-IDF calculation (simplified version)
def compute_tfidf(bow, words):
    tfidf = {word: (bow[word] / len(words)) * math.log(len(words) / (1 + bow.get(word, 0))) for word in bow}
    return tfidf

# 6. Print TF-IDF
tfidf = compute_tfidf(bow, words)
print("TF-IDF:", tfidf)

# 4. Print text after removing punctuation
print("Text without Punctuation:", remove_punctuation(text))

# 3. Print stopwords removed
stopwords_removed = [word for word in text.split() if word.lower() not in stopwords]
print("Text without Stopwords:", stopwords_removed)
