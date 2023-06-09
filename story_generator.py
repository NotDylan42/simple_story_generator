import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# Read the stories from a text file
def read_stories_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    stories = text.split('\n\n')  # Split stories based on '\n\n'
    return stories

# Preprocessing the stories
def preprocess_stories(stories):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(stories)
    total_words = len(tokenizer.word_index) + 1

    act_sequences = []
    for story in stories:
        acts = story.strip().split('---')  # Split story into acts based on '---'
        act_sequence = ' '.join(acts)  # Combine all acts into a single sequence
        act_sequences.append(act_sequence)

    input_sequences = tokenizer.texts_to_sequences(act_sequences)
    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    predictors, labels = input_sequences[:, :-1], input_sequences[:, -1]
    labels = to_categorical(labels, num_classes=total_words)

    return predictors, labels, tokenizer, max_sequence_len, total_words


# Build the model
def build_model(max_sequence_len, total_words):
    model = Sequential()
    model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
    model.add(LSTM(150))
    model.add(Dense(total_words, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

# Train the model
def train_model(model, predictors, labels):
    model.fit(predictors, labels, epochs=100, verbose=1)

# Generate stories based on the 3-act structure
def generate_story(model, tokenizer, max_sequence_len):
    generated_story = ''
    acts = ['Act 1', 'Act 2', 'Act 3']

    for act in acts:
        act_text = act + ':\n'
        generated_story += act_text

        input_sequence = tokenizer.texts_to_sequences([act_text])[0]

        for _ in range(100):  # Generate 100 words or characters for each act
            encoded = pad_sequences([input_sequence], maxlen=max_sequence_len-1, padding='pre')
            predicted_word_index = np.argmax(model.predict(encoded), axis=-1)
            predicted_word = tokenizer.index_word.get(predicted_word_index[0], '')

            generated_story += predicted_word + ' '
            input_sequence.append(predicted_word_index[0])
            input_sequence = input_sequence[1:]  # Remove the first word from the input sequence

        generated_story += '\n\n'

    return generated_story

# File path of the text file containing the stories
file_path = 'stories.txt'

# Read the stories from the file
stories = read_stories_from_file(file_path)

# Preprocess the stories
predictors, labels, tokenizer, max_sequence_len, total_words = preprocess_stories(stories)

# Build the model
model = build_model(max_sequence_len, total_words)

# Train the model
train_model(model, predictors, labels)

# Generate a story based on the 3-act structure
generated_story = generate_story(model, tokenizer, max_sequence_len)
