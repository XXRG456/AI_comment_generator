from project import model_config
from project import tokenize, join_tokens
from project import build_model_1, build_model_2
import numpy as np
import tensorflow as tf


idx_to_word = model_config['idx_to_word']
max_sequence_length = model_config['max_sequence_length']

model = build_model_1()

def predict(seed_word: str) -> str:
    
    prediction_tokens = [seed_word]
    for i in range(max_sequence_length):
        tokenised = tokenize(seed_word)
        tokenised_padded = tf.keras.utils.pad_sequences([tokenised], maxlen = max_sequence_length - 1, padding = 'pre')
        predicted = model.predict(tokenised_padded, verbose = 0)
        predicted = np.argmax(predicted, axis=-1)[0]
        word = idx_to_word[str(predicted)]
        if i == 0 and word == '<EOS>':
            print("EOS predicted!")
            return seed_word
            
        if word == '<EOS>': 
            print("EOS predicted!")
            break
        seed_word += f" {word}"
       
        prediction_tokens.append(word)
       
        sentence = join_tokens(prediction_tokens)
    return sentence


if __name__ == '__main__':
    
    seed_word = 'great'
    print(predict(seed_word=seed_word))
    
