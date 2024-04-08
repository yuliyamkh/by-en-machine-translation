from preprocessing import input_features_dict, target_features_dict, reverse_target_features_dict, max_decoder_seq_length, max_encoder_seq_length, input_docs
from training_model import encoder_input_data, num_decoder_tokens, num_encoder_tokens, latent_dim

from keras.layers import Input, LSTM, Dense
from keras.models import Model, load_model
import numpy as np
import re

training_model = load_model('training_model_3.h5')
print(list(training_model.layers))
print(list(training_model.input))

encoder_inputs = training_model.input[0]
encoder_outputs, state_h_enc, state_c_enc = training_model.layers[2].output
encoder_states = [state_h_enc, state_c_enc]

encoder_model = Model(encoder_inputs, encoder_states)

decoder_inputs = training_model.input[1]
decoder_state_input_hidden = Input(shape=(latent_dim,))
decoder_state_input_cell = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_hidden, decoder_state_input_cell]
decoder_lstm = training_model.layers[3]
decoder_outputs, state_hidden, state_cell = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_hidden, state_cell]
decoder_dense = training_model.layers[4]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)


def decode_sequence(test_input):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(test_input)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first token of target sequence with the start token.
    target_seq[0, 0, target_features_dict['<START>']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    decoded_sentence = ''

    stop_condition = False
    while not stop_condition:
        # Run the decoder model to get possible
        # output tokens (with probabilities) & states
        output_tokens, hidden_state, cell_state = decoder_model.predict(
            [target_seq] + states_value)

        # Choose token with highest probability
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_token = reverse_target_features_dict[sampled_token_index]
        decoded_sentence += " " + sampled_token

        # Exit condition: either hit max length
        # or find stop token.
        if (sampled_token == '<END>' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [hidden_state, cell_state]

    return decoded_sentence


# TEST ONE SEQUENCE
input_sentence = "got it?"
test_sentence_tokenized = np.zeros((1, max_encoder_seq_length, num_encoder_tokens), dtype='float32')
for t, token in enumerate(re.findall(r"[\w']+|[^\s\w]", input_sentence)):
    test_sentence_tokenized[0, t, input_features_dict[token]] = 1.
print(input_sentence)
print(decode_sequence(test_sentence_tokenized))
exit()

# CHANGE RANGE (NUMBER OF TEST SENTENCES TO TRANSLATE) AS YOU PLEASE
for seq_index in range(10):
    test_input = encoder_input_data[seq_index:seq_index+1]
    # print("Test input: ", test_input)
    decoded_sentence = decode_sequence(test_input)
    print('-')
    print('Input sentence:', input_docs[seq_index])
    print('Decoded sentence:', decoded_sentence)