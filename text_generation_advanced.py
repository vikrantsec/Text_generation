import tensorflow as tf
import numpy as np
import os
import time
import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

tf.enable_eager_execution()
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt', cache_subdir='/root/avinash/python_projects/Machine Learning/tensor_flow/text_generation')

text = open('./shakespeare.txt', 'rb').read().decode(encoding='utf-8')

print("Length of text: {} characters".format(len(text)))

print(text[:250])

vocab = sorted(set(text))

print('{} unique characters'.format(len(vocab)))

char2idx = {u:i for i,u in enumerate(vocab)}
idx2char = np.array(vocab)
print(char2idx)

text_as_int = np.array([char2idx[c] for c in text])

seq_length = 100
examples_per_epoch = len(text)//(seq_length+1)

char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

for i in char_dataset.take(5):
    print(idx2char[i.numpy()], end='')

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

dataset = sequences.map(split_input_target)

batch_size = 64
buffer_size = 10000

dataset = dataset.shuffle(buffer_size).batch(batch_size, drop_remainder=True)

vocab_size = len(vocab)
embedding_dim = 256
rnn_units = 1024

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
        tf.keras.layers.GRU(rnn_units, return_sequences=True, stateful=True, recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

model = build_model(vocab_size=vocab_size, embedding_dim=embedding_dim, rnn_units=rnn_units, batch_size=batch_size)

optimizer = tf.keras.optimizers.Adam()

@tf.function
def train_step(input, target):
    with tf.GradientTape() as tape:
        predictions = model(input)
        loss = tf.reduce_mean(tf.keras.losses.sparse_categorical_crossentropy(target, predictions, from_logits=True))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss

checkpoint_dir = './training_checkpoints_advanced'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix, save_weights_only=True)

epochs = 10

for epoch in range(epochs):
    start = time.time()
    hidden = model.reset_states()

    for (batch_n, (inp, target)) in enumerate(dataset):
        loss = train_step(inp, target)

        if batch_n%100 == 0:
            template = "Epoch {} Batch {} Loss {}"
            print(template.format(epoch+1, batch_n, loss))
    
    if (epoch+1)%5 == 0:
        model.save_weights(checkpoint_prefix.format(epoch=epoch))

    print("Epoch {} Loss {:.4f}".format(epoch+1, loss))
    print('Time taken for {} epoch is {} sec\n'.format(epoch+1, time.time() - start))

model.save_weights(checkpoint_prefix.format(epoch=epoch))

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text(model, start_string):
    num_generate = 1000

    input_eval = [char2idx[s] for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)

    text_generated = []

    temperature = 1.0

    model.reset_states()

    for i in range(num_generate):
        predictions = model(input_eval)

        predictions = tf.squeeze(predictions, 0)

        predictions = predictions/temperature

        predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

        input_eval = tf.expand_dims([predicted_id], 0)

        text_generated.append(idx2char[predicted_id])

    return (start_string + ''.join(text_generated))

print(generate_text(model, start_string=u"JULIET: "))