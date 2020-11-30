from tarfile import ENCODING
import torch
from torch import embedding
import torch.nn as nn
import torch.optim as optim
from torchtext.data.iterator import batch
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import numpy as np
import spacy
import random
from torch.utils.tensorboard import SummaryWriter  # to print to tensorboard
from .utils import translate_sentence, bleu, save_checkpoint, load_checkpoint

import spacy.cli

"""
Youtube:
1- ) Machine Translation: https://www.youtube.com/watch?v=EoGUlvhRYpk

"""

"""
Firstly :
python -m spacy download en
python -m spacy download de
"""
spacy_eng = spacy.load("en_core_web_sm")  # Veri yükle
spacy_ger = spacy.load("de_core_news_sm")  # Veri yükle

device = torch.device("cuda:0" if torch.cuda.is_available()
                      else "cpu")  # gpu varsa gpuyu kullan


# german tokenize yapan func.
def tokenizer_ger(text):
    return [tok.text for tok in spacy_ger.tokenizer(text)]


# englis tokenize yapan func.
def tokenizer_eng(text):
    return [tok.text for tok in spacy_eng.tokenizer(text)]


german = Field(tokenize=tokenizer_ger, lower=True,
               init_token="<sos>", eos_token="<eos>")
english = Field(tokenize=tokenizer_eng, lower=True,
                init_token="<sos>", eos_token="<eos>")


# eğitim, validation, test veri setleri ayarla
train_data, validation_data, test_data = Multi30k.splits(
    exts=('.de', '.en'), fields=(german, english))


# kelime listesi oluştur
german.build_vocab(train_data, max_size=1000, min_freq=2)
english.build_vocab(train_data, max_size=1000, min_freq=2)


# Encoder modeli oluştur
class Encoder(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, num_layers, dropout):
        """
        Encoder modelini oluşturmak için şu parametrelere ihtiyaç var:
        input_size: Giriş verisinin boyutu;
        embedding_size , 
        hidden_size, 
        num_layers, 
        dropout
        """
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dropout = nn.Dropout(p=dropout)
        self.emdedding = nn.Embedding(input_size, embedding_dim=embedding_size)
        self.rnn = nn.LSTM(embedding_size, hidden_size,
                           num_layers, dropout=dropout, bidirectional=True)

        self.fc_hidden = nn.Linear(hidden_size*2, hidden_size)
        self.fc_cell = nn.Linear(hidden_size*2, hidden_size)

    def forward(self, x):
        embedding = self.dropout(self.emdedding(x))
        # sadece hidden, cell geri döndür çünkü sadece bunları kullanıyoruz.
        encoding_states, (hidden, cell) = self.rnn(embedding)

        hidden = self.fc_hidden(torch.cat(hidden[0:1], hidden[1:2]), dim=2)
        cell = self.fc_cell(torch.cat(hidden[0:1], hidden[1:2]), dim=2)
        return encoding_states, hidden, cell


class Decoder(nn.Module):

    def __init__(self, input_size, embedding_size, hidden_size, output_size, num_layers, p):
        """
        Decoder modelini oluşturmak için şu parametrelere ihtiyaç var:
        input_size: Giriş verisinin boyutu;
        embedding_size , 
        hidden_size
        output_size, 
        num_layers, 
        dropout=p
        """

        super(Decoder, self).__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_size, embedding_size)
        self.rnn = nn.LSTM(hidden_size*2 + embedding_size,
                           hidden_size, num_layers, dropout=p)
        self.energy = nn.Linear(hidden_size*3, 1)
        self.sofmax = nn.Softmax(dim=0)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, encoder_states, hidden, cell):
        """
        forward
        """

        x = x.unsqueeze(0)
        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        h_reshaped = hidden.repeat(sequence_length, 1, 1)

        energy = self.relu(self.energy(
            torch.cat(h_reshaped, encoder_states), dim=2))

        attention = self.sofmax(energy)
        attention = attention.permute(1, 0, 2)
        encoder_states = encoder_states.permute(1, 0, 2)

        context_vector = torch.bmm(attention, encoder_states).permute(1, 0, 2)

        rnn_input = torch.cat((context_vector, embedding), dim=2)

        outputs, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        predictions = self.fc(outputs)
        predictions = predictions.squeeze(0)
        return predictions, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        """
        docstring
        """
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        """
        docstring
        """

        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(english.vocab)
        outputs = torch.zeros(target_len, batch_size,
                              target_vocab_size).to(device=device)
        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]

        for t in range(1, target_len):
            output, hidden, cell = self.decoder(
                x, encoder_states, hidden, cell)
            outputs[t] = output
            best_guess = output.argmax(1)
            x = target[t] if random.random(
            ) < teacher_force_ratio else best_guess

        return outputs


# training params
num_epochs = 5
learning_rate = 0.001
batch_size = 64

# model hparams
load_model = False
input_size_encoder = len(german.vocab)
input_size_decoder = len(english.vocab)

output_size = len(english.vocab)
encoder_embedding_size = 300
decoder_embedding_size = 300

hidden_size = 1024
num_layers = 1
enc_dropout = 0.5
dec_dropout = 0.5

writer = SummaryWriter(f"runs/loss_plot")
step = 0


train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, validation_data, test_data),
    batch_size=batch_size,
    sort_key=lambda x: len(x.src)
)

encoder_net = Encoder(input_size=input_size_encoder, embedding_size=encoder_embedding_size,
                      hidden_size=hidden_size, num_layers=num_layers, dropout=enc_dropout).to(device=device)


decoder_net = Decoder(input_size_decoder, decoder_embedding_size,
                      hidden_size, output_size, num_layers, dec_dropout).to(device=device)


model = Seq2Seq(encoder=encoder_net, decoder=decoder_net).to(device=device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

pad_idx = english.vocab.stoi["<pad>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

if load_model:
    load_checkpoint(torch.load("my_checkpoint.pth.tar"), model, optimizer)


sentence = "ein boot mit mehreren männern darauf wird von einem großen pferdegespann ans ufer gezogen."

for epoch in range(num_epochs):
    print(f"[Epoch {epoch} / {num_epochs}]")

    checkpoint = {"state_dict": model.state_dict(
    ), "optimizer": optimizer.state_dict()}
    save_checkpoint(checkpoint)

    model.eval()

    translated_sentence = translate_sentence(
        model, sentence, german, english, device, max_length=50
    )

    print(f"Translated example sentence: \n {translated_sentence}")

    model.train()

    for batch_idx, batch in enumerate(train_iterator):
        # Get input and targets and get to cuda
        inp_data = batch.src.to(device)
        target = batch.trg.to(device)

        # Forward prop
        output = model(inp_data, target)

        # Output is of shape (trg_len, batch_size, output_dim) but Cross Entropy Loss
        # doesn't take input in that form. For example if we have MNIST we want to have
        # output to be: (N, 10) and targets just (N). Here we can view it in a similar
        # way that we have output_words * batch_size that we want to send in into
        # our cost function, so we need to do some reshapin. While we're at it
        # Let's also remove the start token while we're at it
        output = output[1:].reshape(-1, output.shape[2])
        target = target[1:].reshape(-1)

        optimizer.zero_grad()
        loss = criterion(output, target)

        # Back prop
        loss.backward()

        # Clip to avoid exploding gradient issues, makes sure grads are
        # within a healthy range
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

        # Gradient descent step
        optimizer.step()

        # Plot to tensorboard
        writer.add_scalar("Training loss", loss, global_step=step)
        step += 1


score = bleu(test_data[1:100], model, german, english, device)
print(f"Bleu score {score*100:.2f}")
