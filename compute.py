
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import json
import re
import pickle
import random
from scipy.special import expit

compute_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_data():
    data_path = 'MLDS_hw2_1_data'
    with open(data_path + 'training_label.json', 'r') as file:
        captions_data = json.load(file)

    vocab = {}
    for entry in captions_data:
        for caption in entry['caption']:
            words = re.sub('[.!,;?]]', ' ', caption).split()
            for word in words:
                word = word.replace('.', '') if '.' in word else word
                vocab[word] = vocab.get(word, 0) + 1

    vocab_filtered = {word: count for word, count in vocab.items() if count > 4}
    index_to_word = {index + 4: word for index, word in enumerate(vocab_filtered)}
    word_to_index = {word: index + 4 for index, word in enumerate(vocab_filtered)}
    tokens = [('<PAD>', 0), ('<SOS>', 1), ('<EOS>', 2), ('<UNK>', 3)]
    for token, idx in tokens:
        index_to_word[idx] = token
        word_to_index[token] = idx

    return index_to_word, word_to_index, vocab_filtered

def split_sentence(sentence, vocab, word_to_index):
    sentence = re.sub(r'[.!,;?]', ' ', sentence).split()
    sentence = [word_to_index.get(word, 3) for word in sentence]
    sentence = [1] + sentence + [2]
    return sentence

def tag_captions(label_json, vocab, word_to_index):
    path = 'MLDS_hw2_1_data' + label_json
    tagged_data = []
    with open(path, 'r') as file:
        data = json.load(file)
    for item in data:
        for caption in item['caption']:
            tagged_data.append((item['id'], split_sentence(caption, vocab, word_to_index)))
    return tagged_data

def load_features(directory):
    features_path = 'MLDS_hw2_1_data' + directory
    features = {}
    for file_name in os.listdir(features_path):
        features[file_name.split('.npy')[0]] = np.load(os.path.join(features_path, file_name))
    return features

def collate_batches(batch_data):
    batch_data.sort(key=lambda x: len(x[1]), reverse=True)
    video_ids, sentences = zip(*batch_data) 
    video_features = torch.stack(video_ids, 0)

    sentence_lengths = [len(sentence) for sentence in sentences]
    padded_sentences = torch.zeros(len(sentences), max(sentence_lengths)).long()
    for i, sentence in enumerate(sentences):
        end = sentence_lengths[i]
        padded_sentences[i, :end] = torch.LongTensor(sentence)

    return video_features, padded_sentences, sentence_lengths

class VideoCaptionDataset(Dataset):
    def __init__(self, label_file, features_dir, vocab, word_to_index):
        self.annotations = tag_captions(features_dir, vocab, word_to_index)
        self.features = load_features(label_file)
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        video_id, sentence = self.annotations[idx]
        feature = torch.Tensor(self.features[video_id])
        feature += torch.randn(feature.size()) * 0.0002
        return feature, torch.LongTensor(sentence)

class Encoder(nn.Module):
    def __init__(self, input_dim=4096, hidden_dim=512, dropout_rate=0.3):
        super(Encoder, self).__init__()
        self.linear_compress = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, inputs):
        compressed = self.dropout(F.relu(self.linear_compress(inputs.view(-1, 4096))))
        outputs, hidden = self.gru(compressed.view(inputs.size(0), inputs.size(1), -1))
        return outputs, hidden

class Decoder(nn.Module):
    def __init__(self, hidden_dim=512, vocab_size=1000, embedding_dim=1024, dropout_rate=0.3):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.attention = Attention(hidden_dim)
        self.gru = nn.GRU(hidden_dim + embedding_dim, hidden_dim, batch_first=True)
        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, hidden, encoder_outputs, targets=None, mode='train', step=None):
        if mode == 'train':
            embedded = self.embedding(targets)
            outputs, hidden = self.process_sequences(hidden, encoder_outputs, embedded)
            return F.log_softmax(self.out(outputs), dim=-1), outputs.argmax(-1)
        else:
            return self.generate(hidden, encoder_outputs)

    def process_sequences(self, hidden, encoder_outputs, embedded):
        seq_len = embedded.size(1)
        outputs = []
        for i in range(seq_len - 1):
            attention_weighted_encoding = self.attention(hidden, encoder_outputs)
            gru_input = torch.cat((embedded[:, i], attention_weighted_encoding), dim=1).unsqueeze(1)
            output, hidden = self.gru(gru_input, hidden)
            outputs.append(output.squeeze(1))
        return torch.stack(outputs, dim=1), hidden

    def generate(self, hidden, encoder_outputs, max_len=20):
        inputs = torch.ones(encoder_outputs.size(0), 1).long().to(compute_device)
        outputs = []
        for _ in range(max_len):
            embedded = self.embedding(inputs).squeeze(1)
            attention_weighted_encoding = self.attention(hidden, encoder_outputs)
            gru_input = torch.cat((embedded, attention_weighted_encoding), dim=1).unsqueeze(1)
            output, hidden = self.gru(gru_input, hidden)
            output = F.log_softmax(self.out(output.squeeze(1)), dim=-1)
            _, topi = output.topk(1)
            inputs = topi.detach()
            outputs.append(output)
        return torch.cat(outputs, dim=1), inputs

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim * 2, hidden_dim)
        self.final = nn.Linear(hidden_dim, 1, bias=False)
    
    def forward(self, hidden, encoder_outputs):
        combined = torch.cat((hidden.repeat(1, encoder_outputs.size(1), 1), encoder_outputs), dim=2)
        energy = F.relu(self.attention(combined.view(-1, combined.size(2))))
        attention = F.softmax(self.final(energy).view(hidden.size(0), -1), dim=1).unsqueeze(1)
        context = torch.bmm(attention, encoder_outputs).squeeze(1)
        return context

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, inputs, targets=None, mode='train', step=None):
        encoder_outputs, hidden = self.encoder(inputs)
        if mode == 'train':
            return self.decoder(hidden, encoder_outputs, targets, mode, step)
        else:
            return self.decoder.generate(hidden, encoder_outputs)

def train_model(model, dataloader, optimizer, criterion, epoch, device):
    model.train()
    total_loss = 0
    for batch_idx, (inputs, targets, _) in enumerate(dataloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        output, _ = model(inputs, targets, mode='train', step=epoch)
        loss = criterion(output.transpose(1, 2), targets[:, 1:])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch: {epoch}, Loss: {total_loss / len(dataloader)}')

def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _, predicted = model(inputs, mode='inference')
            predictions.extend(predicted.cpu().numpy())
    return predictions

def main():
    index_to_word, word_to_index, vocab = preprocess_data()
    train_ds = VideoCaptionDataset('/training_data/feat', 'training_label.json', vocab, word_to_index)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, collate_fn=collate_batches)
    encoder = Encoder()
    decoder = Decoder(vocab_size=len(index_to_word))
    seq2seq = Seq2Seq(encoder, decoder).to(compute_device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(1, 21):
        train_model(seq2seq, train_dl, optimizer, criterion, epoch, compute_device)

    print("Training complete.")

if __name__ == '__main__':
    main()
