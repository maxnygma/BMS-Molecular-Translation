import numpy as np
import pandas as pd

import os
import cv2

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from sklearn import model_selection
from sklearn.model_selection import train_test_split

from custom.augmentations import *


class Tokenizer(object):
    def __init__(self):
        self.stoi = {}
        self.itos = {}
        
    def __len__(self):
        return len(self.stoi)
    
    def fit_on_texts(self, texts):
        vocab = set()
        
        for text in texts:
            vocab.update(text.split(' '))
            
        vocab = sorted(vocab)
        vocab.append('<sos>')
        vocab.append('<eos>')
        vocab.append('<pad>')
        
        for i, string in enumerate(vocab):
            self.stoi[string] = i
            
        self.itos = {item[1]: item[0] for item in self.stoi.items()}
        
    def text_to_sequence(self, text):
        sequence = []
        
        sequence.append(self.stoi['<sos>'])
        
        for string in text.split(' '):
            sequence.append(self.stoi[string])
            
        sequence.append(self.stoi['<eos>'])
        
        return sequence
    
    def texts_to_sequences(self, texts):
        sequences = []
        
        for text in texts:
            sequence = self.text_to_sequence(text)
            sequences.append(sequence)
            
        return sequences
    
    def sequence_to_text(self, sequence):
        return ''.join(list(map(lambda x: self.itos[x], sequence)))
    
    def sequences_to_texts(self, sequences):
        texts = []
        
        for sequence in sequences:
            text = self.sequence_to_text(sequence)
            texts.append(text)
            
        return texts
    
    def predict_caption(self, sequence):
        caption = ''
        
        for x in sequence:
            if x == self.stoi['<eos>'] or x == self.stoi['<pad>']:
                break
                
            caption += self.itos[x]
            
        return caption
    
    def predict_captions(self, sequences):
        captions = []
        
        for sequence in sequences:
            caption = self.predict_caption(sequence)
            captions.append(caption)
            
        return captions


class MolecularDataset(Dataset):
    def __init__(self, config, df, tokenizer, transforms=None):
        super().__init__()
        
        self.df = df
        self.tokenizer = tokenizer
        self.transforms = transforms
        self.image_paths = df['image_path'].values
        self.labels = df[config.data.target_column].values
        self.fix_rotate_transform = A.Compose([A.Transpose(p=1.0), A.VerticalFlip(p=1.0)])
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Fix 90° rotated images
        h, w, _ = image.shape
        if h > w:
            image = self.fix_rotate_transform(image=image)['image']
        
        if self.transforms:
            image = self.transforms(image=image)['image']
        
        label = self.labels[idx]
        label = self.tokenizer.text_to_sequence(label)
        label = torch.LongTensor(label)
        
        label_length = torch.LongTensor([len(label)])
        
        return image, label, label_length


def get_transforms(config):
    '''Get train and validation augmentations.'''

    pre_transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                     else getattr(A, item['name'])(**item['params']) for item in config.augmentations.pre_transforms]
    transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                     else getattr(A, item['name'])(**item['params']) for item in config.augmentations.transforms]
    post_transforms = [globals()[item['name'][8:]](**item['params']) if item['name'].startswith('/custom/')
                      else getattr(A, item['name'])(**item['params']) for item in config.augmentations.post_transforms]

    train_transforms = A.Compose(pre_transforms + transforms + post_transforms)
    val_transforms = A.Compose(pre_transforms + post_transforms)

    return train_transforms, val_transforms


def data_generator(config):
    '''Generate data for train and validation splits.'''

    print('Getting the data')

    assert config.data.train_size + config.data.val_size + config.data.test_size == 1.0, 'sum of the sizes of splits must be equal to 1.0'

    data = pd.read_pickle(config.paths.path_to_csv)

    if config.training.debug:
        data = data.sample(n=config.training.debug_number_of_samples, random_state=config.general.seed).reset_index(drop=True)

    if config.data.kfold.use_kfold:
        kfold = getattr(model_selection, config.data.kfold.name)(**config.data.kfold.params)
        current_fold = config.data.kfold.current_fold
        
        if config.data.kfold.group_column:
            groups = data[config.data.kfold.group_column]
        else:
            groups = None
        
        for fold, (train_index, val_index) in enumerate(kfold.split(data, data[config.data.kfold.split_on_column], groups)):
            if fold == config.data.kfold.current_fold:
                train_images = data[config.data.id_column].iloc[train_index].values
                train_targets = data[config.data.target_column].iloc[train_index].values
                val_images = data[config.data.id_column].iloc[val_index].values
                val_targets = data[config.data.target_column].iloc[val_index].values

                break
        
        if config.data.test_size == 0.0:
            return train_images, train_targets, val_images, val_targets
        
        val_size = config.data.val_size / (config.data.val_size + config.data.test_size)
        test_size = config.data.test_size / (config.data.val_size + config.data.test_size)
        val_images, test_images, val_targets, test_targets = train_test_split(val_images, val_targets,
                                                                              train_size=val_size,
                                                                              test_size=test_size,
                                                                              random_state=config.general.seed,
                                                                              stratify=val_targets)

        return train_images, train_targets, val_images, val_targets, test_images, test_targets
    else:
        pass


def collate_function(batch):
    images, labels, label_lengths = [], [], []

    for data_point in batch:
        images.append(data_point[0])
        labels.append(data_point[1])
        label_lengths.append(data_point[2])

    labels = pad_sequence(labels, batch_first=True, padding_value=192) # padding_value = tokenizer.stoi['<pad>']

    return torch.stack(images), labels, torch.stack(label_lengths).reshape(-1, 1)


def get_loaders(config):
    '''Get data loaders.'''

    train_transforms, val_transorms = get_transforms(config)
    df = pd.read_pickle(config.paths.path_to_csv)
    tokenizer = torch.load(config.paths.path_to_tokenizer_weights)

    if config.data.test_size == 0.0:
        train_images, train_targets, val_images, val_targets = data_generator(config)
        val_inchi_labels = df['InChI'][df[config.data.id_column].isin(val_images)].values

        train_dataset = MolecularDataset(config, df[df[config.data.id_column].isin(train_images)], tokenizer, train_transforms)
        val_dataset = MolecularDataset(config, df[df[config.data.id_column].isin(val_images)], tokenizer, val_transorms)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.data.train_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True, collate_fn=collate_function)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True, collate_fn=collate_function)

        return train_loader, val_loader, val_inchi_labels
    else:
        train_images, train_targets, val_images, val_targets, test_images, test_targets = data_generator(config)

        train_dataset = MolecularDataset(config, df[df[config.data.id_column].isin(train_images)], tokenizer, train_transforms)
        val_dataset = MolecularDataset(config, df[df[config.data.id_column].isin(val_images)], tokenizer, val_transorms)
        test_dataset = MolecularDataset(config, df[df[config.data.id_column].isin(test_images)], tokenizer, val_transorms)

        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=config.data.train_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True, collate_fn=collate_function)
        val_loader = DataLoader(val_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=config.data.val_batch_size, pin_memory=False,
                                  num_workers=config.data.num_workers, persistent_workers=True)

        return train_loader, val_loader, test_loader


def get_loader_inference(config):
    pass
