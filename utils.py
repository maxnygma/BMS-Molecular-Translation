import numpy as np  
import pandas as pd

import os
import gc
import time
import wandb
import random
from tqdm import tqdm

import sklearn

import torch
import timm
import torchvision
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.cuda import amp

import matplotlib.pyplot as plt

from config import config
from data import get_loaders, get_loader_inference
from custom.scheduler import GradualWarmupSchedulerV2
from custom.metric import get_levenshtein_score


def set_seed(seed: int):
    '''Set a random seed for complete reproducibility.'''

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_backbone(config_copy):
    '''Get PyTorch model for backbone.'''

    print('Backbone:', config_copy.model.backbone_name)

    if config_copy.model.backbone_name.startswith('/timm/'): 
        model = timm.create_model(config_copy.model.backbone_name[6:], pretrained=config_copy.model.pretrained)
    elif config_copy.model.backbone_name.startswith('/torch/'):
        model = getattr(torchvision.models, config_copy.model.backbone_name[7:])(pretrained=config_copy.model.pretrained)
    else:
        raise RuntimeError('Unknown model source. Use /timm/ or /torch/.')
    
    last_layer = list(model._modules)[-1]
    dimension = getattr(model, last_layer).in_features

    setattr(model, last_layer, nn.Identity())
    model.global_pool = nn.Identity()

    config.model.recurrent_based.encoder_dimension = dimension
    config.model.recurrent_based.decoder_dimension = dimension

    return model


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.backbone = get_backbone(config)
        
    def forward(self, x):
        batch_size = x.size(0)

        features = self.backbone(x)
        features = features.permute(0, 2, 3, 1) 

        return features


class Attention(nn.Module):
    def __init__(self, config):
        super(Attention, self).__init__()

        self.encoder_linear = nn.Linear(config.model.recurrent_based.encoder_dimension, config.model.recurrent_based.attention_dimension)
        self.decoder_linear = nn.Linear(config.model.recurrent_based.decoder_dimension, config.model.recurrent_based.attention_dimension) 
        self.final_linear = nn.Linear(config.model.recurrent_based.attention_dimension, 1)

        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, encoder_output, decoder_hidden):
        encoder_linear_out = self.encoder_linear(encoder_output)
        decoder_linear_out = self.decoder_linear(decoder_hidden)
        final_out = self.final_linear(self.relu(encoder_linear_out + decoder_linear_out.unsqueeze(1))).squeeze(2)

        alpha = self.softmax(final_out)
        attention_weighted_encoding = (encoder_output * alpha.unsqueeze(2)).sum(dim=1)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, config):
        super(DecoderWithAttention, self).__init__()

        self.config = config
        self.encoder_dimension = config.model.recurrent_based.encoder_dimension
        self.decoder_dimension = config.model.recurrent_based.decoder_dimension
        self.attention_dimension = config.model.recurrent_based.attention_dimension
        self.embedding_dimension = config.model.recurrent_based.embedding_dimension
        self.vocabulary_size = config.model.recurrent_based.vocabulary_size
        self.dropout = config.model.recurrent_based.dropout_rate
        self.device = config.training.device

        self.attention = Attention(config)
        self.embedding = nn.Embedding(self.vocabulary_size, self.embedding_dimension)
        self.dropout = nn.Dropout(p=self.dropout)
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(self.decoder_dimension, self.vocabulary_size)
        
        if config.model.recurrent_based.decoder_type == 'lstm':
            self.decode_step = nn.LSTMCell(self.embedding_dimension + self.encoder_dimension, self.decoder_dimension, bias=True)
        elif config.model.recurrent_based.decoder_type =='gru':
            self.decode_step = nn.GRUCell(self.embedding_dimension + self.encoder_dimension, self.decoder_dimension, bias=True)
        self.init_hidden = nn.Linear(self.encoder_dimension, self.decoder_dimension) # hidden state of LSTM
        self.init_cell = nn.Linear(self.encoder_dimension, self.decoder_dimension) # initial cell state of LSTM
        self.f_beta = nn.Linear(self.encoder_dimension, self.decoder_dimension)

        self.init_weigths()

    def init_weigths(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_lstm_hidden_state(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1)

        h = self.init_hidden(mean_encoder_output)
        c = self.init_cell(mean_encoder_output)

        return h, c

    def init_gru_hidden_state(self, encoder_output):
        mean_encoder_output = encoder_output.mean(dim=1)

        h = self.init_hidden(mean_encoder_output)

        return h

    def forward(self, encoder_output, encoded_captions, caption_lengths):
        batch_size = encoder_output.size(0)
        encoder_dimension = encoder_output.size(-1)

        encoder_output = encoder_output.view(batch_size, -1, encoder_dimension)
        num_pixels = encoder_output.size(1)

        caption_lengths, sort_index = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_output = encoder_output[sort_index]
        encoded_captions = encoded_captions[sort_index]

        embeddings = self.embedding(encoded_captions)

        if config.model.recurrent_based.decoder_type == 'lstm':
            h, c = self.init_lstm_hidden_state(encoder_output)
        elif config.model.recurrent_based.decoder_type =='gru':
            h = self.init_gru_hidden_state(encoder_output)

        decode_lengths = (caption_lengths - 1).tolist() # excluding start token
        predictions = torch.zeros(batch_size, max(decode_lengths), self.vocabulary_size).to(self.device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(self.device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            
            attention_weighted_encoding, alpha = self.attention(encoder_output[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))

            attention_weighted_encoding = gate * attention_weighted_encoding

            if config.model.recurrent_based.decoder_type == 'lstm':
                h, c = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                                    (h[:batch_size_t], c[:batch_size_t]))
            elif config.model.recurrent_based.decoder_type =='gru':
                h = self.decode_step(torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1), h[:batch_size_t])

            preds = self.fc(self.dropout(h))
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_index

    def predict(self, encoder_output, decode_lengths, tokenizer):
        batch_size = encoder_output.size(0)
        encoder_dimension = encoder_output.size(-1)
        
        encoder_output = encoder_output.view(batch_size, -1, encoder_dimension)
        num_pixels = encoder_output.size(1)

        start_tokens = torch.ones(batch_size, dtype=torch.long).to(self.device) * tokenizer.stoi['<sos>']
        embeddings = self.embedding(start_tokens)

        if config.model.recurrent_based.decoder_type == 'lstm':
            h, c = self.init_lstm_hidden_state(encoder_output)
        elif config.model.recurrent_based.decoder_type =='gru':
            h = self.init_gru_hidden_state(encoder_output)

        predictions = torch.zeros(batch_size, decode_lengths, self.vocabulary_size).to(self.device)

        for t in range(decode_lengths):
            attention_weighted_encoding, alpha = self.attention(encoder_output, h)
            gate = self.sigmoid(self.f_beta(h))

            attention_weighted_encoding = gate * attention_weighted_encoding

            if config.model.recurrent_based.decoder_type == 'lstm':
                h, c = self.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), (h, c))
            elif config.model.recurrent_based.decoder_type =='gru':
                h = self.decode_step(torch.cat([embeddings, attention_weighted_encoding], dim=1), h)

            preds = self.fc(self.dropout(h))
            predictions[:, t, :] = preds

            # if np.argmax(preds.detach().cpu().numpy()) == tokenizer.stoi['<eos>']:
            #     break

            embeddings = self.embedding(torch.argmax(preds, -1))

        return predictions


def get_optimizer(config, model):
    '''Get PyTorch optimizer'''
    
    if isinstance(model, Encoder):
        if config.encoder_optimizer.name.startswith('/custom/'):
            encoder_optimizer = globals()[config.encoder_optimizer.name[8:]](model.parameters(), **config.encoder_optimizer.params)
        else:
            encoder_optimizer = getattr(torch.optim, config.encoder_optimizer.name)(model.parameters(), **config.encoder_optimizer.params)

        return encoder_optimizer
    else:
        if config.decoder_optimizer.name.startswith('/custom/'):
            decoder_optimizer = globals()[config.decoder_optimizer.name[8:]](model.parameters(), **config.decoder_optimizer.params)
        else:
            decoder_optimizer = getattr(torch.optim, config.decoder_optimizer.name)(model.parameters(), **config.decoder_optimizer.params)

        return decoder_optimizer


def get_scheduler(config, optimizer, model_type):
    '''Get PyTorch scheduler'''

    if model_type == 'Encoder':
        if config.encoder_scheduler.name.startswith('/custom/'):
            encoder_scheduler = globals()[config.encoder_scheduler.name[8:]](optimizer, **config.encoder_scheduler.params)
        else:
            encoder_scheduler = getattr(torch.optim.lr_scheduler, config.encoder_scheduler.name)(optimizer, **config.encoder_scheduler.params)
    
        if config.training.warmup_scheduler:
            final_scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=config.training.warmup_multiplier,
                                                 total_epoch=config.training.warmup_epochs, after_scheduler=encoder_scheduler)

            return final_scheduler
        else:
            return encoder_scheduler
    else:
        if config.decoder_scheduler.name.startswith('/custom/'):
            decoder_scheduler = globals()[config.decoder_scheduler.name[8:]](optimizer, **config.decoder_scheduler.params)
        else:
            decoder_scheduler = getattr(torch.optim.lr_scheduler, config.decoder_scheduler.name)(optimizer, **config.decoder_scheduler.params)
    
        if config.training.warmup_scheduler:
            final_scheduler = GradualWarmupSchedulerV2(optimizer, multiplier=config.training.warmup_multiplier,
                                                 total_epoch=config.training.warmup_epochs, after_scheduler=decoder_scheduler)

            return final_scheduler
        else:
            return decoder_scheduler

    
def get_loss(config):
    '''Get PyTorch loss function.'''

    if config.loss.name.startswith('/custom/'):
        loss = globals()[config.loss.name[8:]](**config.loss.params)
    else:
        loss = getattr(nn, config.loss.name)(**config.loss.params)

    return loss


def get_score(config, y_true, y_pred):
    '''Calculate metric.'''

    if config.metric.name.startswith('/custom/'):
        score = globals()[config.metric.name[8:]](y_true, y_pred, **config.metric.params)
    else:
        score = getattr(sklearn.metrics, config.metric.name)(y_true, y_pred, **config.metric.params)
    
    return score


def train(config, encoder, decoder, train_loader, loss_function, encoder_optimizer, 
          decoder_optimizer, encoder_scheduler, decoder_scheduler, epoch, scaler):
    total_loss = 0.0

    encoder.train()
    decoder.train()

    global_step = 0

    for step, (images, labels, label_lengths) in enumerate(tqdm(train_loader)):
        images = images.to(config.training.device)
        labels = labels.to(config.training.device)
        label_lengths = label_lengths.to(config.training.device)

        batch_size = images.size(0)

        if not config.training.gradient_accumulation:
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
        
        if config.training.mixed_precision:
            with amp.autocast():
                features = encoder(images)
                predictions, captions_sorted, decode_lengths, alphas, sort_index = decoder(features, labels, label_lengths)

                targets = captions_sorted[:, 1:]

                predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

                loss = loss_function(predictions, targets)

                if config.training.gradient_accumulation:
                    loss = loss / config.training.gradient_accumulation_steps
        else:
            features = encoder(images)
            predictions, captions_sorted, decode_lengths, alphas, sort_index = decoder(features, labels, label_lengths)

            targets = captions_sorted[:, 1:]

            predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
            targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data

            loss = loss_function(predictions, targets)

        total_loss += loss.item()

        if config.training.gradient_clipping:
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), config.training.gradient_clipping_value)
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), config.training.gradient_clipping_value)

        if config.training.mixed_precision:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if config.training.gradient_accumulation:
            if (step + 1) % config.training.gradient_accumulation_steps == 0:
                scaler.step(encoder_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()
                encoder_optimizer.zero_grad()
                decoder_optimizer.zero_grad()

                global_step += 1
        elif config.training.mixed_precision:
            scaler.step(encoder_optimizer)
            scaler.step(decoder_optimizer)
            scaler.update()
        else:
            encoder_optimizer.step()
            decoder_optimizer.step()

        if config.encoder_scheduler.interval == 'step':
            if config.training.warmup_scheduler:
                if epoch >= config.training.warmup_epochs:
                    encoder_scheduler.step()
            else:
                encoder_scheduler.step()

        if config.decoder_scheduler.interval == 'step':
            if config.training.warmup_scheduler:
                if epoch >= config.training.warmup_epochs:
                    decoder_scheduler.step()
            else:
                decoder_scheduler.step()

    if config.training.warmup_scheduler:
        if epoch < config.training.warmup_epochs:
            encoder_scheduler.step()
            decoder_scheduler.step()
        else:
            if config.encoder_scheduler.interval == 'epoch':
                encoder_scheduler.step()
            if config.decoder_scheduler.interval == 'epoch':
                decoder_scheduler.step()
    else:
        if config.encoder_scheduler.interval == 'epoch':
            encoder_scheduler.step()
        if config.decoder_scheduler.interval == 'epoch':
            decoder_scheduler.step()

    # Add warmup support
    return total_loss / len(train_loader)


def validation(config, encoder, decoder, val_loader, tokenizer):
    encoder.eval()
    decoder.eval()

    text_preds = []

    for step, (images, labels, label_lengths) in enumerate(tqdm(val_loader)):
        images = images.to(config.training.device) 

        batch_size = images.size(0)

        with torch.no_grad():
            features = encoder(images)
            predictions = decoder.predict(features, config.model.recurrent_based.max_length, tokenizer)

        predicted_sequence = torch.argmax(predictions.detach().cpu(), -1).numpy()

        _text_preds = tokenizer.predict_captions(predicted_sequence)
        text_preds.append(_text_preds)

    text_preds = np.concatenate(text_preds)

    return text_preds


def run(config):
    '''Main function.'''

    if config.logging.log:
        wandb.init(project=config.logging.wandb_project_name, entity=config.logging.wandb_username)

    # Create working directory
    if not os.path.exists(config.paths.path_to_checkpoints):
        os.makedirs(config.paths.path_to_checkpoints)

    # Get objects
    tokenizer = torch.load(config.paths.path_to_tokenizer_weights)

    if config.data.test_size == 0.0:
        train_loader, val_loader, val_targets = get_loaders(config)
    else:
        train_loader, val_loader, test_loader, val_targets = get_loaders(config)

    torch.cuda.empty_cache()

    encoder = Encoder(config)
    encoder = encoder.to(config.training.device)
    encoder_optimizer = get_optimizer(config, encoder)
    encoder_scheduler = get_scheduler(config, encoder_optimizer, model_type='Encoder')

    decoder = DecoderWithAttention(config)
    decoder = decoder.to(config.training.device)
    decoder_optimizer = get_optimizer(config, decoder)
    decoder_scheduler = get_scheduler(config, decoder_optimizer, model_type='Decoder')

    loss_function = get_loss(config)

    if config.training.mixed_precision:
        scaler = amp.GradScaler()
    else:
        scaler = None

    train_losses, metrics, learning_rates_encoder, learning_rates_decoder = [], [], [], []
    best_metric = np.inf

    epochs_since_improvement = 0

    print('Testing ' + config.general.experiment_name + ' approach')
    if config.paths.log_name:
        with open(os.path.join(config.paths.path_to_checkpoints, config.paths.log_name), 'w') as file:
            file.write('Testing ' + config.general.experiment_name + ' approach\n')
    
    # Training
    # Store transforms to use them after warmup stage
    transforms = config.augmentations.transforms
    image_size = config.data.start_size

    current_epoch = 0

    # Load model chckpoint if needed
    if config.resume_from_checkpoint.resume:
        load_dict = load_model(config, encoder, decoder, 
                               encoder_optimizer, decoder_optimizer, 
                               encoder_scheduler, decoder_scheduler)

        encoder = load_dict['encoder']
        decoder = load_dict['decoder']

        encoder_optimizer = load_dict['encoder_optimizer']
        decoder_optimizer = load_dict['decoder_optimizer']
        encoder_scheduler = load_dict['encoder_scheduler']
        decoder_scheduler = load_dict['decoder_scheduler']

        current_epoch = load_dict['epoch']
        best_metric = load_dict['metric']
        epochs_since_improvement = load_dict['epochs_since_improvement']
        image_size = load_dict['image_size']

    for epoch in range(current_epoch, config.training.num_epochs):
        print('\nEpoch: ' + str(epoch + 1))

        # Applying progressive resizing
        if image_size < config.data.final_size and epoch > config.training.warmup_epochs:
            image_size += config.data.size_step
            config.augmentations.pre_transforms = [
                {
                    'name': 'Resize',
                    'params': {
                        'height': image_size,
                        'width': image_size,
                        'p': 1.0
                    }
                }
            ]
        
        # No transforms for warmup stage
        if epoch < config.training.warmup_epochs:
            config.augmentations.transforms = []
        else:
            config.augmentations.transforms = transforms

        print('Image size: ' + str(image_size))

        start_time = time.time()

        train_loss = train(config, encoder, decoder, train_loader, loss_function, 
                           encoder_optimizer, decoder_optimizer, encoder_scheduler,
                           decoder_scheduler, epoch, scaler)

        text_preds = validation(config, encoder, decoder, val_loader, tokenizer)
        text_preds = [f'InChI=1S/{text}' for text in text_preds]
        print('Target: ', val_targets[0])
        print('Prediction: ', text_preds[0])

        current_metric = get_score(config, val_targets, text_preds)

        train_losses.append(train_loss)
        metrics.append(current_metric)
        learning_rates_encoder.append(encoder_optimizer.param_groups[0]['lr'])
        learning_rates_decoder.append(decoder_optimizer.param_groups[0]['lr'])
        
        if config.logging.log:
            wandb.log({'train_loss': train_loss, 'val_metric': current_metric, 'epoch': epoch,
                       'learning_rate_encoder': encoder_optimizer.param_groups[0]['lr'], 
                       'learning_rate_decoder': decoder_optimizer.param_groups[0]['lr']})

        if current_metric < best_metric:
            print('New Record!')

            epochs_since_improvement = 0
            best_metric = current_metric

            save_model(config, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                       encoder_scheduler, decoder_scheduler, epoch, train_loss, 
                       current_metric, epochs_since_improvement, image_size, 'best.pt') 
        else:
            epochs_since_improvement += 1

        if epoch % config.training.save_step == 0:
            save_model(config, encoder, decoder, encoder_optimizer, decoder_optimizer, 
                       encoder_scheduler, decoder_scheduler, epoch, train_loss, 
                       current_metric, epochs_since_improvement, image_size, f'{epoch + 1}_epoch.pt')

        t = int(time.time() - start_time)
        print_report(t, train_loss, current_metric, best_metric)

        if config.paths.log_name:
            save_log(os.path.join(config.paths.path_to_checkpoints, config.paths.log_name), epoch + 1,
                    train_loss, best_metric)

        if epochs_since_improvement == config.training.early_stopping_epochs:
            print('Training has been interrupted by early stopping.')
            break

        torch.cuda.empty_cache()
        gc.collect()

    if config.data.test_size > 0.0:
        #test(config, model, test_loader, loss_function)
        pass

    if config.training.verbose_plots:
        draw_plots(train_losses, metrics, learning_rates_encoder, learning_rates_decoder)


def save_model(config, encoder, decoder, encoder_optimizer, decoder_optimizer, 
               encoder_scheduler, decoder_scheduler, epoch, train_loss, metric, 
               epochs_since_improvement, image_size, name):
    '''Save PyTorch model.'''

    torch.save({
        'encoder': encoder.state_dict(),
        'decoder': decoder.state_dict(),

        'encoder_optimizer': encoder_optimizer.state_dict(),
        'decoder_optimizer': decoder_optimizer.state_dict(),
        'encoder_scheduler': encoder_scheduler.state_dict(),
        'decoder_scheduler': decoder_scheduler.state_dict(),

        'epoch': epoch,
        'train_loss': train_loss,
        'metric': metric,
        'epochs_since_improvement': epochs_since_improvement,
        'image_size': image_size,
    }, os.path.join(config.paths.path_to_checkpoints, name))


def load_model(config, encoder, decoder, encoder_optimizer, decoder_optimizer, 
               encoder_scheduler, decoder_scheduler):
    checkpoint = torch.load(config.paths.path_to_weights)

    encoder.load_state_dict(checkpoint['encoder'])
    decoder.load_state_dict(checkpoint['decoder'])

    encoder_optimizer.load_state_dict(checkpoint['encoder_optimizer'])
    decoder_optimizer.load_state_dict(checkpoint['decoder_optimizer'])

    encoder_scheduler.load_state_dict(checkpoint['encoder_scheduler'])
    decoder_scheduler.load_state_dict(checkpoint['decoder_scheduler'])

    epoch = checkpoint['epoch']
    train_loss = checkpoint['train_loss']
    metric = checkpoint['metric']
    epochs_since_improvement = checkpoint['epochs_since_improvement']
    image_size = checkpoint['image_size']

    return {
        'encoder': encoder,
        'decoder': decoder,

        'encoder_optimizer': encoder_optimizer, 
        'decoder_optimizer': decoder_optimizer,
        'encoder_scheduler': encoder_scheduler,
        'decoder_scheduler': decoder_scheduler,

        'epoch': epoch,
        'metric': metric,
        'epochs_since_improvement': epochs_since_improvement,
        'image_size': image_size
    }
    

def draw_plots(train_losses, metrics, lr_changes_encoder, lr_changes_decoder):
    '''Draw plots of losses, metrics and learning rate changes.'''

    # Learning rate changes
    plt.plot(range(len(lr_changes_encoder)), lr_changes_encoder, label='Learning Rate Encoder')
    plt.plot(range(len(lr_changes_decoder)), lr_changes_decoder, label='Learning Rate Decoder')
    plt.legend()
    plt.title('Learning rate changes')
    plt.show()

    # Validation and train losses
    plt.plot(range(len(train_losses)), train_losses, label='Train Loss')
    plt.legend()
    plt.title('Changes of train loss')
    plt.show()

    # Metric changes
    plt.plot(range(len(metrics)), metrics, label='Metric')
    plt.legend()
    plt.title('Metric changes')
    plt.show()


def print_report(t, train_loss, metric, best_metric):
    '''Print report of one epoch.'''

    print(f'Time: {t} s')
    print(f'Train Loss: {train_loss:.4f}')
    print(f'Current Metric: {metric:.4f}')
    print(f'Best Metric: {best_metric:.4f}')


def save_log(path, epoch, train_loss, best_metric):
    '''Save log of one epoch.'''

    with open(path, 'a') as file:
        file.write('epoch: ' + str(epoch) + 'train_loss: ' + str(round(train_loss, 5)) + 
                   ' best_metric: ' + str(round(best_metric, 5)) + '\n')


def get_model_inference(config):
    pass


def inference(config):
    pass



