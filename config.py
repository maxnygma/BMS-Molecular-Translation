from omegaconf import OmegaConf, DictConfig, ListConfig

config = {
    'general': {
        'experiment_name': 'default',
        'seed': 42,
        'num_classes': 2
    },
    'paths': {
        'path_to_images': 'data/train/', 
        'path_to_csv': 'train_pseudo_labels.pkl',
        'path_to_checkpoints': 'checkpoints/',
        'path_to_tokenizer_weights': 'data/tokenizer2.pth',
        
        'path_to_pseudo_labels_csv': 'data/submission_norm.csv',
        'path_to_pseudo_labels_images': 'data/train_pseudo_labels/0/0/0/',

        'path_to_sample_submission': 'data/sample_submission.csv',
        'path_to_predicted_submission': 'submission.csv',
        'path_to_inference_images': 'data/test/',

        'log_name': 'log.txt',
        'path_to_weights': '/home/xzcodes/Downloads/4_epoch.pt'
    },
    'training': {
        'num_epochs': 20,
        # 'encoder_lr': 0.000001, # target learning rate = base lr * warmup_multiplier if warmup_multiplier > 1.0 (0.000001)
        # 'decoder_lr': 0.000004,
        'encoder_lr': 0.0001, # target learning rate = base lr * warmup_multiplier if warmup_multiplier > 1.0 (0.000001)
        'decoder_lr': 0.0004,
        'mixed_precision': True,
        'gradient_accumulation': False,
        'gradient_accumulation_steps': 4,
        'gradient_clipping': True,
        'gradient_clipping_value': 5,
        'early_stopping_epochs': 15,

        'warmup_scheduler': True,
        'warmup_epochs': 3,
        'warmup_multiplier': 100, # lr needs to be divided by warmup_multiplier if warmup_multiplier > 1.0

        'debug': False,
        'debug_number_of_samples': 100000,

        'device': 'cuda',
        'save_step': 1,
        'verbose_plots': True
    },
    'data': {
        'id_column': 'image_id',
        'target_column': 'InChI_text', # list of columns
        'image_format': '.png',

        'train_batch_size': 16,
        'val_batch_size': 16,
        'test_batch_size': 1,
        'num_workers': 8,

        'kfold': {
            'use_kfold': True,
            'name': 'StratifiedKFold',
            'split_on_column': 'InChI_length', 
            'group_column': None, # used only for GroupKFold (str or None)
            'current_fold': 0, # 0 - 4
            'params': {
                'n_splits': 20,
                'shuffle': True,
                'random_state': '${general.seed}'
            }
        },

        'train_size': 0.8,
        'val_size': 0.2,
        'test_size': 0.0,

        'start_size': 320,
        'final_size': 320,
        'step_size': 32,

        # Not implemented
        'cutmix': False,
        'mixup': False,
        'fmix': False
    },
    'augmentations': {
        'pre_transforms': [
            {
                'name': 'Resize',
                'params': {
                    'height': '${data.start_size}',
                    'width': '${data.start_size}',
                    'p': 1.0
                }
            }
        ],
        'transforms': [
            {
                'name': '/custom/SaltAndPepperNoise',
                'params': {
                    's_vs_p': 0.1,
                    'amount': 0.0015,
                    'p': 0.5
                }
            },
            # {
            #     'name': 'HorizontalFlip',
            #     'params': {
            #         'p': 0.5
            #     }
            # },
        ],
        'post_transforms': [
            {
                'name': 'Normalize',
                'params': {
                    'p': 1.0
                }
            },
            {
                'name': '/custom/to_tensor',
                'params': {
                    'p': 1.0
                }
            }
        ]
    },
    'model': {
        'recurrent_based': {
            'use': True,
            'decoder_type': 'gru', # lstm or gru

            'encoder_dimension': 512,
            'decoder_dimension': 512,
            'attention_dimension': 256,

            'embedding_dimension': 256,
            'vocabulary_size': 193, # length of a tokenizer
            'max_length': 275,
            'dropout_rate': 0.5,
        },
        'transformer_based': {
            'use': False # not implemented
        },
        'backbone_name': '/timm/tf_efficientnet_b4', 
        'freeze_batchnorms': False,
        'pretrained': True
    },
    'encoder_optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.encoder_lr}'
        }
    },
    'decoder_optimizer': {
        'name': 'Adam',
        'params': {
            'lr': '${training.decoder_lr}'
        }
    },
    'encoder_scheduler': {
        'name': 'CosineAnnealingLR', 
        'interval': 'epoch', # epoch or step
        'params': {
            'T_max': 10,
            'eta_min': 1e-7
        }
    },
    'decoder_scheduler': {
        'name': 'CosineAnnealingLR', 
        'interval': 'epoch', # epoch or step
        'params': {
            'T_max': 10,
            'eta_min': 1e-7
        }
    },
    'loss': {
        'name': 'CrossEntropyLoss',
        'ohem_loss': False,
        'ohem_rate': 0.0,
        'params': {
            'ignore_index': 192, # tokenizer.stoi['<pad>']
        }
    },
    'metric': {
        'name': '/custom/get_levenshtein_score',
        'params': {
            
        }
    },
    'logging': {
        'log': False,
        'wandb_username': 'xzcodes',
        'wandb_project_name': 'bms_molecular_translation'
    },
    'inference': {
        # Not implemented 

        'inference': False,
        'batch_size': 1,
        'preds_columns': 'pred',
        'list_of_models': [
            '/timm/resnet18',
        ],
        'list_of_checkpoints': [
            'checkpoints/best.pt',
        ]
    },
    'resume_from_checkpoint': {
        'resume': True,
    } 
}

config = OmegaConf.create(config)
