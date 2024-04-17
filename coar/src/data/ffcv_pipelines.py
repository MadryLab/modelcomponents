from functools import partial
import torch
from ffcv.fields import decoders as DC
from ffcv import transforms as TF
import torchvision.transforms as T

DS_TO_MEAN = {
    'CIFAR': [125.307, 122.961, 113.8575],
    'CINIC': [0.47889522*255, 0.47227842*255, 0.43047404*255],
    'WATERBIRDS': [123.675, 116.28, 103.53],
    'CELEBA': [123.675, 116.28, 103.53],
    'LIVING17': [0.485*255, 0.456*255, 0.406*255],
    'YEARBOOK': [0.*255, 0.*255, 0.*255],
    'IMAGENET': [0.485*255, 0.456*255, 0.406*255],
    'IMAGENET_ADV': [0.485, 0.456, 0.406],
    'ADE': [0.485*255, 0.456*255, 0.406*255],
    'MIMIC': [0.485*255, 0.456*255, 0.406*255],
    'CHEXPERT': [0.485*255, 0.456*255, 0.406*255],
    'NIH': [0.485*255, 0.456*255, 0.406*255],
    'OBJECTNET20': [0.485*255, 0.456*255, 0.406*255],
    'OBJECTNET20_SAM': [0.*255, 0.*255, 0*255], # DM bug
    'DUMMY': [0,0,0],
    'OPENAI_CLIP': [0.48145466*255.0, 0.4578275*255.0, 0.40821073*255.0]
}

DS_TO_STD = {
    'CIFAR': [51.5865, 50.847 , 51.255],
    'CINIC': [0.24205776*255, 0.23828046*255, 0.25874835*255],
    'WATERBIRDS': [58.395, 57.12 , 57.375],
    'CELEBA': [58.395, 57.12 , 57.375],
    'YEARBOOK': [1.*255, 1.*255, 1.*255],
    'LIVING17': [0.229*255, 0.224*255, 0.225*255],
    'IMAGENET': [0.229*255, 0.224*255, 0.225*255],
    'ADE': [0.229*255, 0.224*255, 0.225*255],
    'IMAGENET_ADV': [0.229, 0.224, 0.225],
    'CHEXPERT': [0.229*255, 0.224*255, 0.225*255],
    'MIMIC': [0.229*255, 0.224*255, 0.225*255],
    'NIH': [0.229*255, 0.224*255, 0.225*255],
    'OBJECTNET20': [0.229*255, 0.224*255, 0.225*255],
    'OBJECTNET20_SAM': [1.*255, 1.*255, 1.*255], # DM bug
    'DUMMY': [255.,255.,255.],
    'ONES': [1.,1.,1.],
    'OPENAI_CLIP': [0.26862954*255, 0.26130258*255, 0.27577711*255]
}

INT_LABEL_PIPELINE = lambda device: [DC.IntDecoder(), TF.ToTensor(), TF.ToDevice(device), TF.Squeeze()]
FLOAT_LABEL_PIPELINE = lambda device: [DC.FloatDecoder(), TF.ToTensor(), TF.ToDevice(device), TF.Squeeze()]
NDARRAY_PIPELINE = lambda device: [DC.NDArrayDecoder(), TF.ToTensor(), TF.ToDevice(device), TF.Squeeze()]
NDARRAY_PIPELINE_V2 = lambda device: [DC.NDArrayDecoder(), TF.ToTensor(), TF.Convert(torch.float16), TF.ToDevice(device), TF.Squeeze()]

IMAGE_PIPELINES = {
    'cifar': {
        'train': lambda device: [ # ffcv
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            TF.Cutout(4, tuple(map(int, DS_TO_MEAN['CIFAR']))),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'train_wo_flip': lambda device: [ # patch+tint+binary
            get_image_decoder('simple', 32),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            TF.Cutout(4, tuple(map(int, DS_TO_MEAN['CIFAR']))),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'randomcrop_flip': lambda device: [ # patch+tint+binary
            get_image_decoder('random_resized_crop', 32),
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'flip': lambda device: [ # patch+tint+binary
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'train_alt': lambda device: [ # without cutout
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ],
        'test': lambda device: [
            get_image_decoder('simple', 32),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CIFAR'], DS_TO_STD['CIFAR'])
        ]
    },
    'cinic': {
        'train': lambda device: [ # ffcv
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            TF.Cutout(4, tuple(map(int, DS_TO_MEAN['CINIC']))),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['CINIC'], DS_TO_STD['CINIC'])
        ],
        'train_alt': lambda device: [ # without cutout
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CINIC'], DS_TO_STD['CINIC'])
        ],
        'test': lambda device: [
            get_image_decoder('simple', 32),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CINIC'], DS_TO_STD['CINIC'])
        ]
    },
    'waterbirds': {
        'train': lambda device: [ # sagawa numbers
            get_image_decoder('resized', 224),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ],
        'train_creager': lambda device: [ # +: center crop
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ],
        'train_heavyaug': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=14, fill=(0,0,0)),
            TF.Cutout(28, tuple(map(int, DS_TO_MEAN['WATERBIRDS']))),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ],
        'test': lambda device: [
            get_image_decoder('resized', 224),
            *get_standard_image_pipeline(device, DS_TO_MEAN['WATERBIRDS'], DS_TO_STD['WATERBIRDS'])
        ]
    },
    'celeba': {
        'train': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=1), # ratio = crop_size / min-side-length (178)
            *get_standard_image_pipeline(device, DS_TO_MEAN['CELEBA'], DS_TO_STD['CELEBA'])
        ],
        'train_flip': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=1), # ratio = crop_size / min-side-length (178)
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CELEBA'], DS_TO_STD['CELEBA'])
        ],
        'test_old': lambda device: [
            get_image_decoder('resized', 224),
            *get_standard_image_pipeline(device, DS_TO_MEAN['CELEBA'], DS_TO_STD['CELEBA'])
        ],
        'test': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=1), # ratio = crop_size / min-side-length (178)
            *get_standard_image_pipeline(device, DS_TO_MEAN['CELEBA'], DS_TO_STD['CELEBA'])
        ]
    },
    'yearbook': {
        'train_v2': lambda device: [ # ffcv
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            TF.RandomTranslate(padding=2, fill=(0,0,0)),
            TF.Cutout(4, tuple(map(int, DS_TO_MEAN['YEARBOOK']))),
            *get_standard_image_pipeline(device,  DS_TO_MEAN['YEARBOOK'], DS_TO_STD['YEARBOOK'])
        ],
        'train': lambda device: [ # without cutout
            get_image_decoder('simple', 32),
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device, DS_TO_MEAN['YEARBOOK'], DS_TO_STD['YEARBOOK'])
        ],
        'test': lambda device: [
            get_image_decoder('simple', 32),
            *get_standard_image_pipeline(device, DS_TO_MEAN['YEARBOOK'], DS_TO_STD['YEARBOOK'])
        ]
    },
    'living17': {
        'train_aug': lambda device: [
            get_image_decoder('random_resized_crop', 224),
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device, DS_TO_MEAN['LIVING17'], DS_TO_STD['LIVING17'])
        ],
        'train_no_aug': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            *get_standard_image_pipeline(device, DS_TO_MEAN['LIVING17'], DS_TO_STD['LIVING17'])
        ],
        'val': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            *get_standard_image_pipeline(device, DS_TO_MEAN['LIVING17'], DS_TO_STD['LIVING17'])
        ]
    },
    'imagenet': {
            'train_aug': lambda device: [
                get_image_decoder('random_resized_crop', 224),
                TF.RandomHorizontalFlip(),
                *get_standard_image_pipeline(device, DS_TO_MEAN['IMAGENET'], DS_TO_STD['IMAGENET'])
            ],
            'train_no_aug': lambda device: [
                get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
                *get_standard_image_pipeline(device, DS_TO_MEAN['IMAGENET'], DS_TO_STD['IMAGENET'])
            ],
            'val': lambda device: [
                get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
                *get_standard_image_pipeline(device, DS_TO_MEAN['IMAGENET'], DS_TO_STD['IMAGENET'])
            ],
            'val01': lambda device: [
                get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
                *get_standard_image_pipeline(device, DS_TO_MEAN['DUMMY'], DS_TO_STD['DUMMY'])
            ]
    },
    'ade': {
        'val': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=1.0),
            *get_standard_image_pipeline(device, DS_TO_MEAN['ADE'], DS_TO_STD['ADE'])
        ]
    },
    'imagenet_adv': {
            'val': lambda device: [
                DC.NDArrayDecoder(),
                TF.ToTensor(),
                TF.Convert(torch.float16),
                T.Normalize(DS_TO_MEAN['IMAGENET_ADV'], DS_TO_STD['IMAGENET_ADV']),
                TF.ToDevice(device),
                TF.Squeeze()
            ]
    },
    'objectnet20': {
        'train': lambda device: [
            get_image_decoder('random_resized_crop', 224),
            TF.RandomHorizontalFlip(),
            *get_standard_image_pipeline(device, DS_TO_MEAN['OBJECTNET20'], DS_TO_STD['OBJECTNET20'])
        ],
        'val': lambda device: [
            get_image_decoder('center_crop', 224, center_crop_ratio=224/256.),
            *get_standard_image_pipeline(device, DS_TO_MEAN['OBJECTNET20'], DS_TO_STD['OBJECTNET20'])
        ]
    },
    'openai_clip': {
        'val': lambda device: [ # imagenet and openai-PIL transformed cifar images
            get_image_decoder('center_crop', 224, center_crop_ratio=1),
            *get_standard_image_pipeline(device, DS_TO_MEAN['OPENAI_CLIP'], DS_TO_STD['OPENAI_CLIP'])
        ],
        'cifar': lambda device: [ # accuracy with this 10% less than vanilla dataloader + openaiclip pipeline
            get_image_decoder('simple', 32),
            TF.ToTensor(),
            TF.ToDevice(device, non_blocking=True),
            TF.Convert(torch.float32),
            TF.ToTorchImage(),
            T.Resize(224, interpolation=T.InterpolationMode.BICUBIC, max_size=None, antialias=False),
            T.Normalize(DS_TO_MEAN['OPENAI_CLIP'], DS_TO_STD['OPENAI_CLIP']),
        ],
        'cifar_np': lambda device: [ # for betons that save open ai transformed images to numpy array
            DC.NDArrayDecoder(),
            TF.ToTensor(),
            TF.Convert(torch.float16),
            TF.ToDevice(device),
            TF.Squeeze()
        ],
    }
}

def get_image_decoder(decoder_name, image_size, center_crop_ratio=224/256.,is_rgb=True):
    img_decoders = {
        'simple': lambda sz: DC.SimpleRGBImageDecoder(is_rgb=is_rgb),
        'resized': lambda sz: DC.CenterCropRGBImageDecoder((sz,sz),1,is_rgb=is_rgb),
        'center_crop': lambda sz,rt: DC.CenterCropRGBImageDecoder((sz,sz),rt,is_rgb=is_rgb),
        'random_resized_crop': lambda sz: DC.RandomResizedCropRGBImageDecoder((sz,sz),is_rgb=is_rgb),
    }

    assert decoder_name.lower() in img_decoders
    img_decoders['center_crop'] = partial(img_decoders['center_crop'], rt=center_crop_ratio)

    return img_decoders[decoder_name](image_size)

def get_standard_image_pipeline(device, mean, std):
    return [
        TF.ToTensor(),
        TF.ToDevice(device),
        TF.ToTorchImage(),
        TF.Convert(torch.float16),
        T.Normalize(mean, std)
    ]

def get_pipelines(dataset_name, aug_name, device):
    proxy_dset_map = {
        'mnist': 'mnist',
        'cinic': 'cinic',
        'cifar2': 'cifar',
        'cifar10': 'cifar',
        'cifar100': 'cifar',
        'indexed-cifar10': 'cifar',
        'indexed-cifar100': 'cifar',
        'waterbirds': 'waterbirds',
        'indexed-waterbirds': 'waterbirds',
        'living17': 'living17',
        'imagenet': 'imagenet',
        'imagenet_adv': 'imagenet_adv',
        'objectnet20': 'objectnet20',
        'yearbook': 'yearbook',
        'ade': 'ade',
        'openai_clip': 'openai_clip',
        'celeba': 'celeba'
    }

    dataset_name, aug_name = dataset_name.lower(), aug_name.lower()
    proxy_dataset_name = proxy_dset_map[dataset_name]

    assert proxy_dataset_name in IMAGE_PIPELINES
    assert aug_name in IMAGE_PIPELINES[proxy_dataset_name], 'aug_name: {}'.format(aug_name)

    img_pipeline = IMAGE_PIPELINES[proxy_dataset_name][aug_name]

    if dataset_name in ['cifar10', 'cifar100', 'cifar2', 'cinic',
                        'indexed-cifar10', 'indexed-cifar100']:
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device)
        }

        if 'indexed' in dataset_name:
            base['index'] = INT_LABEL_PIPELINE(device)
        return base

    elif dataset_name in ['waterbirds', 'indexed-waterbirds', 'celeba']:
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device),
            'group': INT_LABEL_PIPELINE(device)
        }

        if 'indexed' in dataset_name:
            base['index'] = INT_LABEL_PIPELINE(device)

        return base

    elif dataset_name in ['living17']:
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device),
            'orig_label': INT_LABEL_PIPELINE(device),
            'index': INT_LABEL_PIPELINE(device)
        }
        return base

    elif dataset_name in ['imagenet', 'imagenet_adv', 'ade', 'openai_clip']:
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device),
        }
        return base

    elif dataset_name in ['objectnet20']:
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device),
            'index': INT_LABEL_PIPELINE(device)
        }
        return base

    elif dataset_name in ['yearbook']:
        base = {
            'image': img_pipeline(device),
            'label': INT_LABEL_PIPELINE(device),
            'year': INT_LABEL_PIPELINE(device)
        }
        return base
    else:
        raise NotImplementedError