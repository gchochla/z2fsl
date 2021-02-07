"""Dataset."""

import os

import torch

from z2fsl.utils.conf import TRAIN_SPLIT, TEST_SPLIT
from z2fsl.utils.general import configuration_filename, slice_fn, dataset_name


class FeatureEpisodeFactory:
    """Factory of episodes/minibatches in a ZSL setting
    with extracted features instead of images.

    Attributes:
        attributes (`dict`): str-torch.Tensor key-value pairs
            where keys are "name" of classes and values are
            attribute vectors.
        dataset_dir (`str`): dataset root directory.
        feature_dir (`str`): feature directory wrt dataset_dir.
        way (`int`): number of classes to fetch from during
            an episode.
        queries_per (`int`): number of images per class to fetch
            as query images.
        id_mapping (`dict`): mapping between class ids and usable
            labels.
        n_classes (`int`): number of classes.
    """

    def __init__(
        self,
        dataset_dir,
        split='train',
        feature_dir='features',
        classes_fn='classes.txt',
        augmentation=False,
        inner_split='train',
        proposed_splits=True,
        l2_attr_norm=False,
        feature_norm=False,
        generalized=False,
        **kwargs
    ):
        """Init.

        Args:
            dataset_dir (`str`): root directory of dataset.
            split (`str`, optional): train, val or test split (also trainval),
                default=`train`.
            feature_dir (`str`, optional): feature directory w.r.t. dataset_dir,
                default=`images`.
            classes_fn (`str`, optional): txt with id and directory name for
                all classes wrt dataset_dir, default=`classes.txt`.
            augmentation (`bool`, optional): whether to use data augmentation
                (during training), default=`False`. If using the proposed
                features - where no data augmentation is used -, it should
                not be set to `True` (or simply not set).
            inner_split (`str`, optional): to be set if this dataset represents a
                train split to specify if train or test features from the
                split should be fetched in GZSL setting, options [`'train'`,
                `'test'`], default `'train'`.
            proposed_splits (`bool`, optional): whether to use proposed splits (PS)
                from Xian et al., 2018, default=`True`.
            generalized (`bool`, optional): whether testing GZSL, default `False`.
        """

        ds_name = dataset_name(dataset_dir)

        if ds_name == 'cub':
            attributes_fn = 'attributes/class_attribute_labels_continuous.txt'
        elif ds_name == 'awa2':
            attributes_fn = 'predicate-matrix-continuous.txt'
        elif ds_name == 'sun':
            attributes_fn = 'continuous-attributes.txt'

        if feature_norm:
            stat_split = kwargs.get('stat_split', split)
            stats_fn = configuration_filename(feature_dir, proposed_splits, stat_split, generalized)
            stats_fn = os.path.join(dataset_dir, stats_fn)
            minmax = torch.load(stats_fn)
            self.load = lambda fn: (torch.load(fn) - minmax[0]) / (minmax[1] - minmax[0])
        else:
            self.load = torch.load

        available_classes = os.path.join(
            'proposed_splits' if proposed_splits else '', split + 'classes.txt'
        )

        with open(os.path.join(dataset_dir, available_classes), 'r') as avcls:
            available_class_names = [line.strip() for line in avcls.readlines()]

        with open(os.path.join(dataset_dir, classes_fn), 'r') as cfp:
            class_names = [line.strip().split()[1] for line in cfp.readlines()]

        with open(os.path.join(dataset_dir, attributes_fn), 'r') as afp:
            attributes = [[float(val) for val in line.strip().split()] for line in afp.readlines()]

        # attributes and class names from classes.txt file in same order -> zip
        self.attributes = {
            class_name: torch.tensor(attribute)
            for class_name, attribute in zip(
                class_names, attributes
            )
            if class_name in available_class_names
        }
        if l2_attr_norm:
            self.attributes = {
                class_name: self.attributes[class_name] / self.attributes[class_name].norm()
                for class_name in self.attributes
            }

        ### get all available filenames

        filenames = {}
        for class_name in self.attributes:
            class_dir = os.path.join(dataset_dir, feature_dir, class_name)
            filenames[class_name] = os.listdir(class_dir)

        if generalized and split.startswith('train'):
            # if generalized and this is a train split
            # further restrict available files
            loc_fn = '{}{}_{}_loc.txt'.format(
                split, '_proposed_splits' if proposed_splits else '', inner_split
            )
            loc_fn = os.path.join(dataset_dir, loc_fn)
            if os.path.exists(loc_fn):  # if loc_fn exists, there are standard seen/unseen splits
                with open(loc_fn, 'r') as lfp:
                    available_files = [line.strip() for line in lfp.readlines()]
                # filter files not in standard splits
                for class_name in filenames:
                    filenames[class_name] = [
                        fn for fn in filenames[class_name] if fn in available_files
                    ]
            else:  # use arbitrary splits if standard ones are not present
                _slice = slice_fn(TEST_SPLIT if inner_split == 'test' else TRAIN_SPLIT)
                filenames = {class_name: _slice(filenames[class_name]) for class_name in filenames}

        self.filenames = filenames

        self.dataset_dir = dataset_dir
        self.feature_dir = feature_dir
        self.split = split
        self.augmentation = augmentation

        self.id_mapping = {real: usable for usable, real in enumerate(self.attributes)}
        self.n_classes = len(self.attributes)

    def __call__(self, way=None, queries_per=None):  # [:None] returns all
        """Creates an episode/minibatch.

        Args:
            way (`int`, optional): number of classes to fetch from during
                the episode.
            queries_per (`int`, optional): number of images per class to fetch
                as query images.

        Returns:
            A `list` of `torch.Tensor` features, a `list`
            of `torch.Tensor` attributes, of way classes and
            a `list` of IDs.
        """

        rand_inds = torch.randperm(len(self.attributes)).numpy()[:way]
        all_names = list(self.attributes)
        class_names = [all_names[i] for i in rand_inds]

        eps_attributes = [self.attributes[i] for i in class_names]

        eps_features = []
        class_ids = [self.id_mapping[class_name] for class_name in class_names]

        for class_name in class_names:
            class_features = []

            feature_filenames = self.filenames[class_name]
            rand_inds = torch.randperm(len(feature_filenames)).numpy()[:queries_per]
            feature_filenames = [feature_filenames[i] for i in rand_inds]

            for ft_fn in feature_filenames:
                if self.augmentation and self.split.startswith('train'):
                    crop = torch.randint(10, (1,)).item()
                else:
                    crop = 0  # center crop
                features = self.load(
                    os.path.join(self.dataset_dir, self.feature_dir, class_name, ft_fn)
                )[crop]
                class_features.append(features)

            eps_features.append(torch.stack(class_features))

        return eps_features, eps_attributes, class_ids

    def fsl_episode(self, way, shot, queries_per):
        """Fetches and FSL episode.

        Args:
            way (`int`): number of classes.
            shot (`int`): number of samples per class for
                support.
            queries_per (`int`): number of samples per class
                for querying.

        Returns:
            `list` of `way` `torch.Tensor`s with `shot` vectors each,
            `list` of `way` `torch.Tensor`s with `queries_per` vectors
            each and `list` of `way` `int`s with usable labels.
        """

        features, _, class_ids = self(way=way, queries_per=shot + queries_per)
        query = [feats[:queries_per] for feats in features]
        support = [feats[queries_per:] for feats in features]
        return support, query, class_ids
