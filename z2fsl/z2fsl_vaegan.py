"""Z2FSL(VAEGAN, PN) trainer class and script."""

import os
import argparse
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import ParameterGrid

from z2fsl.modules.classifiers import PrototypicalNet
from z2fsl.utils.losses import EpisodeCrossEntropyLoss
from z2fsl.utils.submodules import CVAE, ConcatMLP
from z2fsl.utils.datasets import FeatureEpisodeFactory
from z2fsl.utils.losses import kl_divergence, gradient_penalty
from z2fsl.utils.general import (
    str2bool,
    set_parameter_requires_grad,
    configuration_filename,
    dataset_name,
    manual_seed,
)


class Z2FSL_VAEGAN_Trainer:
    """Z2FSL(VAEGAN, PN) trainer class.

    Attributes:
        vae (`nn.Module`): CVAE.
        critic (`nn.Module`): Discriminator.
        fsl_classifier (`nn.Module`): Prototypical Net.
        episode_factory_kwargs (`dict`): episode factory keywords.
        train_split (`str`): which split to train on (basically
            `'train'` or `'trainval'`).
        eval_split (`str`): which split to evaluate on (basically
            `'val'` or `'test'`).
        generalized (`bool`): whether to train and evaluate on GZSL.
        way (`int`): way of training episodes. 
        shot (`int`): shot of training episodes (provided by generator).
        queries_per (`int`): number of samples per class in query set.
        device (`str`): device to run on.
        noise_dim (`int`): dimension of generator noise / VAE's latent dim.
        lambda_z2fsl (`float`): FSL classifier loss coefficient
            (\gamma in paper).
        lambda_wgan (`float`): WGAN loss coefficient (\\beta in paper).
        lambda_vae (`float`): VAE loss coeffcient (basically used
            to turn off the VAE).
        lambda_pen (`float`): WGAN regularization term coefficient.
        train_fsl (`bool`): whether to train the FSL classifier jointly
            with the generator.
    """

    def __init__(
        self,
        dataset_dir,
        feature_dir='features',
        proposed_splits=True,
        l2_attr_norm=True,
        augmentation=True,
        feature_norm=True,
        train_split='train',
        eval_split='val',
        generalized=False,
        feature_dim=2048,
        metadata_dim=312,
        noise_dim=None,
        generator_hidden_layers=None,
        classifier_hidden_layers=None,
        embedding_dim=128,
        critic_hidden_layers=None,
        way=20,
        shot=5,
        queries_per=10,
        pretrained_fsl_dir='',
        lambda_z2fsl=1e2,
        lambda_pen=10,
        lambda_wgan=1e2,
        lambda_vae=1,
        device='cuda:0',
        **kwargs
    ):
        """Init.

        Args:
            dataset_dir (`str`): root dir of dataset.
            feature_dir (`str`, optional): directory of features relative to `dataset_dir`.
            proposed_splits (`bool`, optional): whether to use proposed splits,
                default `True`.
            l2_attr_norm (`bool`, optional): whether to L2 normalize attributes,
                default `True`.
            augmentation (`bool`, optional): whether ot use augmentation, default `True`. 
            feature_norm (`bool`, optional): whether to minmax normalize features,
                default `True`.
            train_split (`str`, optional): which split to train on (basically
                `'train'` or `'trainval'`), default `'train'`.
            eval_split (`str`, optional): which split to evaluate on (basically
                `'val'` or `'test'`), default `'val'`.
            generalized (`bool`, optional): whether to train and evaluate on GZSL,
                default `False`.
            feature_dim (`int`, optional): dimension of input features, default `2048`.
            metadata_dim (`int`, optional): dimension of auxiliary descriptions.
            noise_dim (`int`, optional): dimension of generator's noise dimension,
                default is same as `metadata_dim`.
            generator_hidden_layers (`int` | `list` of `int`s | `None`, optional):
                generator hidden layers, default `None`.
            classifier_hidden_layers (`int` | `list` of `int`s | `None`, optional):
                classifier hidden layers, default `None`.
            embedding_dim (`int`, optional): dimension of PN output, default `128`.
            critic_hidden_layers (`int` | `list` of `int`s | `None`, optional):
                discriminator hidden layers, default `None`.
            way (`int`, optional): way of training episodes, default `20`.
            shot (`int`, optional): shot of training episodes (provided by generator),
                default `5`. 
            queries_per (`int`, optional): number of samples per class in query set,
                default `10`.
            pretrained_fsl_dir (`str`, optional): directory where pre-trained FSL
                classifier is stored, default `''` (no pretrained classifier,
                classifiers are names a certain way based on hyperparameters
                and benchmarks, so script picks them up automatically).
            lambda_z2fsl (`float`, optional): FSL classifier loss coefficient
            (\gamma in paper), default `100`.
            lambda_pen (`float`, optional): WGAN regularization term coefficient,
                default `10`.
            lambda_wgan (`float`, optional): WGAN loss coefficient (\\beta in paper),
                default `100`.
            lambda_vae (`float`, optional): VAE loss coeffcient (basically used
            to turn off the VAE), default `1`.
            device (`str`, optional): device to run on, default `'cuda:0'`.
            `train_fsl` (`bool`, optional): whether to train FSL classifier.
        """

        if noise_dim is None:
            noise_dim = metadata_dim

        self.vae = CVAE(
            feature_dim=feature_dim,
            latent_dim=noise_dim,
            cond_dim=metadata_dim,
            hidden_layers=generator_hidden_layers,
        ).to(device)

        input_dim = metadata_dim + feature_dim
        self.critic = ConcatMLP(
            in_features=input_dim,
            out_features=1,
            hidden_layers=critic_hidden_layers,
            output_actf=nn.Identity(),
        ).to(device)

        self.episode_factory_kwargs = dict(
            dataset_dir=dataset_dir,
            feature_dir=feature_dir,
            proposed_splits=proposed_splits,
            l2_attr_norm=l2_attr_norm,
            augmentation=augmentation,
            feature_norm=feature_norm,
        )
        self.train_split = train_split
        self.eval_split = eval_split
        self.generalized = generalized
        self.way = way
        self.shot = shot
        self.queries_per = queries_per

        self.device = device
        self.noise_dim = noise_dim
        self.lambda_z2fsl = lambda_z2fsl
        self.lambda_wgan = lambda_wgan
        self.lambda_vae = lambda_vae
        self.lambda_pen = lambda_pen

        self.fsl_classifier = PrototypicalNet(
            in_features=feature_dim,
            out_features=embedding_dim,
            hidden_layers=classifier_hidden_layers,
        )
        self.train_fsl = lambda_z2fsl > 0
        if pretrained_fsl_dir:
            ds_name = dataset_name(dataset_dir)
            fsl_fn = 'fsl_{}{}_{}'.format(
                ds_name,
                '_noaug' if not augmentation else '',
                configuration_filename(feature_dir, proposed_splits, train_split, generalized),
            )
            fsl_fn = os.path.join(pretrained_fsl_dir, fsl_fn)
            state_dict = torch.load(fsl_fn)
            self.fsl_classifier.load_state_dict(state_dict)
            self.train_fsl = kwargs.get('train_fsl', self.train_fsl)
            if not self.train_fsl:
                set_parameter_requires_grad(self.fsl_classifier, False)
        self.fsl_classifier.to(device)

    def train(
        self,
        episodes,
        gen_lr,
        gen_beta1=0.5,
        fsl_lr=None,
        fsl_beta1=None,
        val_epochs=25,
        val_samples=350,
        train_val_samples=None,
        val_lr=1e-3,
        val_beta1=0.5,
        print_interval=25,
        eval_interval=None,
        n_fsl=1,
        n_critic=5,
        mix=False,
        real_support=True,
        softmax=True,
    ):
        """Trains all components (FSL classifier, generator, encoder, discriminator)
        and provides evaluation metrics.

        Args:
            episodes (`int`): number of training iterations.
            gen_lr (`float`): generator (and all backbone VAEGAN components)
                learning rate.
            gen_beta1 (`float`): beta1 of Adam optimizer for VAEGAN backbone,
                default `0.5`.
            fsl_lr (`float` | `None`, optional): FSL classifier's learning rate,
                default is the same as `gen_lr`.
            fsl_beta1 (`float` | `None`, optional): FSL classifier's beta1 for Adam,
                default is the same as `gen_beta1`.
            val_epochs (`int`, optional): epochs of finetuning the FSL classifier on
                unseen classes or linear classifier's training, default `25`.
            val_samples (`int`, optional): number of samples to produce per unseen class
                during evaluation, default `350`.
            train_val_samples (`list` of `int`s | `None`, optional): number of samples to
                produce per seen class during evaluation, default is same as
                `val_samples`.
            val_lr (`float`, optional): learning rate of linear classifier, default `1e-3`.
            val_beta1 (`float`, optional): beta1 of linear classifier's Adam, default `0.5`.
            print_interval (`int`, optional): episodes to wait before printing several training
                losses, default `25`.
            eval_interval (`int` | `None`, optional): episodes to wait before printing evaluation
                metrics, default is at the end of training.
            n_fsl (`int`, optional): number of FSL classifier updates per training iteration,
                default `1`.
            n_critic (`int`, optional): number of FSL classifier updates per training iteration,
                default `5`.
            mix (`bool`, optional): mix support and query sets during training, default `False`.
            real_support (`bool`, optional): if GZSL, whether to use real or synthetic support,
                default `True`.
            softmax (`bool`, optional): whether to evaluate with a linear classifier as well,
                default `True`.

        Returns:
            `dict` of `dict`, indexed by the number of training iterations, that includes
            evaluation method `str`: accuracy key-value pairs, e.g.

            {
                1000: {
                    'normal': 0.45,  # if shot remains the same as training
                    'more-shots': 0.58,
                    'finetune': 0.59,
                    'softmax': 0.55,
                },
                2000: {
                    'normal': 0.48,
                    'more-shots': 0.61,
                    'finetune': 0.62,
                    'softmax': 0.56,
                }
            }
        """

        if eval_interval is None:
            eval_interval = episodes

        if fsl_lr is None:
            fsl_lr = gen_lr
        if fsl_beta1 is None:
            fsl_beta1 = gen_beta1

        evals = {}
        optimizer = optim.Adam(self.vae.parameters(), lr=gen_lr, betas=(gen_beta1, 0.999))
        c_optimizer = optim.Adam(self.critic.parameters(), lr=gen_lr, betas=(gen_beta1, 0.999))

        if self.train_fsl:
            fsl_optimizer = optim.Adam(
                self.fsl_classifier.parameters(), lr=fsl_lr, betas=(fsl_beta1, 0.999)
            )

        episode_factory = FeatureEpisodeFactory(
            split=self.train_split,
            inner_split='train',
            generalized=self.generalized,
            **self.episode_factory_kwargs
        )
        criterion = EpisodeCrossEntropyLoss()
        _bce = nn.BCELoss(reduction='sum')
        bce = lambda pred, tar: _bce(pred, tar) / pred.size(0)

        for episode in range(episodes):

            for i in range(n_critic + n_fsl + 1):

                features, attributes, _ = episode_factory(
                    queries_per=self.queries_per, way=self.way
                )

                if i < n_critic:

                    #######################
                    ### critic training ###

                    features = torch.cat(features)
                    attributes = torch.stack(attributes)
                    attributes = attributes.repeat_interleave(self.queries_per, dim=0)

                    features, attributes = features.to(self.device), attributes.to(self.device)

                    noise = torch.randn(attributes.size(0), self.noise_dim, device=self.device)

                    with torch.no_grad():
                        gen_features = self.vae.decode(attributes, noise)

                    gen_critic_outs = self.critic(gen_features, attributes)
                    real_critic_outs = self.critic(features, attributes)
                    grad_pen = gradient_penalty(self.critic, features, gen_features, attributes)

                    c_loss = (
                        gen_critic_outs.mean(dim=0)
                        - real_critic_outs.mean(dim=0)
                        + self.lambda_pen * grad_pen
                    )

                    c_optimizer.zero_grad()
                    c_loss.backward()
                    c_optimizer.step()

                    ### critic training ###
                    #######################

                elif n_critic <= i < n_critic + n_fsl and self.train_fsl:

                    #########################
                    ### fsl algo training ###

                    features = [class_features.to(self.device) for class_features in features]
                    support = []
                    for class_attributes in attributes:
                        class_attributes = class_attributes.repeat(self.shot, 1).to(self.device)
                        noise = torch.randn(self.shot, self.noise_dim, device=self.device)
                        with torch.no_grad():
                            support.append(self.vae.decode(class_attributes, noise))

                    logits = self.fsl_classifier(support, features, mix=mix)
                    clsf_loss = criterion(logits)

                    fsl_optimizer.zero_grad()
                    clsf_loss.backward()
                    fsl_optimizer.step()

                    ### fsl algo training ###
                    #########################

                else:

                    #######################
                    ### vaegan training ###

                    ### classification

                    features = [class_features.to(self.device) for class_features in features]
                    support = []
                    for class_attributes in attributes:
                        class_attributes = class_attributes.repeat(self.shot, 1).to(self.device)
                        noise = torch.randn(self.shot, self.noise_dim, device=self.device)
                        support.append(self.vae.decode(class_attributes, noise))

                    if self.lambda_z2fsl > 0:
                        logits = self.fsl_classifier(support, features, mix=mix)
                        clsf_loss = criterion(logits)
                    else:
                        clsf_loss = torch.tensor(0.0)

                    ### vae

                    attributes = torch.stack(attributes).to(self.device)

                    if self.lambda_vae > 0:

                        features = torch.cat(features)

                        rec_features, mean, logvar = self.vae(
                            features, attributes.repeat_interleave(self.queries_per, dim=0)
                        )
                        bce_loss = bce(rec_features, features)
                        kl_loss = kl_divergence(mean, logvar)
                        vae_loss = bce_loss + kl_loss

                    else:
                        vae_loss = torch.tensor(0.0)

                    ### wgan

                    if self.lambda_wgan > 0:

                        gen_features = torch.cat(support)
                        critic_outs = self.critic(
                            gen_features, attributes.repeat_interleave(self.shot, dim=0)
                        )
                        wgan_loss = -critic_outs.mean(dim=0)

                    else:
                        wgan_loss = torch.tensor(0.0)

                    if torch.isnan(wgan_loss).any():
                        print('WGAN nan, episode {}'.format(episode))
                    if torch.isnan(vae_loss).any():
                        print('VAE nan, episode {}'.format(episode))
                        if torch.isnan(bce_loss).any():
                            print('BCE')
                        if torch.isnan(kl_loss).any():
                            print('KL')
                    if torch.isnan(clsf_loss).any():
                        print('Classification nan, episode {}'.format(episode))

                    loss = (
                        self.lambda_vae * vae_loss
                        + self.lambda_wgan * wgan_loss
                        + self.lambda_z2fsl * clsf_loss
                    )

                    if torch.isnan(loss):
                        break

                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_value_(self.vae.parameters(), 5)
                    optimizer.step()

                    ### vaegan training ###
                    #######################

                    break  # to avoid repeating if train_fsl is false

            if torch.isnan(loss):
                self.vae.to('cpu')
                dire = 'models'
                if not os.path.exists(dire):
                    os.makedirs(dire)
                torch.save(self.vae.to('cpu').state_dict(), os.path.join(dire, 'vae.pt'))
                break

            if episode == 0 or (episode + 1) % print_interval == 0:

                ##########################
                ### renew measurements ###

                features, attributes, _ = episode_factory(
                    queries_per=self.queries_per, way=self.way
                )

                features = [class_features.to(self.device) for class_features in features]
                support = []
                for class_attributes in attributes:
                    class_attributes = class_attributes.repeat(self.shot, 1).to(self.device)
                    noise = torch.randn(self.shot, self.noise_dim, device=self.device)
                    with torch.no_grad():
                        support.append(self.vae.decode(class_attributes, noise))

                # get classification accuracy
                with torch.no_grad():
                    logits = self.fsl_classifier(support, features)

                c_loss = criterion(logits)

                accs = []
                for i, class_logits in enumerate(logits):
                    preds = class_logits.cpu().argmax(dim=-1)
                    corr = (preds == i).sum().item()
                    tot = preds.size(0)
                    accs.append(corr / tot)
                acc = sum(accs) / len(accs)

                # get vae metrics
                real_features = torch.cat(features)
                attributes = torch.stack(attributes).to(self.device)
                with torch.no_grad():
                    rec_features, mean, logvar = self.vae(
                        real_features, attributes.repeat_interleave(self.queries_per, dim=0)
                    )
                bce_loss = bce(rec_features, real_features)
                kl_loss = kl_divergence(mean, logvar)

                # get wasserstein distance proxy
                gen_features = torch.cat(support)
                with torch.no_grad():
                    gen_critic_outs = self.critic(
                        gen_features, attributes.repeat_interleave(self.shot, dim=0)
                    )
                    real_critic_outs = self.critic(
                        real_features, attributes.repeat_interleave(self.queries_per, dim=0)
                    )
                w_dist = real_critic_outs.mean(dim=0) - gen_critic_outs.mean(dim=0)

                ### renew measurements ###
                ##########################

                epoch_msg = (
                    'Episode {} - Rec loss = {:.3f} KL = {:.3f} W dist = {:.3f} '
                    'Clsf loss = {:.3f} Clsf acc = {:.1f}%'
                ).format(
                    episode + 1,
                    bce_loss.item(),
                    kl_loss.item(),
                    w_dist.item(),
                    c_loss.item(),
                    acc * 100,
                )
                print(epoch_msg)

            if (episode + 1) % eval_interval == 0:

                evals[episode + 1] = {}

                if self.generalized:
                    if train_val_samples is None:
                        train_val_samples = [val_samples]

                    acc = self.eval()
                    acc_ms = {}
                    for train_samples in train_val_samples:
                        acc_ms[train_samples] = self.eval(
                            shot=val_samples,
                            train_shot=train_samples,
                            real_support=real_support,
                        )
                    max_train_samples = max(acc_ms.keys(), key=(lambda key: acc_ms[key]))

                    acc_ms_finetune = self.eval(
                        shot=val_samples,
                        train_shot=max_train_samples,
                        episodes=val_epochs,
                        fsl_lr=fsl_lr,
                        fsl_beta1=fsl_beta1,
                        real_support=real_support,
                    )

                    accs = {
                        'normal': acc,
                        'more-shots-{}'.format(max_train_samples): acc_ms[max_train_samples],
                        'finetune': acc_ms_finetune,
                    }

                    if softmax:

                        softmax_acc = self.eval_softmax(
                            epochs=val_epochs, lr=val_lr, beta1=val_beta1, per_class=val_samples
                        )
                        accs['softmax'] = softmax_acc

                else:
                    acc = self.eval()
                    acc_ms = self.eval(shot=val_samples)
                    acc_ms_finetune = self.eval(
                        shot=val_samples, episodes=val_epochs, fsl_lr=fsl_lr, fsl_beta1=fsl_beta1
                    )

                    accs = {
                        'normal': acc,
                        'more-shots': acc_ms,
                        'finetune': acc_ms_finetune,
                    }

                    if softmax:
                        softmax_acc = self.eval_softmax(
                            epochs=val_epochs, lr=val_lr, beta1=val_beta1, per_class=val_samples
                        )
                        softmax_mapped_acc = self.eval_softmax(
                            epochs=val_epochs,
                            lr=val_lr,
                            beta1=val_beta1,
                            per_class=val_samples,
                            use_mapper=True,
                        )
                        accs['softmax'] = softmax_acc
                        accs['mapped-softmax'] = softmax_mapped_acc

                print('Evaluation: ', end='')
                print(
                    ' '.join(
                        [
                            '{} {}'.format(
                                key.title().replace('-', ' '),
                                ' '.join(['{:.1f}%'.format(acc * 100) for acc in accs[key]]),
                            )
                            for key in accs
                        ]
                    )
                )

                evals[episode + 1] = accs

        return evals

    def eval(
        self, episodes=0, shot=None, train_shot=None, fsl_lr=5e-5, fsl_beta1=0.5, real_support=True
    ):
        """Evaluates generator in the desired ZSL setting
        within the Z2FSL framework.

        Args:
            episodes (`int`, optional): number of episodes to finetune FSL classifier,
                default `0`.
            shot (`int` | `None`, optional): shot in test episode of unseen classes,
                default is same as training shot.
            train_shot (`int` | `None`, optional): shot in test episode of seen classes,
                default is same as `shot`.
            fsl_lr (`float`, optional): learning rate of FSL classifier, default `5e-5`.
            fsl_beta1 (`float`, optional): beta1 of FSL classifier, default `0.5`.
            real_support (`bool`, optional): whether to use real support in GZSL.

        Returns:
            `float` accuracy.
        """

        self.vae.eval()
        set_parameter_requires_grad(self.vae, False)

        if shot is None:
            shot = self.shot
        if train_shot is None:
            train_shot = shot

        eval_episode_factory = FeatureEpisodeFactory(
            split=self.eval_split,
            stat_split=self.train_split,
            dataset_dir=self.episode_factory_kwargs['dataset_dir'],
            feature_dir=self.episode_factory_kwargs['feature_dir'],
            proposed_splits=self.episode_factory_kwargs['proposed_splits'],
            l2_attr_norm=self.episode_factory_kwargs['l2_attr_norm'],
            feature_norm=self.episode_factory_kwargs['feature_norm'],
            augmentation=False,
        )
        features, attributes, _ = eval_episode_factory()

        if self.generalized:
            train_episode_factory = FeatureEpisodeFactory(
                split=self.train_split,
                inner_split='test',
                dataset_dir=self.episode_factory_kwargs['dataset_dir'],
                feature_dir=self.episode_factory_kwargs['feature_dir'],
                proposed_splits=self.episode_factory_kwargs['proposed_splits'],
                l2_attr_norm=self.episode_factory_kwargs['l2_attr_norm'],
                feature_norm=self.episode_factory_kwargs['feature_norm'],
                augmentation=False,
                generalized=True,
            )
            seen_test_features, train_attributes, train_ids = train_episode_factory()

            if real_support:
                episode_factory = FeatureEpisodeFactory(
                    split=self.train_split,
                    inner_split='train',
                    generalized=True,
                    augmentation=False,
                    dataset_dir=self.episode_factory_kwargs['dataset_dir'],
                    feature_dir=self.episode_factory_kwargs['feature_dir'],
                    proposed_splits=self.episode_factory_kwargs['proposed_splits'],
                    l2_attr_norm=self.episode_factory_kwargs['l2_attr_norm'],
                    feature_norm=self.episode_factory_kwargs['feature_norm'],
                )
                seen_train_features, _, train_ids_perm = episode_factory()
                # align train and test seen-classes features
                seen_train_features = [
                    seen_train_features[train_ids_perm.index(tid)][:train_shot] for tid in train_ids
                ]

        fsl_classifier = deepcopy(self.fsl_classifier)  # copy to finetune
        set_parameter_requires_grad(fsl_classifier, True)

        ###########################
        ### finetuning fsl algo ###

        fsl_optimizer = optim.Adam(fsl_classifier.parameters(), lr=fsl_lr, betas=(fsl_beta1, 0.999))
        criterion = EpisodeCrossEntropyLoss()

        for _ in range(episodes):
            rand_inds = torch.randperm(len(attributes)).numpy()[: self.way]
            support = []
            query = []
            for i in rand_inds:
                class_attributes = attributes[i].repeat(self.shot, 1).to(self.device)
                noise = torch.randn(self.shot, self.noise_dim, device=self.device)
                support.append(self.vae.decode(class_attributes, noise))

                class_attributes = attributes[i].repeat(self.queries_per, 1).to(self.device)
                noise = torch.randn(self.queries_per, self.noise_dim, device=self.device)
                query.append(self.vae.decode(class_attributes, noise))

            logits = fsl_classifier(support, query)
            clsf_loss = criterion(logits)

            fsl_optimizer.zero_grad()
            clsf_loss.backward()
            fsl_optimizer.step()

        ### finetuning fsl algo ###
        ###########################

        fsl_classifier.eval()
        set_parameter_requires_grad(fsl_classifier, False)

        support = []
        for class_attributes in attributes:
            class_attributes = class_attributes.repeat(shot, 1).to(self.device)
            noise = torch.randn(shot, self.noise_dim, device=self.device)
            support.append(self.vae.decode(class_attributes, noise))

        if self.generalized:
            features.extend(seen_test_features)
            if real_support:
                seen_train_features = [
                    class_features.to(self.device) for class_features in seen_train_features
                ]
                support.extend(seen_train_features)
            else:
                for class_attributes in train_attributes:
                    class_attributes = class_attributes.repeat(train_shot, 1).to(self.device)
                    noise = torch.randn(train_shot, self.noise_dim, device=self.device)
                    support.append(self.vae.decode(class_attributes, noise))

        features = [class_features.to(self.device) for class_features in features]

        logits = fsl_classifier(support, features)

        accs = []
        for i, class_logits in enumerate(logits):
            preds = class_logits.cpu().argmax(dim=-1)
            corr = (preds == i).sum().item()
            tot = preds.size(0)
            accs.append(corr / tot)

        self.vae.train()
        set_parameter_requires_grad(self.vae, True)

        if self.generalized:
            split_dim = len(features) - len(seen_test_features)
            acc_u = sum(accs[:split_dim]) / len(accs[:split_dim])
            acc_s = sum(accs[split_dim:]) / len(accs[split_dim:])
            acc = 2 * acc_s * acc_u / (acc_s + acc_u)
            return [acc, acc_s, acc_u]

        return [sum(accs) / len(accs)]

    def eval_softmax(self, epochs, lr, beta1, per_class, weight_decay=0, use_mapper=False):
        """Evaluates generator in the desired ZSL setting
        within the Z2FSL framework.

        Args:
            epochs (`int`): number of epochs to finetune FSL classifier.
            lr (`float`): learning rate of linear classifier.
            beta1 (`float`): beta1 of linear classifier, default `0.5`.
            per_class (`int`): number of generations for unseen classes.
            weight_decay (`float`, optional): weight decay (L2 regularization), default `0`.
            use_mapper (`bool`, optional): use FSL classifier's net to map features to
                its space, default `False`.

        Returns:
            `float` accuracy.
        """

        if use_mapper:
            mapper = deepcopy(self.fsl_classifier.mapper)
            mapper.eval()
            set_parameter_requires_grad(mapper, False)
        else:
            mapper = nn.Identity().to(self.device)

        self.vae.eval()
        set_parameter_requires_grad(self.vae, False)

        eval_episode_factory = FeatureEpisodeFactory(
            split=self.eval_split,
            stat_split=self.train_split,
            dataset_dir=self.episode_factory_kwargs['dataset_dir'],
            feature_dir=self.episode_factory_kwargs['feature_dir'],
            proposed_splits=self.episode_factory_kwargs['proposed_splits'],
            l2_attr_norm=self.episode_factory_kwargs['l2_attr_norm'],
            feature_norm=self.episode_factory_kwargs['feature_norm'],
            augmentation=False,
            generalized=self.generalized,
        )
        features, attributes, class_ids = eval_episode_factory()

        if self.generalized:
            train_episode_factory = FeatureEpisodeFactory(
                split=self.train_split,
                inner_split='test',
                dataset_dir=self.episode_factory_kwargs['dataset_dir'],
                feature_dir=self.episode_factory_kwargs['feature_dir'],
                proposed_splits=self.episode_factory_kwargs['proposed_splits'],
                l2_attr_norm=self.episode_factory_kwargs['l2_attr_norm'],
                feature_norm=self.episode_factory_kwargs['feature_norm'],
                augmentation=False,
                generalized=True,
            )
            seen_test_features, _, train_class_ids = train_episode_factory()
            train_class_ids = [cid + len(class_ids) for cid in train_class_ids]

            episode_factory = FeatureEpisodeFactory(
                split=self.train_split,
                inner_split='train',
                dataset_dir=self.episode_factory_kwargs['dataset_dir'],
                feature_dir=self.episode_factory_kwargs['feature_dir'],
                proposed_splits=self.episode_factory_kwargs['proposed_splits'],
                l2_attr_norm=self.episode_factory_kwargs['l2_attr_norm'],
                feature_norm=self.episode_factory_kwargs['feature_norm'],
                augmentation=False,
                generalized=True,
            )

            seen_train_features, _, train_class_ids_perm = episode_factory()

            train_class_ids_perm = [cid + len(class_ids) for cid in train_class_ids_perm]

            # align train and test seen-classes features
            seen_train_features = [
                seen_train_features[train_class_ids_perm.index(tid)] for tid in train_class_ids
            ]

            features.extend(seen_test_features)
            # class_ids.extend(train_class_ids)

        attributes = torch.stack(attributes).to(self.device)
        attributes = attributes.repeat_interleave(per_class, dim=0)
        noise = torch.randn(attributes.size(0), self.noise_dim, device=self.device)
        gen_features = self.vae.decode(attributes, noise)
        gen_features = mapper(gen_features)
        gen_ids = torch.tensor(class_ids).repeat_interleave(per_class, dim=0)

        train_features = gen_features.to('cpu')
        train_ids = gen_ids

        n_classes = eval_episode_factory.n_classes
        if self.generalized:
            n_classes += train_episode_factory.n_classes

            seen_ids = torch.cat(
                [
                    torch.tensor([sid]).repeat(feats.size(0))
                    for sid, feats in zip(train_class_ids, seen_train_features)
                ]
            )
            train_ids = torch.cat([train_ids, seen_ids])

            seen_train_features = mapper(torch.cat(seen_train_features).to(self.device)).to('cpu')

            train_features = torch.cat([train_features, seen_train_features])

        train_dl = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(train_features, train_ids),
            batch_size=128,
            shuffle=True,
        )
        classifier = nn.Linear(train_features.size(1), n_classes).to(self.device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            classifier.parameters(), lr=lr, weight_decay=weight_decay, betas=(beta1, 0.999)
        )

        for _ in range(epochs):
            for feats, lbls in train_dl:
                feats, lbls = feats.to(self.device), lbls.to(self.device)
                logits = classifier(feats)
                loss = criterion(logits, lbls)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        classifier.eval()
        set_parameter_requires_grad(classifier, False)

        test_ids = []
        test_features = []

        if self.generalized:
            class_ids.extend(train_class_ids)

        for feats, cids in zip(features, class_ids):
            test_ids.append(torch.tensor([cids] * feats.size(0)))
            test_features.append(mapper(feats.to(self.device)).to('cpu'))

        accs = []
        for feats, lbls in zip(test_features, test_ids):
            feats, lbls = feats.to(self.device), lbls.to(self.device)
            preds = classifier(feats).argmax(dim=-1)
            accs.append((preds == lbls).sum().item() / feats.size(0))

        self.vae.train()
        set_parameter_requires_grad(self.vae, True)

        if self.generalized:
            split_dim = len(features) - len(seen_test_features)
            acc_u = sum(accs[:split_dim]) / len(accs[:split_dim])
            acc_s = sum(accs[split_dim:]) / len(accs[split_dim:])
            acc = 2 * acc_s * acc_u / (acc_s + acc_u)
            return [acc, acc_s, acc_u]

        return [sum(accs) / len(accs)]


def test_setting(
    generalized,
    pretrained_fsl,
    pretrained_dir,
    dataset_dir,
    feature_dir,
    proposed_splits,
    l2_attr_norm,
    feature_norm,
    augmentation,
    gen_learning_rates,
    fsl_learning_rates,
    metadata_dim,
    gen_beta1s,
    fsl_beta1s,
    generator_hidden_layers,
    critic_hidden_layers,
    classifier_hidden_layers,
    embedding_dims,
    mix,
    ways,
    shots,
    queries_pers,
    lambda_z2fsl,
    lambda_wgan,
    lambda_vae,
    device,
    val_epochs,
    val_samples,
    train_val_samples,
    episodes,
    eval_interval,
    train_fsl,
    real_support,
    testing,
    val_lr=1e-3,
    val_beta1=0.5,
    log_fn=None,
    grid_search=True,
    softmax=True,
):
    """Performs (optional) grid search and (optional) evaluation.
    For description of args, see `Z2FSL_VAEGAN_Trainer` and its methods.
    `pretrained_fsl`, `generalized`, `gen_learning_rates`, `gen_beta1s`,
    `fsl_learning_rates`, `fsl_beta1s`, `generator_hidden_layers`,
    `classifier_hidden_layers`, `critic_hidden_layers`, `lambda_z2fsl`,
    `l2_attr_norm`, `proposed_splits`, `augmentation`, `ways`, `shots`, `queries_pers`,
    `embedding_dims`, `train_fsl`, `mix`, `lambda_wgan`, `lambda_vae`, `real_support`
    should be lists so that the grid search can search through."""

    if log_fn is not None:
        dire = os.path.split(log_fn)[0]
        if dire and not os.path.exists(dire):
            os.makedirs(dire)

    search_params = dict(
        pretrained_fsl=pretrained_fsl,
        generalized=generalized,
        gen_lr=gen_learning_rates,
        gen_beta1=gen_beta1s,
        fsl_lr=fsl_learning_rates,
        fsl_beta1=fsl_beta1s,
        generator_hidden_layers=generator_hidden_layers,
        classifier_hidden_layers=classifier_hidden_layers,
        critic_hidden_layers=critic_hidden_layers,
        lambda_z2fsl=lambda_z2fsl,
        l2_attr_norm=l2_attr_norm,
        proposed_splits=proposed_splits,
        augmentation=augmentation,
        way=ways,
        shot=shots,
        queries_per=queries_pers,
        embedding_dim=embedding_dims,
        train_fsl=train_fsl,
        mix=mix,
        feature_norm=[feature_norm],
        lambda_wgan=lambda_wgan,
        lambda_vae=lambda_vae,
        real_support=real_support,
    )

    max_accs = dict()
    best_eps = dict()
    best_params = dict()
    best_train_samples = train_val_samples[0]  # only for more-shots, no need for dict

    param_grid = ParameterGrid(search_params)
    if grid_search or len(param_grid) > 1:
        for i, params in enumerate(param_grid):
            init_msg = 'Running {}/{} {}'.format(i + 1, len(param_grid), params)
            print('#' * len(init_msg) + '\n' + init_msg + '\n')

            n_critic = 5 if params['lambda_wgan'] > 0 else 0

            trainer = Z2FSL_VAEGAN_Trainer(
                train_split='train',
                eval_split='val',
                metadata_dim=metadata_dim,
                generator_hidden_layers=params['generator_hidden_layers'],
                classifier_hidden_layers=params['classifier_hidden_layers'],
                critic_hidden_layers=params['critic_hidden_layers'],
                embedding_dim=params['embedding_dim'],
                lambda_wgan=params['lambda_wgan'],
                way=params['way'],
                shot=params['shot'],
                queries_per=params['queries_per'],
                pretrained_fsl_dir=pretrained_dir if params['pretrained_fsl'] else '',
                lambda_z2fsl=params['lambda_z2fsl'],
                lambda_vae=params['lambda_vae'],
                dataset_dir=dataset_dir,
                feature_dir=feature_dir,
                feature_norm=feature_norm,
                proposed_splits=params['proposed_splits'],
                l2_attr_norm=params['l2_attr_norm'],
                device=device,
                train_fsl=params['train_fsl'],
                augmentation=params['augmentation'],
                generalized=params['generalized'],
            )

            evals = trainer.train(
                episodes=episodes,
                eval_interval=eval_interval,
                print_interval=50,
                gen_lr=params['gen_lr'],
                gen_beta1=params['gen_beta1'],
                fsl_lr=params['fsl_lr'],
                fsl_beta1=params['fsl_beta1'],
                val_epochs=val_epochs,
                val_samples=val_samples,
                train_val_samples=train_val_samples,
                val_beta1=val_beta1,
                val_lr=val_lr,
                mix=params['mix'],
                real_support=params['real_support'],
                softmax=softmax,
                n_critic=n_critic,
            )

            # renew best measurements
            for eps in evals:
                # check if best result in every category
                for method in evals[eps]:
                    try:
                        method_splits = method.split('-')
                        # check if method name has train_val_samples at the end
                        train_samples = int(method_splits[-1])
                        actual_method = '-'.join(method_splits[:-1])
                    except:
                        train_samples = best_train_samples  # make sure max value doesnt change
                        actual_method = method

                    if evals[eps][method][0] > max_accs.get(actual_method, [-1])[0]:
                        max_accs[actual_method] = evals[eps][method]
                        best_eps[actual_method] = eps
                        best_params[actual_method] = params
                        best_train_samples = train_samples

            if log_fn is not None and evals:
                with open(log_fn, 'a') as lfp:
                    lfp.write('#' * len(str(params)) + '\n' + str(params) + '\n')
                    lfp.write(','.join(['eps'] + list(max_accs)) + '\n')  # get eval method names
                    for eps in evals:
                        lfp.write(str(eps) + ',')
                        lfp.write(
                            ','.join(
                                [
                                    '-'.join(['{:.5f}'.format(acc) for acc in evals[eps][method]])
                                    for method in evals[eps]
                                ]
                            )
                        )
                        lfp.write('\n')
                    lfp.write('#' * len(str(params)) + '\n')

        best_val_msg = []
        for method in max_accs:
            best_msg_params = {'eps': best_eps[method]}
            if generalized:
                best_msg_params.update(
                    {'train_samples': best_train_samples, 'val_samples': val_samples}
                )
            best_msg_params.update(best_params[method])

            best_val_msg.append(
                '{}:{}={}'.format(
                    method,
                    best_msg_params,
                    '-'.join(['{:.5f}'.format(acc) for acc in max_accs[method]]),
                )
            )

        delim = '^' * len(best_val_msg[0])
        best_val_msg = [delim] + best_val_msg + [delim]
        best_val_msg = '\n'.join(best_val_msg) + '\n'
        print(best_val_msg)
        if log_fn is not None:
            with open(log_fn, 'a') as lfp:
                lfp.write(best_val_msg)

    else:
        best_params = {testing: next(iter(param_grid))}
        best_eps = {testing: episodes}

    if testing is not None or (not grid_search and len(param_grid) == 1):
        params = best_params[testing]
        eps = best_eps[testing]
        best_msg_params = {'eps': eps}
        if generalized:
            best_msg_params.update(
                {'train_samples': best_train_samples, 'val_samples': val_samples}
            )
        best_msg_params.update(params)
        best_msg = 'Running best params {}'.format(best_msg_params)
        print(best_msg + '\n' + '$' * len(best_msg))

        n_critic = 5 if params['lambda_wgan'] > 0 else 0

        trainer = Z2FSL_VAEGAN_Trainer(
            train_split='trainval',
            eval_split='test',
            metadata_dim=metadata_dim,
            generator_hidden_layers=params['generator_hidden_layers'],
            classifier_hidden_layers=params['classifier_hidden_layers'],
            critic_hidden_layers=params['critic_hidden_layers'],
            embedding_dim=params['embedding_dim'],
            lambda_wgan=params['lambda_wgan'],
            way=params['way'],
            shot=params['shot'],
            queries_per=params['queries_per'],
            pretrained_fsl_dir=pretrained_dir if params['pretrained_fsl'] else '',
            lambda_z2fsl=params['lambda_z2fsl'],
            lambda_vae=params['lambda_vae'],
            dataset_dir=dataset_dir,
            feature_dir=feature_dir,
            feature_norm=feature_norm,
            proposed_splits=params['proposed_splits'],
            l2_attr_norm=params['l2_attr_norm'],
            device=device,
            train_fsl=params['train_fsl'],
            augmentation=params['augmentation'],
            generalized=params['generalized'],
        )

        evals = trainer.train(
            episodes=eps,
            eval_interval=eps,
            print_interval=50,
            gen_lr=params['gen_lr'],
            gen_beta1=params['gen_beta1'],
            fsl_lr=params['fsl_lr'],
            fsl_beta1=params['fsl_beta1'],
            val_epochs=val_epochs,
            val_samples=val_samples,
            train_val_samples=[best_train_samples],
            val_beta1=val_beta1,
            val_lr=val_lr,
            mix=params['mix'],
            real_support=params['real_support'],
            softmax=softmax,
            n_critic=n_critic,
        )

        if log_fn is not None:
            eval_msg = 'Evaluation {}={}\n'.format(
                best_msg_params, ','.join([str(acc) for acc in evals[eps].values()])
            )
            with open(log_fn, 'a') as lfp:
                lfp.write(eval_msg)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-ms', '--manual_seed', type=int, help='set random seed')
    parser.add_argument(
        '-dt',
        '--direct_testing',
        default=False,
        action='store_true',
        help='whether to directly run test if only one set of params is given',
    )
    parser.add_argument(
        '-test',
        '--testing',
        type=str,
        choices=['normal', 'more-shots', 'finetune', 'softmax', 'mapped-softmax'],
        help='if set, evaluates best configuration on test',
    )
    parser.add_argument(
        '-gen',
        '--generalized',
        type=str2bool,
        nargs='+',
        default=[False],
        help='whether to run GZSL',
    )
    parser.add_argument(
        '-pfsl',
        '--pretrained_fsl',
        type=str2bool,
        nargs='+',
        default=[True],
        help='whether to use pretrained prototypical net',
    )
    parser.add_argument(
        '-fsld',
        '--pretrained_fsl_dir',
        type=str,
        default='models',
        help='directory of pretrained fsls.',
    )
    parser.add_argument('-md', '--metadata_dim', type=int, help='dimensionality of metadata')
    parser.add_argument(
        '-tfsl',
        '--train_fsl',
        type=str2bool,
        nargs='+',
        default=[False],
        help='whether to train the fsl algo',
    )
    parser.add_argument(
        '-dd', '--dataset_dir', type=str, default='CUB_200_2011', help='dataset root directory'
    )
    parser.add_argument(
        '-fd',
        '--feature_dir',
        type=str,
        default='features',
        help='feature directory wrt dataset_dir',
    )
    parser.add_argument(
        '-fn',
        '--feature_norm',
        default=False,
        action='store_true',
        help='whether to normalize features',
    )
    parser.add_argument(
        '-ps',
        '--proposed_splits',
        type=str2bool,
        nargs='+',
        default=[True],
        help='whether to use proposed splits',
    )
    parser.add_argument(
        '-l2',
        '--l2_attr_norm',
        type=str2bool,
        nargs='+',
        default=[True],
        help='whether to perform l2 norm on individual attributes',
    )
    parser.add_argument(
        '-rs',
        '--real_support',
        type=str2bool,
        nargs='+',
        default=[True],
        help='whether support of seen classes is real or generated',
    )
    parser.add_argument(
        '-aug',
        '--augmentation',
        type=str2bool,
        nargs='+',
        default=[True],
        help=('whether to use data augmentation ' '(not available for proposed features)'),
    )
    parser.add_argument('-l', '--log_fn', type=str, help='file to log val metrics')
    parser.add_argument(
        '-glr',
        '--gen_learning_rates',
        type=float,
        nargs='+',
        default=[1e-4],
        help='wgan learning rates to search over',
    )
    parser.add_argument(
        '-flr',
        '--fsl_learning_rates',
        type=float,
        nargs='+',
        default=[1e-4],
        help='fsl learning rates to search over',
    )
    parser.add_argument(
        '-gb1',
        '--gen_beta1s',
        type=float,
        nargs='+',
        default=[
            0.5,
        ],
        help='wgan beta_1 coefficients of Adam to search over',
    )
    parser.add_argument(
        '-fb1',
        '--fsl_beta1s',
        type=float,
        nargs='+',
        default=[
            0.5,
        ],
        help='fsl beta_1 coefficients of Adam to search over',
    )
    parser.add_argument(
        '-chl',
        '--critic_hidden_layers',
        type=str,
        nargs='+',
        default=['4096'],
        help=(
            'per layer hidden neurons for the critic to search over'
            ', integers split by dashes for multiple hidden layers'
        ),
    )
    parser.add_argument(
        '-ghl',
        '--generator_hidden_layers',
        type=str,
        nargs='+',
        default=['4096'],
        help=(
            'per layer hidden neurons for the generator to search over'
            ', integers split by dashes for multiple hidden layers'
        ),
    )
    parser.add_argument(
        '-clsfhl',
        '--classifier_hidden_layers',
        type=str,
        nargs='+',
        default=['2048'],
        help=(
            'per layer hidden neurons for the fsl algo to search over'
            ', integers split by dashes for multiple hidden layers'
        ),
    )
    parser.add_argument(
        '-emb',
        '--embedding_dims',
        type=int,
        nargs='+',
        default=[2048],
        help='embedding dim for protonet (fsl algo)',
    )
    parser.add_argument(
        '-mix',
        dest='mix',
        type=str2bool,
        nargs='+',
        default=[False],
        help='whether to mix support and queries during training',
    )
    parser.add_argument(
        '-s',
        '--shots',
        type=int,
        nargs='+',
        default=[
            10,
        ],
        help='samples per class during training',
    )
    parser.add_argument(
        '-w',
        '--ways',
        type=int,
        nargs='+',
        default=[
            20,
        ],
        help='number of classes per episode',
    )
    parser.add_argument(
        '-qp',
        '--queries_pers',
        type=int,
        nargs='+',
        default=[
            10,
        ],
        help='number of samples to fetch per class',
    )
    parser.add_argument(
        '-z2f',
        '--lambda_z2fsl',
        type=float,
        nargs='+',
        default=[0],
        help='classifier loss coefficients to search over',
    )
    parser.add_argument(
        '-wgan',
        '--lambda_wgan',
        type=float,
        nargs='+',
        default=[1000],
        help='wgan loss coefficients to search over',
    )
    parser.add_argument(
        '-vae',
        '--lambda_vae',
        type=float,
        nargs='+',
        default=[1],
        help='vae loss coefficients to search over',
    )
    parser.add_argument(
        '-vs',
        '--val_samples',
        type=int,
        default=500,
        help='number of samples per class to generate during testing',
    )
    parser.add_argument(
        '-tvs',
        '--train_val_samples',
        type=int,
        nargs='+',
        help='number of samples per class for seen classes during testing',
    )
    parser.add_argument(
        '-ve',
        '--val_epochs',
        type=int,
        default=25,
        help='number of epochs to train on synthetic data',
    )
    parser.add_argument(
        '-vlr', '--val_lr', type=float, default=1e-3, help='learning rate of synthetic classifier'
    )
    parser.add_argument(
        '-vb1',
        '--val_beta1',
        type=float,
        default=0.5,
        help='beta1 coefficient of Adam for synthetic classifier',
    )
    parser.add_argument('-dvc', '--device', type=str, default='cuda:0', help='device to run on')
    parser.add_argument('-eps', '--episodes', type=int, default=15000, help='episodes to run')
    parser.add_argument(
        '-eval',
        '--eval_interval',
        type=int,
        default=1000,
        help='per how many episodes to evaluate model',
    )
    parser.add_argument(
        '-soft',
        '--softmax',
        default=False,
        action='store_true',
        help='whether to run softmax or not',
    )

    args = parser.parse_args()

    if args.manual_seed is not None:
        manual_seed(args.manual_seed)

    ds_name = dataset_name(args.dataset_dir)
    if args.metadata_dim is None:
        if ds_name == 'cub':
            args.metadata_dim = 312
        elif ds_name == 'awa2':
            args.metadata_dim = 85
        elif ds_name == 'sun':
            args.metadata_dim = 102

    if args.train_val_samples is None:
        args.train_val_samples = [args.val_samples]

    # correct if more that necessary options are provided
    args.proposed_splits = list(set(args.proposed_splits))
    args.l2_attr_norm = list(set(args.l2_attr_norm))
    args.train_fsl = list(set(args.train_fsl))
    args.generalized = list(set(args.generalized))
    args.pretrained_fsl = list(set(args.pretrained_fsl))
    args.real_support = list(set(args.real_support))

    # just turn into lists
    args.critic_hidden_layers = [
        [int(n_h) for n_h in chl.split('-') if chl] for chl in args.critic_hidden_layers
    ]

    args.generator_hidden_layers = [
        [int(n_h) for n_h in chl.split('-') if chl] for chl in args.generator_hidden_layers
    ]

    args.classifier_hidden_layers = [
        [int(n_h) for n_h in chl.split('-') if chl] for chl in args.classifier_hidden_layers
    ]

    if args.testing == 'softmax':
        args.softmax = True

    test_setting(
        pretrained_fsl=args.pretrained_fsl,
        pretrained_dir=args.pretrained_fsl_dir,
        dataset_dir=args.dataset_dir,
        grid_search=not args.direct_testing,
        metadata_dim=args.metadata_dim,
        feature_dir=args.feature_dir,
        proposed_splits=args.proposed_splits,
        l2_attr_norm=args.l2_attr_norm,
        augmentation=args.augmentation,
        feature_norm=args.feature_norm,
        gen_learning_rates=args.gen_learning_rates,
        fsl_beta1s=args.fsl_beta1s,
        fsl_learning_rates=args.fsl_learning_rates,
        gen_beta1s=args.gen_beta1s,
        generator_hidden_layers=args.generator_hidden_layers,
        classifier_hidden_layers=args.classifier_hidden_layers,
        critic_hidden_layers=args.critic_hidden_layers,
        embedding_dims=args.embedding_dims,
        mix=args.mix,
        ways=args.ways,
        shots=args.shots,
        queries_pers=args.queries_pers,
        lambda_z2fsl=args.lambda_z2fsl,
        lambda_vae=args.lambda_vae,
        lambda_wgan=args.lambda_wgan,
        device=args.device,
        val_epochs=args.val_epochs,
        val_samples=args.val_samples,
        train_val_samples=args.train_val_samples,
        episodes=args.episodes,
        eval_interval=args.eval_interval,
        val_lr=args.val_lr,
        val_beta1=args.val_beta1,
        log_fn=args.log_fn,
        train_fsl=args.train_fsl,
        generalized=args.generalized,
        testing=args.testing,
        real_support=args.real_support,
        softmax=args.softmax,
    )
