"""Prototypical Net training."""

import os
import argparse

import torch
import torch.optim as optim
import torch.nn as nn
from sklearn.model_selection import ParameterGrid

from z2fsl.utils.losses import EpisodeCrossEntropyLoss
from z2fsl.modules.classifiers import PrototypicalNet
from z2fsl.utils.datasets import FeatureEpisodeFactory
from z2fsl.utils.general import (
    configuration_filename,
    str2bool,
    set_parameter_requires_grad,
    dataset_name,
)
from z2fsl.utils.submodules import GaussianNoiseAugmentation


class FSLTrainer:
    """PN trainer class.

    Attributes:
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
        noise_aug (`nn.Module`): Gaussian noise augmentation.
    """

    def __init__(
        self,
        dataset_dir,
        feature_dir='features',
        train_split='traintrain',
        eval_split='valtrain',
        proposed_splits=True,
        augmentation=False,
        feature_norm=True,
        feature_dim=2048,
        way=20,
        shot=10,
        queries_per=10,
        init_diagonal=True,
        n_hidden=0,
        device='cuda:0',
        noise_std=0,
        generalized=False,
    ):
        """Init.

        Args:
            dataset_dir (`str`): root dir of dataset.
            feature_dir (`str`, optional): directory of features relative to `dataset_dir`.
            proposed_splits (`bool`, optional): whether to use proposed splits,
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
            n_hidden (`int`, optional): number of hidden layers, default `0`.
            way (`int`, optional): way of training episodes, default `20`.
            shot (`int`, optional): shot of training episodes (provided by generator),
                default `5`. 
            queries_per (`int`, optional): number of samples per class in query set,
                default `10`.
            init_diagonal (`bool`, optional): whether to use PN intialization trick,
                default `True`.
            device (`str`, optional): device to run on, default `'cuda:0'`.
            `train_fsl` (`bool`, optional): whether to train FSL classifier.
            noise_std (`float`, optional): Gaussian noise std, default `0` (no noise).
        """



        self.fsl_classifier = PrototypicalNet(
            in_features=feature_dim,
            out_features=feature_dim,
            hidden_layers=n_hidden * [feature_dim],
            init_diagonal=init_diagonal,
        ).to(device)

        if noise_std > 0:
            self.noise_aug = GaussianNoiseAugmentation(std=noise_std)
        else:
            self.noise_aug = nn.Identity()
        self.noise_aug.to(device)

        self.episode_factory_kwargs = dict(
            dataset_dir=dataset_dir,
            feature_dir=feature_dir,
            feature_norm=feature_norm,
            generalized=generalized,
            proposed_splits=proposed_splits,
            augmentation=augmentation,
        )
        self.train_split = train_split
        self.eval_split = eval_split
        self.generalized = generalized
        self.way = way
        self.shot = shot
        self.queries_per = queries_per

        self.device = device

    def train(
        self,
        episodes=15000,
        lr=1e-4,
        beta1=0.5,
        l2_coef=0,
        lr_decay=False,
        print_interval=30,
        eval_interval=None,
    ):
        """Trains all components (FSL classifier, generator, encoder, discriminator)
        and provides evaluation metrics.

        Args:
            episodes (`int`): number of training iterations.
            lr (`float` | `None`, optional): FSL classifier's learning rate,
                default `1e-4`.
            beta1 (`float` | `None`, optional): FSL classifier's beta1 for Adam,
                default `0.5`.
            l2_coef (`float`, optional): weight decay, default `0`.
            lr_decay (`bool`, optional): whether to apply learning rate decay,
                default `False`.
            print_interval (`int`, optional): episodes to wait before printing several
                training losses, default `30`.
            eval_interval (`int` | `None`, optional): episodes to wait before printing
                evaluation metrics, default is at the end of training.

        Returns:
            `dict` iterations: accuracy key-value pairs, e.g.

            {
                1000: 0.34
                2000: 0.54,
                3000: 0.60
            }
        """

        if eval_interval is None:
            eval_interval = episodes

        episode_factory = FeatureEpisodeFactory(
            split=self.train_split, inner_split='train', **self.episode_factory_kwargs
        )
        optimizer = optim.Adam(
            self.fsl_classifier.parameters(), lr=lr, weight_decay=l2_coef, betas=(beta1, 0.999)
        )

        if lr_decay:
            lr_decay = optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=range(eval_interval, episodes, eval_interval)
            )
        else:
            lr_decay = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda x: 1)

        criterion = EpisodeCrossEntropyLoss()
        evals = {}

        for episode in range(episodes):
            support, query, _ = episode_factory.fsl_episode(self.way, self.shot, self.queries_per)
            support = [self.noise_aug(sup.to(self.device)) for sup in support]
            query = [self.noise_aug(que.to(self.device)) for que in query]
            logits = self.fsl_classifier(support, query)
            loss = criterion(logits)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_decay.step()

            if episode == 0 or (episode + 1) % print_interval == 0:
                correct = 0
                total = 0
                for i, class_logits in enumerate(logits):
                    preds = class_logits.cpu().argmax(dim=-1)
                    correct += (preds == i).sum().item()
                    total += preds.size(0)
                acc = correct / total
                print(
                    'Episode {} - Loss = {:.4f}, Accuracy = {:.2f}%'.format(
                        episode + 1, loss.item(), acc * 100
                    )
                )

            if (episode + 1) % eval_interval == 0:
                evals[episode + 1] = self.eval()

        return evals

    def eval(self, shot=None, episodes=500):
        """Calculates test accuracy.
        
        Args:
            shot (`int` | `None`, optional): shot of test support, default is
                same as training shot.
            episodes (`int`, optional): number of episodes to average accuracy
                over, default `500`.

        Returns:
            `float` accuracy. 
        """

        self.fsl_classifier.eval()
        set_parameter_requires_grad(self.fsl_classifier, False)

        if shot is None:
            shot = self.shot

        eval_episode_factory = FeatureEpisodeFactory(
            split=self.eval_split,
            stat_split=self.train_split,
            dataset_dir=self.episode_factory_kwargs['dataset_dir'],
            feature_dir=self.episode_factory_kwargs['feature_dir'],
            proposed_splits=self.episode_factory_kwargs['proposed_splits'],
            feature_norm=self.episode_factory_kwargs['feature_norm'],
            augmentation=False,
            generalized=self.generalized,
        )
        if self.generalized:
            train_episode_factory = FeatureEpisodeFactory(
                split=self.train_split,
                inner_split='test',
                dataset_dir=self.episode_factory_kwargs['dataset_dir'],
                feature_dir=self.episode_factory_kwargs['feature_dir'],
                proposed_splits=self.episode_factory_kwargs['proposed_splits'],
                feature_norm=self.episode_factory_kwargs['feature_norm'],
                augmentation=False,
                generalized=self.generalized,
            )

        acc = 0
        accs_s = 0
        accs_u = 0
        for _ in range(episodes):

            if self.generalized:
                n_classes = eval_episode_factory.n_classes + train_episode_factory.n_classes
                eval_way = int(eval_episode_factory.n_classes ** 2 / n_classes)
                train_way = eval_episode_factory.n_classes - eval_way
            else:
                n_classes = eval_episode_factory.n_classes
                eval_way = n_classes

            support, query, _ = eval_episode_factory.fsl_episode(
                way=eval_way, shot=shot, queries_per=self.queries_per
            )
            if self.generalized:
                _support, _query, _ = train_episode_factory.fsl_episode(
                    way=train_way, shot=shot, queries_per=self.queries_per
                )
                support.extend(_support)
                query.extend(_query)

            support = [sup.to(self.device) for sup in support]
            query = [que.to(self.device) for que in query]
            logits = self.fsl_classifier(support, query)

            accs = []
            for i, class_logits in enumerate(logits):
                preds = class_logits.cpu().argmax(dim=-1)
                accs.append((preds == i).sum().item() / len(preds))

            if self.generalized:
                acc_u = sum(accs[:eval_way]) / len(accs[:eval_way])
                acc_s = sum(accs[eval_way:]) / len(accs[eval_way:])
                accs_u += acc_u
                accs_s += acc_s
                acc += 2 * acc_s * acc_u / (acc_s + acc_u)
            else:
                acc += sum(accs) / len(accs)

        self.fsl_classifier.train()
        set_parameter_requires_grad(self.fsl_classifier, True)

        if self.generalized:
            return [acc / episodes, accs_u / episodes, accs_s / episodes]
        return [acc / episodes]


def test_setting(
    dataset_dir,
    feature_dir,
    feature_norm,
    episodes,
    eval_interval,
    way,
    shots,
    queries_pers,
    lrs,
    beta1s,
    weight_decays,
    init_diagonals,
    lr_decays,
    n_hiddens,
    proposed_splits,
    device,
    augmentation,
    noise_stds,
    generalized,
    log_fn=None,
):
    """Performs (optional) grid search and (optional) evaluation.
    For description of args, see `FSLTrainer` and its methods.
    `shots`, `queries_pers`, `lrs`, `beta1s`, `init_diagonals`,
    `lr_decays`, `n_hiddens`, `proposed_splits`, `weight_decays`,
    `augmentation`, `noise_stds`, `generalized` should be lists
    so that the grid search can search through."""

    if log_fn is not None:
        dire = os.path.split(log_fn)[0]
        if dire and not os.path.exists(dire):
            os.makedirs(dire)

    search_params = dict(
        shot=shots,
        queries_per=queries_pers,
        lr=lrs,
        beta1=beta1s,
        init_diagonal=init_diagonals,
        lr_decay=lr_decays,
        n_hidden=n_hiddens,
        proposed_splits=proposed_splits,
        weight_decay=weight_decays,
        augmentation=augmentation,
        feature_norm=[feature_norm],
        noise_std=noise_stds,
        generalized=generalized,
    )

    param_grid = ParameterGrid(search_params)
    for i, params in enumerate(param_grid):
        init_msg = 'Running {}/{}: {}'.format(i + 1, len(param_grid), params)
        print(init_msg + '\n' + '#' * len(init_msg))

        trainer = FSLTrainer(
            dataset_dir=dataset_dir,
            feature_dir=feature_dir,
            way=way,
            shot=params['shot'],
            queries_per=params['queries_per'],
            n_hidden=params['n_hidden'],
            init_diagonal=params['init_diagonal'],
            proposed_splits=params['proposed_splits'],
            device=device,
            augmentation=params['augmentation'],
            feature_norm=feature_norm,
            noise_std=params['noise_std'],
            generalized=params['generalized'],
        )

        evals = trainer.train(
            lr=params['lr'],
            beta1=params['beta1'],
            lr_decay=params['lr_decay'],
            l2_coef=params['weight_decay'],
            episodes=episodes,
            eval_interval=eval_interval,
        )

        print()
        for eps in evals:
            print(
                'Validation accuracy after {} episodes = {}'.format(
                    eps, ' '.join(['{:.1f}%'.format(acc * 100) for acc in evals[eps]])
                )
            )
        print()

        if log_fn is not None:
            with open(log_fn, 'a') as lfp:
                lfp.write(str(params) + '\n')
                for eps in evals:
                    lfp.write(str(eps) + ',')
                    lfp.write(','.join(['{:.5f}'.format(acc) for acc in evals[eps]]))
                    lfp.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-dd', '--dataset_dir', type=str, default='CUB_200_2011', help='dataset root directory'
    )
    parser.add_argument(
        '-fd',
        '--feature_dir',
        type=str,
        default='normalized_proposed_features',
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
        '-gen',
        '--generalized',
        type=str2bool,
        nargs='+',
        default=[False],
        help='whether to run GZSL',
    )
    parser.add_argument('-lr', '--learning_rates', type=float, default=1e-4, help='learning rate')
    parser.add_argument('-w', '--way', type=int, default=10, help='number of classes per episode')
    parser.add_argument('-s', '--shot', type=int, default=5, help='samples per class in support')
    parser.add_argument(
        '-qp', '--queries_pers', type=int, nargs='+', default=[5], help='queries per class in query'
    )
    parser.add_argument('-n', '--n_hidden', type=int, default=0, help='number of hidden layers')
    parser.add_argument('-dvc', '--device', type=str, default='cuda:0', help='device to run on')
    parser.add_argument(
        '-spl',
        '--split',
        type=str,
        choices=['train', 'trainval'],
        default='trainval',
        help='train split',
    )
    parser.add_argument('-eps', '--episodes', type=int, default=10000, help='episodes to run')
    parser.add_argument('-mf', '--model_fn', type=str, required=True, help='model directory')

    args = parser.parse_args()

    trainer = FSLTrainer(
        dataset_dir=args.dataset_dir,
        way=args.way,
        queries_per=args.queries_per,
        shot=args.shot,
        n_hidden=args.n_hidden,
        train_split=args.split,
        augmentation=True,
        generalized=args.generalized,
        feature_dir=args.feature_dir,
        proposed_splits=True,
        feature_norm=True,
        device=args.device,
        init_diagonal=True,
    )

    trainer.train(episodes=args.episodes, lr=args.learning_rate, beta1=0.5, print_interval=200)
    model_fn = os.path.join(
        args.model_path,
        'fsl_{}_{}'.format(
            dataset_name(args.dataset_dir),
            configuration_filename(args.feature_dir, True, args.split, args.generalized),
        ),
    )
    torch.save(trainer.fsl_classifier.to('cpu'), model_fn)
