"""Supervised Classifiers."""

import torch
import torch.nn as nn

from z2fsl.utils.submodules import MLP


class PrototypicalNet(nn.Module):
    """Classifies examples based on distance metric
    from class prototypes. FSL setting.

    Attributes:
        mapper (`nn.Module`): mapper from feature to
            embedding space.
        dist (function): distance function. Accepts
            2D torch.Tensor prototypes and 2D
            torch.Tensor queries and returns a 2D torch.Tensor
            whose [i, j] element is the distance between
            queries[i] nad prototypes[j].
    """

    def __init__(
        self,
        in_features=512,
        out_features=512,
        hidden_layers=None,
        dist='euclidean',
        init_diagonal=False,
    ):
        """Init.

        Args:
            in_features (`int`, optional): input features dimension,
                default=`512`.
            out_features (`int`, optional): final output dimension,
                default=`512`.
            hidden_layers (`list` | `None`, optional): number of neurons
                in hidden layers, default is one hidden
                layer with units same as input.
            dist (function | `str`, optional): distance metric. If str,
                predefined distance is used accordingly.
                If function is passed, it should accept
                2D torch.Tensor prototypes and 2D
                torch.Tensor queries and return a 2D torch.Tensor
                whose [i, j] element is the distance between
                queries[i] nad prototypes[j].
            init_diagonal (`bool`, optional): whether to init linear layers
                with diagonal weights and zero biases, default=`False`.
        """

        super().__init__()

        if hidden_layers is None:
            hidden_layers = [in_features]

        self.mapper = MLP(in_features, out_features, hidden_layers, hidden_actf=nn.ReLU())
        if init_diagonal:
            self.mapper.init_diagonal()

        if isinstance(dist, str):
            self.dist = self.__getattribute__(dist)
        else:
            self.dist = dist

    @staticmethod
    def cosine(prototypes, queries):
        """Computes cosine distance between prototypes
        and set of queries.

        Args:
            prototypes (`torch.Tensor`): prototypes of size
                (way, embedding_dim).
            queries (`torch.Tensor`): queries of size
                (n_queries, embedding_dim).

        Returns:
            A torch.Tensor of size (n_queries, way) where
            element [i,j] contains distance between queries[i]
            and prototypes[j].
        """

        inner_prod = queries.matmul(prototypes.T)
        norm_i = queries.norm(dim=1, keepdim=True)
        norm_j = prototypes.norm(dim=1, keepdim=True).T
        return 1 - inner_prod / norm_i / norm_j

    @staticmethod
    def euclidean(prototypes, queries):
        """Computes euclidean distance between prototypes
        and set of queries.

        Args:
            prototypes (`torch.Tensor`): prototypes of size
                (way, embedding_dim).
            queries (`torch.Tensor`): queries of size
                (n_queries, embedding_dim).

        Returns:
            A torch.Tensor of size (n_queries, way) where
            element [i,j] contains distance between queries[i]
            and prototypes[j].
        """
        way = prototypes.size(0)
        n_queries = queries.size(0)

        prototypes = prototypes.repeat(n_queries, 1)
        queries = queries.repeat_interleave(way, 0)
        # after the repeats, prototypes have way classes after way classes after ...
        # and queries have way repeats of 1st query, way repeats of 2nd query, ...
        # so initial dist vector has distance of first query to all way classes
        # then the distance of the second query to all way class, etc
        return torch.norm(prototypes - queries, dim=1).view(n_queries, way)

    def forward(self, support, query, mix=False):
        """Episodic forward propagation.

        Computes prototypes given the support set of an episode
        and then makes inference on the corresponding query set.

        Args:
            support (`list` of `torch.Tensor`s): support set list
                whose every element is tensor of size
                (shot, feature_dim), i.e. shot image features
                belonging to the same class.
            query(`list` of `torch.Tensor`s): query set list
                whose every element is tensor of size
                (n_queries, feature_dim), i.e. n_queries image
                features belonging to the same class (for consistency
                purposes with support).
            mix(`bool`, optional): mix support and query, default `False`.

        Returns:
            A list of torch.Tensor of size (n_queries, way) logits
            whose i-th element consists of logits of queries belonging
            to the i-th class.
        """

        if mix:
            mix_support = []
            mix_query = []

            shot = support[0].size(0)

            for sup_feats, que_feats in zip(support, query):
                feats = torch.cat((sup_feats, que_feats))
                rand_idc = torch.randperm(len(feats)).numpy()

                mix_support.append(feats[rand_idc[:shot]])
                mix_query.append(feats[rand_idc[shot:]])

            support = mix_support
            query = mix_query

        prototypes = []
        for class_features in support:
            # class_features are (shot, feature_dim)
            prototypes.append(self.mapper(class_features).mean(dim=0))
        prototypes = torch.stack(prototypes)

        logits = []
        for class_features in query:
            # class_features are (n_queries, feature_dim)
            logits.append(-self.dist(prototypes, self.mapper(class_features)))

        return logits
