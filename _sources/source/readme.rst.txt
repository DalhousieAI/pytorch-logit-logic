Pytorch Logit Logic
===================

A pytorch extension which provides functions and classes for logit-space operators
equivalent to probabilistic Boolean logic-gates AND, OR, and XNOR for independent probabilities.

This provides the activation functions used in our paper:

    SC Lowe, R Earle, J d'Eon, T Trappenberg, S Oore (2022). Logical Activation Functions: Logit-space equivalents of Probabilistic Boolean Operators. In *Advances in Neural Information Processing Systems*, volume 36.
    doi: |nbsp| `10.48550/arxiv.2110.11940 <doi_>`_.

.. _doi: https://www.doi.org/10.48550/arxiv.2110.11940


For your convenience, we provide a copy of this citation in `bibtex`_ format.

.. _bibtex: https://raw.githubusercontent.com/DalhousieAI/pytorch-logit-logic/master/CITATION.bib


Example usage::

    from pytorch_logit_logic import actfun_name2factory
    from torch import nn

    class MLP(nn.Module):
        """
        A multi-layer perceptron which supports higher-dimensional activations.

        Parameters
        ----------
        in_channels : int
            Number of input channels.
        out_channels : int
            Number of output channels.
        n_layer : int, default=1
            Number of hidden layers.
        hidden_width : int, optional
            Pre-activation width. Default: same as ``in_channels``.
            Note that the actual pre-act width used may differ by rounding to
            the nearest integer that is divisible by the activation function's
            divisor.
        actfun : str, default="ReLU"
            Name of activation function to use.
        actfun_k : int, optional
            Dimensionality of the activation function. Default is the lowest
            ``k`` that the activation function supports, i.e. ``1`` for regular
            1D activation functions like ReLU, and ``2`` for GLU, MaxOut, and
            NAIL_OR.
        """
        def __init__(
            self,
            in_channels,
            out_channels,
            n_layer=1,
            hidden_width=None,
            actfun="ReLU",
            actfun_k=None,
        ):
            super().__init__()

            # Create a factory that generates objects that perform this activation
            actfun_factory = actfun_name2factory(actfun, k=actfun_k)
            # Get the divisor and space reduction factors for this activation
            # function. The pre-act needs to be divisible by the divisor, and
            # the activation will change the channel dimension by feature_factor.
            _actfun = actfun_factory()
            divisor = getattr(_actfun, "k", 1)
            feature_factor = getattr(_actfun, "feature_factor", 1)

            if hidden_width is None:
                hidden_width = in_channels

            # Ensure the hidden width is divisible by the divisor
            hidden_width = int(int(round(hidden_width / divisor)) * divisor)

            layers = []
            n_current = in_channels
            for i_layer in range(0, n_layer):
                layer = []
                layer.append(nn.Linear(n_current, hidden_width))
                n_current = hidden_width
                layer.append(actfun_factory())
                n_current = int(round(n_current * feature_factor))
                layers.append(nn.Sequential(*layer))
            self.layers = nn.Sequential(*layers)
            self.classifier = nn.Linear(n_current, out_channels)

        def forward(self, x):
            x = self.layers(x)
            x = self.classifier(x)
            return x


        model = MLP(
            in_channels=512,
            out_channels=10,
            n_layer=2,
            actfun="nail_or",
        )



.. |nbsp| unicode:: 0xA0
   :trim:
