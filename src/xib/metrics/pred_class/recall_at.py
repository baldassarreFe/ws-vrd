from typing import Tuple, List

import torch
from ignite.engine import Engine
from ignite.metrics import Metric


def recall_at(annotations, scores, sizes):
    """Recall@x

    - rank the relationships by their score and keep the top x
    - compute how many of the relevant relationships are among the retrieved

    ::

                # ( relevant items retrieved )
      Recall  = ------------------------------ = P ( retrieved | relevant )
                     # ( relevant items )

    Args:
        annotations: tensor of shape [num_samples, num_relationships] and values {0, 1}
        scores: tensor of shape [num_samples, num_relationships] and float scores

    References:

        `"Visual Relationship Detection with Language Priors" (Lu et al.) <https://arxiv.org/abs/1608.00187>`_
        describes `recall@x` as:

            The evaluation metrics we report is recall @ 100 and recall @ 50.
            Recall @ x computes the fraction of times the correct relationship is predicted
            in the top x confident relationship predictions. Since we have 70 predicates and
            an average of 18 objects per image, the total possible number of relationship
            predictions is 100×70×100, which implies that the random guess will result in a
            recall @ 100 of 0.00014.

        `Weakly-supervised learning of visual relations (Peyre et al.) <https://arxiv.org/abs/1707.09472>`_
        cites the previous paper and describes `recall@x` as:

            We compute recall @ x which corresponds to the proportion of ground truth
            pairs among the x top scored candidate pairs in each image.
    """

    result = {}
    # Sorted labels are the indexes that would sort y_score, e.g.
    # [[ 10, 3, 4, ....., 5, 41 ],
    #  [  1, 2, 6, ....., 8, 78 ]]
    # means that for the first image class 10 is the top scoring class
    sorted_labels = torch.argsort(scores, dim=1, descending=True)

    # One could do this to get the sorted scores
    # sorted_scores = torch.gather(y_scores, index=sorted_labels, dim=1)

    # Use these indexes to index into y_true, but keep only max(self.sizes) columns
    annotations_of_top_max_s = torch.gather(
        annotations, index=sorted_labels[:, : max(sizes)], dim=1
    )

    # cumsum[i, j] = number of relevant items within the top j+1 retrieved items
    # Cast to float to avoid int/int division later.
    cumsum = annotations_of_top_max_s.cumsum(dim=1).float()

    # Divide each row (a sample) by the total number of relevant document for
    # that row, to get the recall per sample.
    # If for a specific sample there are 0 relevant documents we get NaN in the division.
    # Last, take the batch mean skipping NaN values.
    # If we get a batch where all documents have 0 relevant documents, it's a problem.
    num_relevant_per_sample = annotations.sum(dim=1, keepdims=True)
    recall_per_sample = cumsum / num_relevant_per_sample
    for s in sizes:
        recall_at_s_per_sample = recall_per_sample[:, s - 1]
        finite = torch.isfinite(recall_at_s_per_sample)
        result[s] = recall_at_s_per_sample[finite].mean(dim=0).item()

    return result


class RecallAtBatch(object):
    """Recall@x over the output of the last batch"""

    def __init__(self, sizes: Tuple[int, ...] = (10, 30, 50)):
        self._sorted_sizes = list(sorted(sizes))

    def __call__(self, engine: Engine):
        y_true = engine.state.output["target"]
        y_score = engine.state.output["output"]
        recalls = recall_at(y_true, y_score, self._sorted_sizes)

        # Return tensors so that ignite.metrics.Average takes batch size into account
        B = len(y_true)
        engine.state.output["recalls"] = {}
        for k, r in recalls.items():
            r = torch.full((B, 1), fill_value=r)
            engine.state.output["recalls"][f"pc/recall_at_{k}"] = r


class RecallAtEpoch(Metric):
    """Recall@x by accumulating outputs over epochs"""

    _y_true: List[torch.Tensor]
    _y_score: List[torch.Tensor]

    def __init__(
        self,
        sizes: Tuple[int, ...] = (10, 30, 50),
        output_transform=lambda x: x,
        device=None,
    ):
        self._sorted_sizes = list(sorted(sizes))
        super(RecallAtEpoch, self).__init__(output_transform, device)

    def reset(self):
        self._y_true = []
        self._y_score = []

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_true, y_score = output
        self._y_true.append(y_true)
        self._y_score.append(y_score)

    def compute(self):
        y_true = torch.cat(self._y_true, dim=0)
        y_score = torch.cat(self._y_score, dim=0)
        return recall_at(y_true, y_score, self._sorted_sizes)

    def completed(self, engine, name):
        result = self.compute()
        for k, v in result.items():
            engine.state.metrics[f"{name}_{k}"] = v
