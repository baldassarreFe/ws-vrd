import torch


def precision_at(annotations, scores, sizes):
    """Precision@x

    - rank the relationships by their score and keep the top x
    - compute how many of those retrieved relationships are actually relevant

    Args:
        annotations: tensor of shape [num_samples, num_relationships] and values {0, 1}
        scores: tensor of shape [num_samples, num_relationships] and float scores

    ::

                  # ( relevant items retrieved )
      Precision = ------------------------------ = P ( relevant | retrieved )
                      # ( retrieved items )
    """
    result = {}
    # Sorted labels are the indexes that would sort y_score, e.g.
    # [[ 10, 3, 4, ....., 5, 41 ],
    #  [  1, 2, 6, ....., 8, 78 ]]
    # means that for the first image class 10 is the top scoring class
    sorted_labels = torch.argsort(scores, dim=1, descending=True)

    # One could do this to get the sorted scores
    # sorted_scores = torch.gather(y_scores, index=sorted_labels, dim=1)

    # Use these indexes to index into y_true, but keep only max(sizes) columns
    annotations_of_top_max_s = torch.gather(
        annotations, index=sorted_labels[:, : max(sizes)], dim=1
    )

    # cumsum[i, j] = number of relevant items within the top j+1 retrieved items
    # Cast to float to avoid int/int division.
    cumsum = annotations_of_top_max_s.cumsum(dim=1).float()

    # Given a size s, `cumsum[i, s-1] / s` gives the precision for sample i.
    # If we end up doing 0 / 0, it simply means that in the top-s documents
    # there was no relevant document, so precision should be 0.
    # Then we take the batch mean.
    for s in sizes:
        precision_per_sample = cumsum[:, (s - 1)] / s
        precision_per_sample[torch.isnan(precision_per_sample)] = 0.0
        result[s] = precision_per_sample.mean(dim=0).item()

    return result
