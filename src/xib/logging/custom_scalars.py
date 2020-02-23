from tensorboardX import SummaryWriter


def add_custom_scalars(writer: SummaryWriter):
    writer.add_custom_scalars(
        {
            "Predicate classification": {
                "BCE": ["MultiLine", ["(train|val)_gt/loss/bce"]],
                "Rank": ["MultiLine", ["(train|val)_gt/loss/rank"]],
                "Recall@5": ["MultiLine", ["(train|val)_gt/pc/recall_at_5"]],
                "Recall@10": ["MultiLine", ["(train|val)_gt/pc/recall_at_10"]],
                "Mean Average Precision": ["MultiLine", ["(train|val)_gt/pc/mAP"]],
            },
            "Visual relations detection metrics": {
                "Predicate Recall@50": [
                    "MultiLine",
                    ["(val|test)_gt/vr/predicate/recall_at_50"],
                ],
                "Predicate Recall@100": [
                    "MultiLine",
                    ["(val|test)_gt/vr/predicate/recall_at_100"],
                ],
                "Phrase Recall@50": [
                    "MultiLine",
                    ["(val|test)_d2/vr/phrase/recall_at_50"],
                ],
                "Phrase Recall@100": [
                    "MultiLine",
                    ["(val|test)_d2/vr/phrase/recall_at_100"],
                ],
                "Relationship Recall@50": [
                    "MultiLine",
                    ["(val|test)_d2/vr/relationship/recall_at_50"],
                ],
                "Relationship Recall@100": [
                    "MultiLine",
                    ["(val|test)_d2/vr/relationship/recall_at_100"],
                ],
            },
            "Others": {"GPU (MB)": ["MultiLine", ["(train|val|test)_(gt|vr)/gpu_mb"]]},
        }
    )
