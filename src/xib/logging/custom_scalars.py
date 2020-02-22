from tensorboardX import SummaryWriter


def add_custom_scalars(writer: SummaryWriter):
    writer.add_custom_scalars(
        {
            "Predicate classification": {
                "BCE": ["MultiLine", ["(train|val)/loss/bce"]],
                "Rank": ["MultiLine", ["(train|val)/loss/rank"]],
                "Recall@5": ["MultiLine", ["(train|val)/recall_at_5"]],
                "Recall@10": ["MultiLine", ["(train|val)/recall_at_10"]],
                "Mean Average Precision": ["MultiLine", ["(train|val)/mAP"]],
            },
            "Visual relations detection metrics": {
                "Predicate Recall@50": [
                    "MultiLine",
                    ["val_vr/predicate/(with_obj_scores|no_obj_scores)/recall_at_50"],
                ],
                "Predicate Recall@100": [
                    "MultiLine",
                    ["val_vr/predicate/(with_obj_scores|no_obj_scores)/recall_at_100"],
                ],
                "Phrase Recall@50": [
                    "MultiLine",
                    ["val_vr/phrase/(with_obj_scores|no_obj_scores)/recall_at_50"],
                ],
                "Phrase Recall@100": [
                    "MultiLine",
                    ["val_vr/phrase/(with_obj_scores|no_obj_scores)/recall_at_100"],
                ],
                "Relationship Recall@50": [
                    "MultiLine",
                    [
                        "val_vr/relationship/(with_obj_scores|no_obj_scores)/recall_at_50"
                    ],
                ],
                "Relationship Recall@100": [
                    "MultiLine",
                    [
                        "val_vr/relationship/(with_obj_scores|no_obj_scores)/recall_at_100"
                    ],
                ],
            },
            "Others": {"GPU (MB)": ["MultiLine", ["(train|val|val_vr)/gpu_mb"]]},
        }
    )
