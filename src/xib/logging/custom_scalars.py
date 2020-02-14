from tensorboardX import SummaryWriter


def add_custom_scalars(writer: SummaryWriter):
    writer.add_custom_scalars({
        'Predicate prediction losses': {
            'BCE': ['MultiLine', ['(train|val)/loss/bce']],
            'Rank': ['MultiLine', ['(train|val)/loss/rank']],
        },
        'Predicate prediction metrics': {
            'Recall@5': ['MultiLine', ['(train|val)/recall_at_5']],
            'Recall@10': ['MultiLine', ['(train|val)/recall_at_10']],
            'Mean Average Precision': ['MultiLine', ['(train|val)/mAP']],
        },
        'Visual relations metrics': {
            'Recall@50': ['MultiLine', ['val_vr/recall_at_50']],
            'Recall@100': ['MultiLine', ['val_vr/recall_at_100']],
        },
        'Others': {
            'GPU (MB)': ['MultiLine', ['(train|val|val_vr)/gpu_mb']],
        },
    })
