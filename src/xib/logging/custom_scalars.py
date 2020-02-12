from tensorboardX import SummaryWriter


def add_custom_scalars(writer: SummaryWriter):
    writer.add_custom_scalars({
        'Losses': {
            'BCE': ['MultiLine', ['(train|val)/loss/bce']],
            'Rank': ['MultiLine', ['(train|val)/loss/rank']],
        },
        'Metrics': {
            'Recall@5': ['MultiLine', ['(train|val)/recall_at_5']],
            'Recall@10': ['MultiLine', ['(train|val)/recall_at_10']],
            'Mean Average Precision': ['MultiLine', ['(train|val)/mAP']],
        },
        'Others': {
            'GPU (MB)': ['MultiLine', ['(train|val)/gpu_mb']],
        },
    })
