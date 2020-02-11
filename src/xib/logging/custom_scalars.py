from tensorboardX import SummaryWriter


def add_custom_scalars(writer: SummaryWriter):
    writer.add_custom_scalars({
        'Losses': {
            'BCE': ['MultiLine', ['(train|val)/loss/bce']],
            'Rank': ['MultiLine', ['(train|val)/loss/rank']],
        },
        'Metrics': {
            'Recall@10': ['MultiLine', ['(train|val)/recall_at_10']],
            'Recall@30': ['MultiLine', ['(train|val)/recall_at_30']],
            'Recall@50': ['MultiLine', ['(train|val)/recall_at_50']],
            'Mean Average Precision': ['MultiLine', ['(train|val)/mAP']],
        },
        'Others': {
            'GPU': ['MultiLine', ['(train|val)/gpu_mb']],
        },
    })
