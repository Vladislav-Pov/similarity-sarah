cd "$(dirname "$0")/.."

# CUDA_VISIBLE_DEVICES=6 python main.py algorithm=distributed_sarah algorithm.lr=0.0013432590551431026 algorithm.momentum=0.9 algorithm.weight_decay=0.0007216635702812182 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t10

# CUDA_VISIBLE_DEVICES=6 python main.py algorithm=distributed_sarah algorithm.lr=0.0019470695875950575 algorithm.momentum=0.9 algorithm.weight_decay=0.00022650651832453877 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t22


#####

# CUDA_VISIBLE_DEVICES=6 python main.py algorithm=distributed_sarah algorithm.lr=0.004615898199487465 algorithm.momentum=0.95 algorithm.weight_decay=0.0005016255800735732 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t44

# CUDA_VISIBLE_DEVICES=6 python main.py algorithm=distributed_sarah algorithm.lr=0.003388817629101525 algorithm.momentum=0.95 algorithm.weight_decay=0.0001461899916533015 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-500-t28




##### NEW (train-loss теперь логируется): диагностика t39 — augmentation A/B, 500 эпох, GPU 6
CUDA_VISIBLE_DEVICES=2 python main.py algorithm=distributed_sarah algorithm.lr=0.004830576584733714 algorithm.momentum=0.95 algorithm.weight_decay=0.00013829366234897962 algorithm.num_epochs=500 runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-t39-noaug

CUDA_VISIBLE_DEVICES=2 python main.py algorithm=distributed_sarah algorithm.lr=0.004830576584733714 algorithm.momentum=0.95 algorithm.weight_decay=0.00013829366234897962 algorithm.num_epochs=500 data.augment_train=true runtime.wandb.enabled=true runtime.wandb.project=distributed_sarah runtime.wandb.name=dsarah-t39-aug
