CUDA_VISIBLE_DEVICES=7 python main.py \
    algorithm=distributed_sarah \
    algorithm.lr=0.001 \
    algorithm.num_epochs=50 \
    runtime.wandb.enabled=true \
    runtime.wandb.project="distributed-sarah" \
    runtime.wandb.name="dist-baseline"