set -ex

TRAIN_FLAGS="--lr 1e-4 --batch_size 128"

MODEL_FLAGS="--image_size 32 --num_channels 128 --num_res_blocks 3 --learn_sigma True --dropout 0.3"
DIFFUSION_FLAGS="--diffusion_steps 4000  --noise_schedule cosine --timestep_respacing ddim250 --num_samples 16 --use_ddim True --batch_size 16"

MODEL_PATH=model_ckpts/cifar10_uncond_50M_500K.pt

python3 scripts/image_sample.py --model_path $MODEL_PATH $MODEL_FLAGS $DIFFUSION_FLAGS
