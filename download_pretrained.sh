"""
Code adpated from: Planning with Diffusion for Flexible Behavior Synthesis (
https://arxiv.org/abs/2205.09991)
"""

mkdir -p logs
gdown https://drive.google.com/uc?id=1wc1m4HLj7btaYDN8ogDIAV9QzgWEckGy
tar -xvf pretrained.tar --directory logs
rm pretrained.tar