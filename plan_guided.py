import pdb
import diffuser.sampling as sampling
import diffuser.utils as utils
import pickle
import torch
from PIL import Image
import cv2
import numpy as np

#------------------ setup --------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


#--------------- loading ---------------------#

## load diffusion model and value function from disk
diffusion_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.diffusion_loadpath,
    epoch=args.diffusion_epoch, seed=args.seed,
)
value_experiment = utils.load_diffusion(
    args.loadbase, args.dataset, args.value_loadpath,
    epoch=args.value_epoch, seed=args.seed,
)

## ensure that the diffusion model and value function are compatible with each other
utils.check_compatibility(diffusion_experiment, value_experiment)

diffusion = diffusion_experiment.ema.to('cuda')
dataset = diffusion_experiment.dataset
renderer = diffusion_experiment.renderer

## initialize value guides
value_function = value_experiment.ema.to('cuda')
guide_config = utils.Config(args.guide, model=value_function, verbose=False)
guide = guide_config()

logger_config = utils.Config(
    utils.Logger,
    renderer=renderer,
    logpath=args.savepath,
    vis_freq=args.vis_freq,
    max_render=args.max_render,
)

## policies are wrappers around an unconditional diffusion model and 
# a value guide
policy_config = utils.Config(
    args.policy,
    guide=guide,
    scale=args.scale,
    diffusion_model=diffusion,
    normalizer=dataset.normalizer,
    preprocess_fns=args.preprocess_fns,
    ## sampling kwargs
    sample_fn=sampling.n_step_guided_p_sample,
    n_guide_steps=args.n_guide_steps,
    t_stopgrad=args.t_stopgrad,
    scale_grad_by_std=args.scale_grad_by_std,
    verbose=False,
)

logger = logger_config()
policy = policy_config()


#----------------- main loop --------------------#

env = dataset.env

# torch.cuda.empty_cache() to clear any unused memory
torch.cuda.empty_cache()

# default tensor type to CUDA for faster operations
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# batch size to utilize more GPU memory
args.batch_size = 128 

rewards = []
states = []
frames = []
# Video recording
fourcc = cv2.VideoWriter_fourcc(*'XVID')
fps = 30.0  # Frames per second for the output video
frame_width = int(env.render(mode="rgb_array").shape[1])
frame_height = int(env.render(mode="rgb_array").shape[0])
out = cv2.VideoWriter(args.dataset + '.avi', fourcc, fps, (frame_width, frame_height))


# torch.no_grad() to disable gradient computation during inference
with torch.no_grad():
    for i in range(1000):
       
        observation = env.reset(seed=i)
        total_reward = 0
        rollout = [observation.copy()]
        states.append(observation.copy())
        
        # Pre-allocate tensors for the entire episode
        episode_actions = torch.zeros((50, env.action_space.shape[0]), device='cuda')
        episode_rewards = torch.zeros(50, device='cuda')
        
        for t in range(50):
            if t % 10 == 0: print(args.savepath, flush=True)

            state = env.state_vector().copy()
            conditions = {0: observation}
            
            # Perform batch inference
            actions, samples = policy(conditions, batch_size=args.batch_size, verbose=args.verbose)
            
            # Store the action
            episode_actions[t] = torch.tensor(actions, device='cuda')
            
            # Execute action in environment
            next_observation, reward, terminal, _ = env.step(actions)
            
            # Store the reward
            episode_rewards[t] = reward
            
            total_reward += reward
            rollout.append(next_observation.copy())
            frame = env.render(mode='rgb_array')
            if terminal:
                break

            observation = next_observation

            out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        out.release()

        print("Video saved as hopper_diffuser.mp4")
        # Process the entire episode at once
        rewards.append(total_reward)
        print(f'i: {i} |   R: {total_reward:.2f}', flush=True)


# Final save
filehandler = open("data/diffuser_train_"+ args.dataset + "_states","wb")
pickle.dump(states,filehandler)
filehandler.close()

filehandler = open("data/diffuser_train_" + args.dataset + "_rewards","wb")
pickle.dump(rewards,filehandler)
filehandler.close()
