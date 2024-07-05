import pdb

import diffuser.sampling as sampling
import diffuser.utils as utils
import pickle
import torch

#-----------------------------------------------------------------------------#
#----------------------------------- setup -----------------------------------#
#-----------------------------------------------------------------------------#

class Parser(utils.Parser):
    dataset: str = 'walker2d-medium-replay-v2'
    config: str = 'config.locomotion'

args = Parser().parse_args('plan')


#-----------------------------------------------------------------------------#
#---------------------------------- loading ----------------------------------#
#-----------------------------------------------------------------------------#

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

## policies are wrappers around an unconditional diffusion model and a value guide
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


#-----------------------------------------------------------------------------#
#--------------------------------- main loop ---------------------------------#
#-----------------------------------------------------------------------------#

env = dataset.env

# Use torch.cuda.empty_cache() to clear any unused memory
torch.cuda.empty_cache()

# Set the default tensor type to CUDA for faster operations
torch.set_default_tensor_type(torch.cuda.FloatTensor)

# Increase the batch size to utilize more GPU memory
args.batch_size = 128  # Adjust this value based on your GPU capacity

rewards = []
states = []

# Use torch.no_grad() to disable gradient computation during inference
with torch.no_grad():
    for i in range(500):
        observation = env.reset(seed=i)
        total_reward = 0
        rollout = [observation.copy()]
        states.append(observation.copy())
        
        # Pre-allocate tensors for the entire episode
        episode_actions = torch.zeros((200, env.action_space.shape[0]), device='cuda')
        episode_rewards = torch.zeros(200, device='cuda')
        
        for t in range(200):
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
            env.render()
            if terminal:
                break

            observation = next_observation
        
        # Process the entire episode at once
        rewards.append(total_reward)
        print(f'i: {i} |   R: {total_reward:.2f}', flush=True)
        
        if i == 249:
            print("saving")
            filehandler = open("data/diffuser_"+ args.dataset + "_states","wb")
            pickle.dump(states,filehandler)
            filehandler.close()

            filehandler = open("data/diffuser_" + args.dataset + "_rewards","wb")
            pickle.dump(rewards,filehandler)
            filehandler.close()

# Final save
filehandler = open("data/diffuser_"+ args.dataset + "_states","wb")
pickle.dump(states,filehandler)
filehandler.close()

filehandler = open("data/diffuser_" + args.dataset + "_rewards","wb")
pickle.dump(rewards,filehandler)
filehandler.close()

#logger.finish(t, total_reward, terminal, diffusion_experiment, value_experiment)