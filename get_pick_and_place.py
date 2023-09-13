"""
Remenber to set the environment variable CLIPORT_ROOT to the root directory of the repository.
"""


# import libraries
import os
import json
import numpy as np
from cliport import agents
from cliport.utils import utils
from cliport.real_dataset import RealCamDataset
import numpy as np
import matplotlib.pyplot as plt



def get_agent(root_dir=os.environ['CLIPORT_ROOT']):
    """
    get the pre-trained agent using pre-defined args.

    params:
        root_dir: the root directory of the repository
    return:
        agent: the pre-trained agent

    TODO: get the agent using the config file and the input args

    """

    # model settings
    train_demos = 1000 # number training demonstrations used to train agent
    n_eval = 1 # number of evaluation instances
    mode = 'test' # val or test

    agent_name = 'cliport'
    model_task = 'multi-language-conditioned' # multi-task agent conditioned with language goals

    model_folder = 'cliport_quickstart' # path to pre-trained checkpoint
    ckpt_name = 'steps=400000-val_loss=0.00014655.ckpt' # name of checkpoint to load

    eval_task = 'packing-unseen-google-objects-seq'

    config_file = 'eval.yaml'

    vcfg = utils.load_hydra_config(os.path.join(root_dir, f'cliport/cfg/{config_file}'))
    vcfg['data_dir'] = os.path.join(root_dir, 'data')
    vcfg['mode'] = mode

    vcfg['model_task'] = model_task
    vcfg['eval_task'] = eval_task
    vcfg['agent'] = agent_name

    # Model and training config paths
    model_path = os.path.join(root_dir, model_folder)
    vcfg['train_config'] = f"{model_path}/{vcfg['model_task']}-{vcfg['agent']}-n{train_demos}-train/.hydra/config.yaml"
    vcfg['model_path'] = f"{model_path}/{vcfg['model_task']}-{vcfg['agent']}-n{train_demos}-train/checkpoints/"

    tcfg = utils.load_hydra_config(vcfg['train_config'])

    eval_run = 0
    name = '{}-{}-{}-{}'.format(vcfg['eval_task'], vcfg['agent'], n_eval, eval_run)
    print(f'\nEval ID: {name}\n')

    utils.set_seed(eval_run, torch=True)
    agent = agents.names[vcfg['agent']](name, tcfg, None, None)

    ckpt_path = os.path.join(vcfg['model_path'], ckpt_name)
    print(f'\nLoading checkpoint: {ckpt_path}')
    agent.load(ckpt_path)

    return agent


def show_imgs(img,agent,l):
    """
    show the input images and the affordance map of the input image
    run the forward pass of the agent to get the affordance map

    params:
        img: the input image
        agent: the pre-trained agent
        l: the language goal
    return:
        pick_conf: the pick affordance map
        place_conf: the place affordance map
    """

    fig, axs = plt.subplots(2, 2, figsize=(13, 7))

    color = np.uint8(img.detach().cpu().numpy()*255)[:,:,:3]
    color = color.transpose(1,0,2)
    depth = np.array(img.detach().cpu().numpy())[:,:,3]
    depth = depth.transpose(1,0)

    # Display input RGB
    axs[0,0].imshow(color)
    axs[0,0].axes.xaxis.set_visible(False)
    axs[0,0].axes.yaxis.set_visible(False)
    axs[0,0].set_title('Input RGB')

    # Display input depth
    axs[0,1].imshow(depth)
    axs[0,1].axes.xaxis.set_visible(False)
    axs[0,1].axes.yaxis.set_visible(False)
    axs[0,1].set_title('Input Depth')

    # Display predicted pick affordance
    axs[1,0].imshow(color)
    axs[1,0].axes.xaxis.set_visible(False)
    axs[1,0].axes.yaxis.set_visible(False)
    axs[1,0].set_title('Pick Affordance')

    # Display predicted place affordance
    axs[1,1].imshow(color)
    axs[1,1].axes.xaxis.set_visible(False)
    axs[1,1].axes.yaxis.set_visible(False)
    axs[1,1].set_title('Place Affordance')

    pick_inp = {'inp_img':img, 'lang_goal': l}
    pick_conf = agent.attn_forward(pick_inp)
    logits = pick_conf.detach().cpu().numpy()

    pick_conf = pick_conf.detach().cpu().numpy()
    argmax = np.argmax(pick_conf)
    argmax = np.unravel_index(argmax, shape=pick_conf.shape)
    p0 = argmax[:2]
    p0_theta = (argmax[2] * (2 * np.pi / pick_conf.shape[2])) * -1.0

    line_len = 30
    pick0 = (p0[0] + line_len/2.0 * np.sin(p0_theta), p0[1] + line_len/2.0 * np.cos(p0_theta))
    pick1 = (p0[0] - line_len/2.0 * np.sin(p0_theta), p0[1] - line_len/2.0 * np.cos(p0_theta))

    axs[1,0].plot((pick1[0], pick0[0]), (pick1[1], pick0[1]), color='r', linewidth=1)

    place_inp = {'inp_img': img, 'p0': [p0[0],p0[1],p0_theta], 'lang_goal': l}
    place_conf = agent.trans_forward(place_inp)

    place_conf = place_conf.permute(1, 2, 0)
    place_conf = place_conf.detach().cpu().numpy()
    argmax = np.argmax(place_conf)
    argmax = np.unravel_index(argmax, shape=place_conf.shape)
    p1_pix = argmax[:2]
    p1_theta = (argmax[2] * (2 * np.pi / place_conf.shape[2]) + p0_theta) * -1.0

    line_len = 30
    place0 = (p1_pix[0] + line_len/2.0 * np.sin(p1_theta), p1_pix[1] + line_len/2.0 * np.cos(p1_theta))
    place1 = (p1_pix[0] - line_len/2.0 * np.sin(p1_theta), p1_pix[1] - line_len/2.0 * np.cos(p1_theta))

    axs[1,1].plot((place1[0], place0[0]), (place1[1], place0[1]), color='g', linewidth=1)

    affordance_heatmap_scale = 30
    pick_logits_disp = np.uint8(logits * 255 * affordance_heatmap_scale).transpose(1,0,2)
    place_logits_disp = np.uint8(np.sum(place_conf, axis=2)[:,:,None] * 255 * affordance_heatmap_scale).transpose(1,0,2)

    pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)
    place_logits_disp_masked = np.ma.masked_where(place_logits_disp < 0, place_logits_disp)

    axs[1][0].imshow(pick_logits_disp_masked, alpha=0.75)
    axs[1][1].imshow(place_logits_disp_masked, cmap='viridis', alpha=0.75)

    plt.show()

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def get_pre_trained_pick_and_place(dataset=None, agent=None, l=None):
    """
    get the pick and place action using the pre-trained agent and the input image and language goal

    params:
        dataset: the input dataset
        agent: the pre-trained agent
        l: the language goal
    return:
        pick: the pick action
        place: the place action
    """

    # load agent
    agent = get_agent() if agent is None else agent

    # Load dataset
    ds =  RealCamDataset('real_cam_data') if dataset is None else  RealCamDataset(dataset)

    # get the language goal
    l = "pick up the green bottle cap to the cup" if l is None else l

    # get the first image
    img = ds.__getitem__(0)

    # get the pick and place action
    act = agent.act_img(img,l)
    pick = [float(i) for i in act['pick']]
    place = [float(i) for i in act['place']]
    out_dict = {'pick_loc':pick[:2], 'place_loc': place[:2], 'pick_rot': pick[2], 'place_rot': place[2]}

    print(act)

    with open('output.json', 'w') as f:
        json.dump(out_dict,f)

    # show the affordance
    show_imgs(img,agent,l)

if __name__ == '__main__':
    get_pre_trained_pick_and_place()