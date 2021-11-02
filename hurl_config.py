
# Default config
def default_config(env_name,
                   h_algo_name='VPG',
                   w_algo_name='BC'):

    # base config
    config = dict(
        algo_name='SAC',
        discount = None,
        n_epochs = 50,
        env_name = env_name,
        batch_size = 10000,
        seed=1,
        # offline batch data
        data_path='',  # directory of the snapshot
        data_itr='',
        episode_batch_size = 50000,
        # pretrain policy
        warmstart_policy=w_algo_name is not None,
        w_algo_name=w_algo_name,
        w_n_epoch = 30,
        # short-horizon RL params
        lambd=1.0,
        ls_rate=1.0,
        ls_cls='TanhLS',
        use_raw_snapshot=False,
        use_heuristic=h_algo_name is not None,
        h_algo_name=h_algo_name,
        h_n_epoch=30,
        vae_loss_percentile=99,  # an interger from 0-99
        # logging
        load_pretrained_data=False,
        snapshot_frequency=0,
        log_root=None,
        log_prefix='hp_tuning',
        save_mode='light',
        # optimization
        policy_lr=1e-3,       # Policy optimizer's learning rate
        value_lr=1e-3,          # Value function optimizer's learning rate
        opt_minibatch_size=128,  # optimization/replaybuffer minibatch size
        opt_n_grad_steps=1000,   # number of gradient updates
        num_evaluation_episodes=10,  # Number of evaluation episodes
        value_network_hidden_sizes=[256,256],
        policy_network_hidden_sizes=[64,64],
        n_workers=4,             # CAREFUL! Check the "conc_runs_per_node" property above. If conc_runs_per_node * n_workers > number of CPU cores on the target machine, the concurrent runs will likely interfere with each other.
        use_gpu=False,
        sampler_mode='ray',
        kl_constraint=0.05,      # kl constraint between policy updates
        gae_lambda=0.98,         # lambda of gae estimator
        lr_clip_range=0.2,
        eps_greed_decay_ratio=1.0,
        target_update_tau=5e-4,
        reward_avg_rate=1e-3,
        reward_scale=1.0
    )

    # Provide data_path and data_itr below
    if env_name=='HalfCheetah-v2':
        # setup
        config['batch_size'] = 4000
        config['n_epochs'] = 200

        # optimization run1823.49
        config['policy_lr'] = 0.00025
        config['value_lr'] = 0.00050
        config['discount'] = 0.99
        config['target_update_tau'] = 0.0400

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 30
        config['w_n_epoch'] = 30

        # hurl
        config['data_path'] = 'snapshots/SAC_HalfC_1.0_None_F_200/786495378/'
        config['data_itr'] = [0,199,4]


        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.99 100000.0 78 4433.1 10 run2347.91
            config['lambd'] = 0.99
            config['ls_rate'] =  100000
        elif config['h_algo_name'] == 'VPG':
            # 0.99 100000.000000 48 4864.0 8 run2351.0
            config['lambd'] = 0.99
            config['ls_rate'] =  100000.0


    if env_name=='Hopper-v2':
        # setup
        config['batch_size'] = 4000
        config['n_epochs'] = 200

        # optimization run1833.205
        config['policy_lr'] = 0.00025
        config['value_lr'] = 0.00050
        config['discount'] = 0.999
        config['target_update_tau'] = 0.0200

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 30
        config['w_n_epoch'] = 30

        # hurl
        config['data_path'] = 'snapshots/SAC_Hoppe_1.0_None_F_200/581079651/'
        config['data_itr'] = [0,199,4]


        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.98 0.000010 21 1301.0 13 run2344.60
            config['lambd'] = 0.98
            config['ls_rate'] =  0.000010
        elif config['h_algo_name'] == 'VPG':
            # 0.95 100000.000000 13 1827.3 12 run2352.14
            config['lambd'] = 0.95
            config['ls_rate'] =  100000.000000


    if env_name=='Humanoid-v2':
        # setup
        config['batch_size'] = 10000
        config['n_epochs'] = 500

        # optimization run1887.113
        config['policy_lr'] = 0.00200
        config['value_lr'] = 0.00025
        config['discount'] = 0.99
        config['target_update_tau'] = 0.0200

        # architecture
        config['policy_network_hidden_sizes'] = [256,256]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 80
        config['w_n_epoch'] = 50

        # hurl
        config['data_path'] = 'snapshots/SAC_Human_1.0_F_F/293494415/'
        config['data_itr'] = [0,200,4]

        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.90 0.000010 23 1965.5 3 run2164.87
            # 0.95 0.000010 24 1907.1 8 run2377.94 (after batch bug fix)
            config['lambd'] = 0.95
            config['ls_rate'] =  0.000010
        elif config['h_algo_name'] == 'VPG':
            # 0.90 1.000000 7 2640.7 11 run2379.119
            config['lambd'] = 0.9
            config['ls_rate'] =  1.0


    if env_name=='Swimmer-v2':
        # setup
        config['batch_size'] = 4000
        config['n_epochs'] = 200

        # optimization  run1888.258
        config['policy_lr'] = 0.00050
        config['value_lr'] = 0.00050
        config['discount'] = 0.999
        config['target_update_tau'] = 0.0100

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['episode_batch_size'] = config['batch_size']
        config['h_n_epoch'] = 30
        config['w_n_epoch'] = 30

        # hurl
        config['data_path'] = 'snapshots/SAC_Swimm_1.0_None_F/355552195'
        config['data_itr'] = [0,199,4]

        if config['h_algo_name'] is None and config['warmstart_policy']:
            # 0.99 1.000000 24 164.8 8 run2343.41
            config['lambd'] = 0.99
            config['ls_rate'] =  1.000000
        elif config['h_algo_name'] == 'VPG':
            # 0.95 1.000000 6 205.1 10 run2350.74
            config['lambd'] = 0.95
            config['ls_rate'] =  1.000000


    # Sparse Reacher Environment
    if env_name=='Sparse-Reacher-v2':
        # setup
        config['batch_size'] = 10000
        config['n_epochs'] = 200


        # optimization run3001.172 (for thres 0.01)
        config['policy_lr'] = 0.00025
        config['value_lr'] =  0.00025
        config['discount'] = 0.9
        config['target_update_tau'] = 0.0200

        # architecture
        config['policy_network_hidden_sizes'] = [64,64]
        config['value_network_hidden_sizes'] = [256,256]

        # batch training
        config['warmstart_policy'] = False

        if config['h_algo_name'] is not None:
            config['h_algo_name']=='GIVEN'
            config['lambd'] = 0.5
            config['ls_rate'] =  100000.0

    return config
