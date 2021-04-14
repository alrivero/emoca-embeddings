from utils.condor import execute_on_cluster
from pathlib import Path
import train_expdeca
import datetime
from omegaconf import OmegaConf
import time as t

def submit(cfg_coarse, cfg_detail, bid=10):
    cluster_repo_path = "/home/rdanecek/workspace/repos/gdl"
    submission_dir_local_mount = "/home/rdanecek/Workspace/mount/scratch/rdanecek/emoca/submission"
    submission_dir_cluster_side = "/ps/scratch/rdanecek/emoca/submission"
    time = datetime.datetime.now().strftime("%Y_%m_%d_%H-%M-%S")
    submission_folder_name = time + "_" + "submission"
    submission_folder_local = Path(submission_dir_local_mount) / submission_folder_name
    submission_folder_cluster = Path(submission_dir_cluster_side) / submission_folder_name

    local_script_path = Path(train_expdeca.__file__).absolute()
    cluster_script_path = Path(cluster_repo_path) / local_script_path.parents[1].name \
                          / local_script_path.parents[0].name / local_script_path.name

    submission_folder_local.mkdir(parents=True)

    coarse_file = submission_folder_local / "submission_coarse_config.yaml"
    detail_file = submission_folder_local / "submission_detail_config.yaml"

    with open(coarse_file, 'w') as outfile:
        OmegaConf.save(config=cfg_coarse, f=outfile)
    with open(detail_file, 'w') as outfile:
        OmegaConf.save(config=cfg_detail, f=outfile)


    # python_bin = 'python'
    python_bin = '/home/rdanecek/anaconda3/envs/<<ENV>>/bin/python'
    username = 'rdanecek'
    gpu_mem_requirement_mb = cfg_coarse.learning.gpu_memory_min_gb * 1024
    # gpu_mem_requirement_mb = None
    cpus = cfg_coarse.data.num_workers + 2 # 1 for the training script, 1 for wandb or other loggers (and other stuff), the rest of data loading
    gpus = cfg_coarse.learning.num_gpus
    num_jobs = 1
    max_time_h = 36
    max_price = 10000
    job_name = "train_deca"
    cuda_capability_requirement = 6
    mem_gb = 12
    args = f"{coarse_file.name} {detail_file.name}"

    execute_on_cluster(str(cluster_script_path),
                       args,
                       str(submission_folder_local),
                       str(submission_folder_cluster),
                       str(cluster_repo_path),
                       python_bin=python_bin,
                       username=username,
                       gpu_mem_requirement_mb=gpu_mem_requirement_mb,
                       cpus=cpus,
                       mem_gb=mem_gb,
                       gpus=gpus,
                       num_jobs=num_jobs,
                       bid=bid,
                       max_time_h=max_time_h,
                       max_price=max_price,
                       job_name=job_name,
                       cuda_capability_requirement=cuda_capability_requirement
                       )
    t.sleep(1)


def train_on_selected_sequences():
    from hydra.core.global_hydra import GlobalHydra

    coarse_conf = "deca_train_coarse_cluster"
    detail_conf = "deca_train_detail_cluster"

    finetune_modes = [
        # # DEFAULT without jaw
        # [
        #     ['model.useSeg=gt', 'model.exp_deca_jaw_pose=False'],
        #     ['model.useSeg=rend']
        # ],
        # # DEFAULT
        # [
        #     ['model.useSeg=gt'],
        #     ['model.useSeg=rend']
        # ],

        # DEFAULT, DISABLED UNNECESSARY DEEP LOSSES, HIGHER BATCH SIZE, NO SHAPE RING
        [
            ['model.useSeg=gt', 'model.idw=0', 'learning/batching=single_gpu_expdeca_coarse', 'model.shape_constrain_type=None'],
            ['model.useSeg=rend', 'model.idw=0', 'learning/batching=single_gpu_expdeca_detail',
                'model.shape_constrain_type=None', 'model.detail_constrain_type=None']
        ],

        # # DEFAULT but train coarse with detail
        # [
        #     ['model.useSeg=gt'],
        #     ['model.useSeg=rend', 'model.train_coarse=true']
        # ],

        # # DEFAULT but cloned ResNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=deca_clone'],
        #     ['model.useSeg=rend', 'model.expression_backbone=deca_clone']
        # ],
        #
        # # DEFAULT but static EmoNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_static'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_static']
        # ],
        #
        # # DEFAULT but trainable EmoNet backbone
        # [
        #     ['model.useSeg=gt', 'model.expression_backbone=emonet_trainable'],
        #     ['model.useSeg=rend', 'model.expression_backbone=emonet_trainable']
        # ]
    ]
    fixed_overrides_coarse = ['model/settings=coarse_train_expdeca',
                              'model.resume_training=True', # load the original DECA model
                              'learning.early_stopping.patience=5',
                              ]
    fixed_overrides_detail = ['model/settings=detail_train_expdeca',
                              'learning.early_stopping.patience=5',
                              ]

    # emonet_weights = [0.15,] #default
    emonet_weights = [0.15/100,] #new default
    # emonet_weights = [0.15, 0.15/5, 0.15/10, 0.15/50, 0.15/100]

    config_pairs = []
    for emeonet_reg in emonet_weights:
        for fmode in finetune_modes:
            coarse_overrides = fixed_overrides_coarse.copy()
            detail_overrides = fixed_overrides_detail.copy()
            # if len(fmode[0]) != "":
            coarse_overrides += fmode[0]
            detail_overrides += fmode[1]

            # data_override = f'data.sequence_index={video_index}'
            # pretrain_coarse_overrides += [data_override]
            # coarse_overrides += [data_override]
            # detail_overrides += [data_override]
            emonet_weight_override = f'model.emonet_weight={emeonet_reg}'
            coarse_overrides += [emonet_weight_override]
            detail_overrides += [emonet_weight_override]

            cfgs = train_expdeca.configure(
                coarse_conf, coarse_overrides,
                detail_conf, detail_overrides
            )

            GlobalHydra.instance().clear()
            config_pairs += [cfgs]

            submit(cfgs[0], cfgs[1])
            # break
        # break

    # for cfg_pair in config_pairs:
    #     submit(cfg_pair[0], cfg_pair[1])


def default_main():
    coarse_conf = "deca_train_coarse_cluster"
    coarse_overrides = []

    detail_conf = "deca_train_detail_cluster"
    detail_overrides = []

    cfg_coarse_pretrain, cfg_coarse, cfg_detail = train_expdeca.configure(
        coarse_conf, coarse_overrides,
        detail_conf, detail_overrides)

    submit(cfg_coarse_pretrain, cfg_coarse, cfg_detail)


if __name__ == "__main__":
    # default_main()
    train_on_selected_sequences()
