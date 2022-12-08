from test_tube import HyperOptArgumentParser, SlurmCluster

from train import main

# grid search 3 values of learning rate and 3 values of number of layers for your net
# this generates 9 experiments (lr=1e-3, layers=16), (lr=1e-3, layers=32),
# (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)

num_gpu = 2

parser = HyperOptArgumentParser(strategy="grid_search", add_help=False)
parser.opt_list("--shadow_weight", default=1, type=float, options=[1e-4, 1e-3, 1e-2, 1, 10], tunable=True)
parser.add_argument('--cfg_path', type=str, help='config path')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--num_epochs', type=int, default=16, help='number of training epochs')
parser.add_argument('--num_gpus', type=int, default=num_gpu, help='number of gpus')
parser.add_argument('--num_nodes', type=int, default=1, help='number of nodes')
parser.add_argument('--num_workers', type=int, default=4, help='number of workers for data_loader')
parser.add_argument('--exp_name', type=str, default='exp', help='experiment name')

parser.add_argument('--test_batch_size', type=int, default=256, help='test batch size')

hyperparams = parser.parse_args()

# Slurm cluster submits 9 jobs, each with a set of hyperparams
cluster = SlurmCluster(
    hyperparam_optimizer=hyperparams,
    log_path="./logs",
)

# OPTIONAL FLAGS WHICH MAY BE CLUSTER DEPENDENT
# which interface your nodes use for communication
# cluster.add_command("export NCCL_SOCKET_IFNAME=^docker0,lo")

# see the output of the NCCL connection process
# NCCL is how the nodes talk to each other
# cluster.add_command("export NCCL_DEBUG=INFO")

# setting a main port here is a good idea.
# cluster.add_command("export MASTER_PORT=%r" % PORT)

# ************** DON'T FORGET THIS ***************
# MUST load the latest NCCL version
cluster.load_modules(["tools/anaconda3/2021.05"])
cluster.add_command('eval "$(conda shell.bash hook)"')
cluster.add_command('conda activate historic')


# configure cluster
cluster.per_experiment_nb_nodes = 1
cluster.per_experiment_nb_gpus = num_gpu

cluster.add_slurm_cmd(cmd="ntasks-per-node", value=32, comment="1 task per gpu")
cluster.add_slurm_cmd(cmd="partition", value='gpu', comment="gpu partition")
cluster.add_slurm_cmd(cmd="mem", value='200GB', comment="ram per gpu")

# submit a script with 9 combinations of hyper params
# (lr=1e-3, layers=16), (lr=1e-3, layers=32), (lr=1e-3, layers=64), ... (lr=1e-1, layers=64)
cluster.optimize_parallel_cluster_gpu(
    main, nb_trials=5, job_name="shadow_hp_search"  # how many permutations of the grid search to run
)