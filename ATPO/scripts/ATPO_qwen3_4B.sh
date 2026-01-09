# Switch to the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"

# ============================ Environment Setting ============================
export NCCL_PRIMS_PROFILE_ENABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_CHECK_DISABLE=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_LL_THRESHOLD=16384
export NCCL_IB_CUDA_SUPPORT=1
export NCCL_SOCKET_IFNAME=bond1
export UCX_NET_DEVICES=bond1
export NCCL_IB_HCA=mlx5_bond_1,mlx5_bond_5,mlx5_bond_3,mlx5_bond_7,mlx5_bond_4,mlx5_bond_8,mlx5_bond_2,mlx5_bond_6
export NCCL_COLLNET_ENABLE=0
export SHARP_COLL_ENABLE_SAT=0
export NCCL_NET_GDR_LEVEL=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_TC=160
export NCCL_PXN_DISABLE=0
export NCCL_DEBUG="WARN"
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Set basic environment variables
#export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1           
#export VLLM_ATTENTION_BACKEND=XFORMERS
export VLLM_ATTENTION_BACKEND=FLASH_ATTN 
export VERL_LOGGING_LEVEL=WARN
export MKL_SERVICE_FORCE_INTEL=1    
export MKL_THREADING_LAYER=GNU       
export RAY_memory_usage_threshold=0.8  
export RAY_memory_monitor_refresh_ms=0 
export RAY_DEBUG=1

# Set Python path
export PYTHONPATH=${PARENT_DIR}/verl_atpo:$PYTHONPATH

# ============================ Basic Configuration ============================
# Experiment name and project
PROJECT_NAME="ATPO"
EXPERIMENT_NAME="multihop_qwen3_4B"

# Configuration file path
CONFIG_PATH="${PARENT_DIR}/scripts/config" # Modify the absolute path of the config folder, relative path is not recommended
CONFIG_NAME="ppo_trainer_dr.yaml"

# Distributed training settings
NNODES=1                            
N_GPUS_PER_NODE=8                   

# ============================ Data Configuration ============================
# Data parameters
PROMPT_KEY="prompt"                # Prompt field name
TRAIN_BATCH_SIZE=64                # Training batch size
PPO_MINI_BATCH_SIZE=8              # PPO mini-batch size
MAX_PROMPT_LENGTH=2000             # Maximum prompt length
MAX_RESPONSE_LENGTH=6192           # Maximum response length

# Data file paths
#TRAIN_FILES="${PARENT_DIR}/../ARPO/rl_datasets/train_10k.parquet"
#VALID_FILES=["${PARENT_DIR}/../ARPO/rl_datasets/valid.parquet"]

TRAIN_FILES="path/to/your/train.parquet"
VALID_FILES=["path/to/your/test.parquet"]

# ============================ Model Configuration ============================
# Actor model path
ACTOR_MODEL_PATH="xxxxx"

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="vllm"                 # Use vllm engine
ROLLOUT_MODE="sync_with_tool_tree"       # Synchronous mode with tool support
BRANCH_PROBABILITY=0.5             # Branch probability  not used in offline tree search
Entropy_weight=0.2                 # used in offline tree search
# Tree related settings 
INITIAL_ROLLOUTS=10                 # Initial rollout number
EXPANSION_MODE="entropy"       # random or entropy
EXPANSION_ITERATIONS=2        # Number of expansion iterations
BEAM_SIZE=6                        # Beam size
SAMPLES_PER_TREE=22                  # Number of samples per tree
ROLLOUT_N=$SAMPLES_PER_TREE          # Number of responses generated per sample

# ============================ Reward Model Configuration ==========================
# Reward model settings
REWARD_MANAGER="naive"              # Reward manager type
CUSTOM_REWARD_FUNCTION_PATH="${PARENT_DIR}/verl_atpo/verl/utils/reward_score/deep_research_em.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS=5                      # Total training epochs
SAVE_FREQ=50                        # Save frequency
TEST_FREQ=20                        # Test frequency

# ============================ Path Configuration ============================
# Save path
CURRENT_DATE=$(date +%Y%m%d_%H%M%S)
SAVE_PATH="/mnt/private/logs/AEPO/${EXPERIMENT_NAME}/${CURRENT_DATE}/"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"
VALIDATION_SAVE_PATH="${SAVE_PATH}/validation"


# ============================ Preparation ============================
# Login to WandB (if API key is provided)
if [ "$WANDB_API_KEY" != "" ]; then
    wandb login --relogin $WANDB_API_KEY
    export WANDB_DIR=${SAVE_PATH}
fi

# Create save directory
if [ ! -d "$SAVE_PATH" ]; then
    mkdir -p $SAVE_PATH
fi

# Create rollout save directory
if [ ! -d "$ROLLOUT_SAVE_PATH" ]; then
    mkdir -p $ROLLOUT_SAVE_PATH
fi

# Create validation save directory
if [ ! -d "$VALIDATION_SAVE_PATH" ]; then
    mkdir -p $VALIDATION_SAVE_PATH
fi



# ============================ Start Training ============================
python3 -m verl.trainer.main_ppo \
    --config-path=$CONFIG_PATH \
    --config-name=$CONFIG_NAME \
    algorithm.adv_estimator=grpo \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.policy_loss=gspo_turn \
    actor_rollout_ref.actor.clip_ratio_low=3e-3 \
    actor_rollout_ref.actor.clip_ratio_high=4e-3 \
    actor_rollout_ref.actor.enable_entropy_balanced_clipping=${ENABLE_ENTROPY_BALANCED_CLIPPING} \
    actor_rollout_ref.actor.enable_entropy_balanced_advantage=${ENABLE_ENTROPY_BALANCED_ADVANTAGE} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.enable_dynamic_rollouts=${ENABLE_DYNAMIC_ROLLOUTS} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=${ROLLOUT_NAME} \
    actor_rollout_ref.rollout.mode=${ROLLOUT_MODE} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=${ROLLOUT_N} \
    actor_rollout_ref.rollout.initial_rollouts=${INITIAL_ROLLOUTS} \
    actor_rollout_ref.rollout.expansion_mode=${EXPANSION_MODE} \
    actor_rollout_ref.rollout.expansion_iterations=${EXPANSION_ITERATIONS} \
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.samples_per_tree=${SAMPLES_PER_TREE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
    actor_rollout_ref.rollout.leaf_value_norm=True \
    actor_rollout_ref.rollout.node_value_mode=child_softmax \
    actor_rollout_ref.rollout.node_adv_mode=node_value \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${API_KEY} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    custom_reward_function.path=${CUSTOM_REWARD_FUNCTION_PATH} \
    custom_reward_function.name=${CUSTOM_REWARD_FUNCTION_NAME} \
    trainer.critic_warmup=0 \
    trainer.logger="[console, wandb]" \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=${TOTAL_EPOCHS} \
    trainer.default_local_dir=${SAVE_PATH} \
    trainer.val_before_train=False \
    trainer.rollout_data_dir=${ROLLOUT_SAVE_PATH} \
    trainer.validation_data_dir=${VALIDATION_SAVE_PATH} \
    hydra.run.dir=${SAVE_PATH}/outputs 2>&1 | tee ${SAVE_PATH}/run.log 
    