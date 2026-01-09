# Switch to the directory of the script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PARENT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PARENT_DIR"
echo "Switched to parent directory: $PARENT_DIR"

# ============================ Environment Setting ============================
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

# Set Python path
export PYTHONPATH=${PARENT_DIR}/verl_aepo_entropy:$PYTHONPATH

# ============================ Basic Configuration ============================
# Experiment name and project
PROJECT_NAME="agentic_rl_search"
EXPERIMENT_NAME="GRPO_qwen3_4B"

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
TRAIN_FILES="path/to/your/train.parquet"
VALID_FILES=["path/to/your/test.parquet"]

# ============================ Model Configuration ============================
# Actor model path
ACTOR_MODEL_PATH="path/to/your/actor-model"  # Modify the model path

# ============================ AEPO Configuration ============================
ENABLE_DYNAMIC_ROLLOUTS=False
ENABLE_ENTROPY_BALANCED_CLIPPING=False
ENABLE_ENTROPY_BALANCED_ADVANTAGE=False
ENABLE_POLICY_LOSS_GSPO=False

# ============================ Rollout Configuration ==========================
# Rollout settings
ROLLOUT_NAME="vllm"                 # Use vllm engine
ROLLOUT_MODE="sync_with_tool"       # Synchronous mode with tool support
ROLLOUT_N=16                         # Number of responses generated per sample
INITIAL_ROLLOUTS=16                 # Initial rollout number
BEAM_SIZE=2                        # Beam size
BRANCH_PROBABILITY=0.5             # Branch probability
Entropy_weight=0.2
# ============================ Rollout Tools Configuration ==========================
SEARCH_CACHE_PATH="path/to/your/search_cache.json" # Modify
API_KEY=None  # Modify your Bing Search API key
# ============================ Reward Model Configuration ==========================
# Reward model settings
REWARD_MANAGER="naive"              # Reward manager type
CUSTOM_REWARD_FUNCTION_PATH="${PARENT_DIR}/verl_aepo_entropy/verl/utils/reward_score/deep_research_em.py"
CUSTOM_REWARD_FUNCTION_NAME="compute_score"

# ============================ DAPO Configuration ==========================
ENABLE_OVERLONG_BUFFER=False
OVERLONG_BUFFER_LEN=2000
OVERLONG_BUFFER_PENALTY_FACTOR=1.0

ENABLE_FILTER_GROUPS=False
FILTER_GROUPS_METRIC="seq_reward"
FILTER_GROUPS_MAX_NUM_GEN_BATCHES=2
GEN_BATCH_SIZE=64


# ============================ Training Configuration ============================
# Training parameters
TOTAL_EPOCHS=5                      # Total training epochs
SAVE_FREQ=60                        # Save frequency
TEST_FREQ=20                        # Test frequency

# ============================ Path Configuration ============================
# Save path
SAVE_PATH="path/to/your/save_directory/${EXPERIMENT_NAME}"
ROLLOUT_SAVE_PATH="${SAVE_PATH}/rollout"
VALIDATION_SAVE_PATH="${SAVE_PATH}/validation"

# ============================ WandB Configuration ============================
# WandB settings
WANDB_API_KEY="/your/wandb_api_key" # Modify your wandb key

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
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.filter_groups.enable=${ENABLE_FILTER_GROUPS} \
    algorithm.filter_groups.metric=${FILTER_GROUPS_METRIC} \
    algorithm.filter_groups.max_num_gen_batches=${FILTER_GROUPS_MAX_NUM_GEN_BATCHES} \
    data.train_files=${TRAIN_FILES} \
    data.val_files=${VALID_FILES} \
    data.prompt_key=${PROMPT_KEY} \
    data.train_batch_size=${TRAIN_BATCH_SIZE} \
    data.gen_batch_size=${GEN_BATCH_SIZE} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    actor_rollout_ref.model.path=${ACTOR_MODEL_PATH} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.enable_entropy_balanced_clipping=${ENABLE_ENTROPY_BALANCED_CLIPPING} \
    actor_rollout_ref.actor.enable_entropy_balanced_advantage=${ENABLE_ENTROPY_BALANCED_ADVANTAGE} \
    actor_rollout_ref.actor.enable_policy_loss_gspo=${ENABLE_POLICY_LOSS_GSPO} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE} \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$((2*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
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
    actor_rollout_ref.rollout.beam_size=${BEAM_SIZE} \
    actor_rollout_ref.rollout.branch_probability=${BRANCH_PROBABILITY} \
    actor_rollout_ref.rollout.entropy_weight=${Entropy_weight} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.cache_file=${SEARCH_CACHE_PATH} \
    ++actor_rollout_ref.rollout.tools.tool_instances.search.params.api_key=${API_KEY} \
    actor_rollout_ref.rollout.multi_turn.enable=${ENABLE_MULTI_TURN} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$((4*(MAX_PROMPT_LENGTH+MAX_RESPONSE_LENGTH))) \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.reward_manager=${REWARD_MANAGER} \
    reward_model.overlong_buffer.enable=${ENABLE_OVERLONG_BUFFER} \
    reward_model.overlong_buffer.len=${OVERLONG_BUFFER_LEN} \
    reward_model.overlong_buffer.penalty_factor=${OVERLONG_BUFFER_PENALTY_FACTOR}  \
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
