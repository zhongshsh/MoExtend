import json
import wandb
import pytz
import deepspeed
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother
from datetime import datetime

from src import *

wandb_key = "xxxx" # https://docs.wandb.ai/quickstart#2-log-in-to-wb
wandb.login(key=wandb_key)
IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def get_args_parser():
    parser = argparse.ArgumentParser("Mixtral vision", add_help=False)
    parser.add_argument("--save_steps", type=int, default=300)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--eval_batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--save_total_limit", type=int, default=1)

    parser.add_argument("--checkpoint_idx", type=int, default=7)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--router_aux_loss_coef", type=float, default=0.001)
    parser.add_argument(
        "--project_name", type=str, default="VisionMixtralForCausalLM-test"
    )
    parser.add_argument("--model_id", type=str, default=None)
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument(
        "--vision_encode", action="store_true", help="whether train vision_encode."
    )
    parser.add_argument(
        "--expert_gate", action="store_true", help="whether just expert gate."
    )
    parser.add_argument(
        "--resume", action="store_true", help="whether resume checkpoint."
    )
    parser = deepspeed.add_config_arguments(parser)

    return parser


def preprocess(
    sources,
    tokenizer,
    max_len: int,
):
    talk_start = [tokenizer("<s>").input_ids[-1]]
    talk_end = [tokenizer("</s>").input_ids[-1]]

    # Apply prompt templates
    input_ids, targets = [], []
    for i, source in enumerate(sources):
        input_id, target = talk_start.copy(), talk_start.copy()
        assert len(input_id) == len(target)
        for j, sentence in enumerate(source):
            role = sentence["from"]
            if role == "user" or role == "human":
                _input_id = (
                    tokenizer.encode("[INST]", add_special_tokens=False)
                    + tokenizer.encode(sentence["value"], add_special_tokens=False)
                    + tokenizer.encode("[/INST]", add_special_tokens=False)
                )
                _target = [IGNORE_TOKEN_ID] * len(_input_id)
            elif role == "assistant" or role == "gpt":
                _input_id = tokenizer.encode(
                    sentence["value"], add_special_tokens=False
                )
                _target = _input_id
            else:
                raise NotImplementedError
            input_id += _input_id
            target += _target
        assert len(input_id) == len(target)
        input_id += [tokenizer.pad_token_id]
        target += [tokenizer.pad_token_id]
        input_id += [tokenizer.pad_token_id] * (max_len - len(input_id))
        target += [IGNORE_TOKEN_ID] * (max_len - len(target))
        input_ids.append(input_id[:max_len])
        targets.append(target[:max_len])

    input_ids = torch.tensor(input_ids, dtype=torch.int)
    targets = torch.tensor(targets, dtype=torch.int)
    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(SupervisedDataset, self).__init__()

        print("Formatting inputs...")
        sources = [example["conversations"] for example in raw_data]
        data_dict = preprocess(sources, tokenizer, max_len)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self, raw_data, tokenizer: transformers.PreTrainedTokenizer, max_len: int
    ):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_len = max_len

        print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i):
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        ret = preprocess(
            [self.raw_data[i]["conversations"]], self.tokenizer, self.max_len
        )
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_path,
    max_len,
    lazy_preprocess=True,
):
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = LazySupervisedDataset if lazy_preprocess else SupervisedDataset
    print("Loading data...")
    print(f"Data path {data_path}")

    train_json = json.load(open(data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer, max_len=max_len)

    eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# https://github.com/microsoft/DeepSpeed/pull/5008#issuecomment-1911564136
# https://github.com/hiyouga/LLaMA-Factory/pull/2319/files
# https://github.com/hiyouga/LLaMA-Factory/pull/2283/files
def patch_mixtral_replace_moe_impl(model, project) -> None:
    def mlp_forward(self, hidden_states):
        current_hidden_states = self.act_fn(self.w1(hidden_states)) * self.w3(
            hidden_states
        )
        current_hidden_states = self.w2(current_hidden_states)
        return current_hidden_states

    ## Ref. https://huggingface.co/deepseek-ai/deepseek-moe-16b-base/blob/main/modeling_deepseek.py
    if "MoExtendForCausalLM" in project:

        def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = self.gate(hidden_states)
            if self.use_gate_vision:
                router_logits_vision = self.gate_vision(hidden_states)
                router_logits = torch.concat(
                    [router_logits, router_logits_vision], dim=1
                )

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            _, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1, sorted=False)
            if self.use_gate_vision:
                routing_weights2 = routing_weights.to(hidden_states.dtype)
                routing_weights2 = routing_weights2 + self.weight_scale(
                    routing_weights2
                )
                topk_weight = torch.gather(routing_weights2, 1, topk_idx)
            else:
                topk_weight = torch.gather(routing_weights, 1, topk_idx)

            topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            topk_weight = topk_weight.to(hidden_states.dtype)

            hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
            y = torch.empty_like(hidden_states)
            flat_topk_idx = topk_idx.view(-1)
            for i in range(self.num_experts):
                expert = self.experts[i]
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])

            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            final_hidden_states = y.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

    else:

        def moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:

            batch_size, sequence_length, hidden_dim = hidden_states.shape
            hidden_states = hidden_states.view(-1, hidden_dim)
            router_logits = self.gate(hidden_states)

            routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
            topk_weight, topk_idx = torch.topk(
                routing_weights, self.top_k, dim=-1, sorted=False
            )
            topk_weight /= topk_weight.sum(dim=-1, keepdim=True)
            # we cast back to the input dtype
            topk_weight = topk_weight.to(hidden_states.dtype)

            hidden_states = hidden_states.repeat_interleave(self.top_k, dim=0)
            y = torch.empty_like(hidden_states)
            flat_topk_idx = topk_idx.view(-1)
            for i in range(self.num_experts):
                expert = self.experts[i]
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            final_hidden_states = y.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

    model.model.layers[0].block_sparse_moe.experts[0].__class__.forward = mlp_forward
    model.model.layers[0].block_sparse_moe.__class__.forward = moe_forward


def safe_save_model_for_hf_trainer(
    trainer: transformers.Trainer, output_dir: str, bias="none"
):
    """Collects the state dict and dump to disk."""
    # check if zero3 mode enabled
    if transformers.deepspeed.is_deepspeed_zero3_enabled():
        state_dict = trainer.model_wrapped._zero3_consolidated_16bit_state_dict()
    else:
        state_dict = trainer.model.state_dict()
    if trainer.args.should_save and trainer.args.local_rank == 0:
        gradient_state_dict = {}
        for name, param in trainer.model.named_parameters():
            if param.requires_grad == True:
                gradient_state_dict[name] = state_dict[name]

        trainer._save(output_dir, state_dict=gradient_state_dict)


def main():
    args = get_args_parser().parse_args()
    save_steps = args.save_steps
    vision_encode = args.vision_encode
    project = args.project_name
    deepspeed_config = args.deepspeed_config
    output_dir = args.output_dir
    use_reentrant = True

    if "VisionMixtralForCausalLM" in project:
        batch_size = 64
        base_model_id = "zhongshsh/vision_gate"
        data_path = "data/558k_pretrain.json"
        max_len = 1024
        learning_rate = 1e-3
        gradient_accumulation_steps = 1
        epoch = 1

    elif "MoExtendForCausalLM" in project:
        batch_size = 32
        base_model_id = f"checkpoints/vision_gate_ft"
        batch_size = 32
        data_path = "data/665k_finetune.json"
        max_len = 2048
        learning_rate = 2e-5
        gradient_accumulation_steps = 1
        epoch = 1

    if args.learning_rate != None:
        learning_rate = args.learning_rate
    if args.gradient_accumulation_steps != None:
        gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.batch_size != None:
        batch_size = args.batch_size
    if args.data_path != None:
        data_path = args.data_path
    if args.max_steps != -1:
        project += f"-step{args.max_steps}"
    if args.model_id != None:
        base_model_id = args.model_id

    print(f"Project name {project}")
    print(f"Output dir {output_dir}")

    training_args = transformers.TrainingArguments(
        optim="adamw_torch",
        bf16=True,
        fp16=False,
        num_train_epochs=epoch,
        max_steps=args.max_steps,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        do_eval=True,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=args.save_total_limit,
        learning_rate=learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=1,
        tf32=True,
        gradient_checkpointing=True,
        dataloader_num_workers=8,
        report_to="wandb",  # Comment this out if you don't want to use weights & baises
        run_name=project,  # Name of the W&B run (optional)
        logging_dir="./logs",  # Directory for storing logs
        output_dir=output_dir,
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant},
        deepspeed=deepspeed_config,
        seed=42,
        data_seed=42,
    )
    if training_args.local_rank == 0:
        time_step = datetime.now()
        time_step = time_step.astimezone(pytz.timezone("Asia/Shanghai"))
        time_step = time_step.strftime("%Y-%m-%d-%H:%M")
        wandb.init(project=project, name=f"{project}-{time_step}", resume=args.resume)

    tokenizer = Tokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    tokenizer.image_token_span = 576
    tokenizer.pad_token = tokenizer.eos_token
    config = MixtralConfig.from_pretrained(
        base_model_id,
    )
    config.pad_token_id = tokenizer.pad_token_id
    config.output_router_logits = True
    config.router_aux_loss_coef = args.router_aux_loss_coef
    config.use_cache = False
    config.test = False

    if training_args.local_rank == 0:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(f"{output_dir}/a_train_config.json", "w") as f:
            f.write(
                json.dumps(
                    {
                        "model_type": project,
                        "base_model_id": base_model_id,
                        "add_layer": config.add_layer,
                        "data_path": data_path,
                        "expert_gate": args.expert_gate,
                        "vision_encode": args.vision_encode,
                        "output_router_logits": config.output_router_logits,
                        "router_aux_loss_coef": config.router_aux_loss_coef,
                        "max_len": max_len,
                        "epoch": epoch,
                        "per_device_batch_size": batch_size,
                        "gradient_accumulation_steps": gradient_accumulation_steps,
                        "learning_rate": learning_rate,
                        "max_steps": args.max_steps,
                        "save_steps": save_steps,
                        "weight_decay": args.weight_decay,
                        "warmup_ratio": args.warmup_ratio,
                        "lr_scheduler_type": "cosine",
                    }
                )
            )

    if "VisionMixtralForCausalLM" in project:
        model = VisionMixtralForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            torch_dtype=torch.bfloat16,
        )
    elif "MoExtendForCausalLM" in project:
        model = MoExtendForCausalLM.from_pretrained(
            base_model_id,
            config=config,
            torch_dtype=torch.bfloat16,
        )

    if transformers.deepspeed.is_deepspeed_zero3_enabled():
        patch_mixtral_replace_moe_impl(model, project)

    model.train()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    model._init_vision_expert(
        vision_encode=args.vision_encode,
        expert_gate=args.expert_gate,
    )
    model.config.use_cache = (
        False  # silence the warnings. Please re-enable for inference!
    )

    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    data_module = make_supervised_data_module(
        tokenizer=tokenizer, data_path=data_path, max_len=max_len
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_state()

    safe_save_model_for_hf_trainer(trainer=trainer, output_dir=output_dir)


if __name__ == "__main__":
    main()
