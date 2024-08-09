import torch
from src import *


def save_model(class_name, model_path, save_path, add_layer, expert_idx):
    layer2expert = {}
    for (
        i,
        v,
    ) in enumerate(add_layer):
        layer2expert[v] = expert_idx[i]

    with torch.no_grad():
        config = MixtralConfig.from_pretrained(
            model_path,
        )
        config.add_layer = add_layer
        model = globals()[class_name].from_pretrained(
            model_path, torch_dtype=torch.bfloat16, device_map="auto", config=config
        )
        model.eval()
        model._tran_vision_expert(layer2expert=layer2expert)
        model.save_pretrained(save_path)

    with open(f"{save_path}/a_model_info.txt", "w") as f:
        f.write(f"class_name: {class_name}\n")
        f.write(f"org_model_path: {model_path}\n")
        f.write(f"add_expert_layer: {add_layer}\n")
        f.write(f"init_expert_idx: {expert_idx}\n")


class_name = "MoExtendForCausalLM"
model_path = "checkpoints/vision_gate"
save_path = "checkpoints/vision_gate_ft"
add_layer = [2, 3, 5, 6, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 25, 27]
expert_idx = [1, 3, 3, 1, 0, 7, 2, 7, 0, 5, 5, 3, 0, 3, 3, 1]
save_model(
    class_name, model_path, save_path, add_layer=add_layer, expert_idx=expert_idx
)
