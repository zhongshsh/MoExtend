import torch
import torch.nn.functional as F

import json
import pandas as pd

from src import *

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = Tokenizer.from_pretrained(model_id)


def count_distribution(class_name, model_path, return_percent=False):
    print(class_name, model_path)

    with torch.no_grad():
        config = MixtralConfig.from_pretrained(model_path)
        config.test = True
        config.output_router_logits = True
        config.use_cache = False
        model = globals()[class_name].from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        data_path = "data/10k_finetune.json"
        with open(data_path, "r") as f:
            data = json.loads(f.read())
            texts = []
            answers = []
            for i in data:
                texts.append(i["conversations"][0]["value"])
                answers.append(i["conversations"][1]["value"])

        count_dict = {i: {} for i in range(32)}
        for k in count_dict.keys():
            for i in range(8):
                count_dict[k][i] = 0

        for text_id, text in enumerate(texts):
            inputs = tokenizer(text, return_tensors="pt")
            outputs = model(**inputs, return_dict=True)
            for layer_id, gate_logit in enumerate(outputs.router_logits):
                routing_weights = F.softmax(gate_logit, dim=-1, dtype=torch.float)
                selected_experts = torch.argsort(routing_weights, dim=-1)[..., :2]
                for expert_id in range(8):
                    count_expert = torch.sum(
                        torch.eq(selected_experts, expert_id)
                    ).item()
                    count_dict[layer_id][expert_id] += count_expert

    del model

    if return_percent:
        for i in count_dict.keys():
            count_dict[i] = dict(sorted(count_dict[i].items(), key=lambda x: x[0]))
            count = 0
            for k in count_dict[i].keys():
                count += count_dict[i][k]
            for k in count_dict[i].keys():
                count_dict[i][k] = (
                    "{:.2f}".format(count_dict[i][k] / count * 100).rjust(5) + "%"
                )
        return count_dict

    data = pd.DataFrame(count_dict)
    data = data / data.sum(0)

    return data


org_distribute = count_distribution("1", "checkpoints/vision")
gate_distribute = count_distribution("2", "checkpoints/vision_gate")

result = (gate_distribute - org_distribute).std()
selected_layers = (
    result.sort_values(ascending=False).index[: len(result) // 2].sort_values()
)
print("Selected layers: ", selected_layers)
print("Number of selected layers: ", selected_layers.shape[0])

expert_idxmax = gate_distribute.idxmax().T[selected_layers].values
print("Init Expert id: ", expert_idxmax)
