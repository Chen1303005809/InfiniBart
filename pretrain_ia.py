import transformers
import torch
import os

from transformers import BartConfig, AutoTokenizer
from datasets import load_dataset
from utils.utils import sentence_permutation, document_rotation
from utils.utils import token_infilling, token_masking, token_deletion
from InfiniBART import InfiniBART
from utils.ShardedTextDataset import ShardedTextDataset
from utils.ShardBartTrainer import ShardedBartTrainer

import random


# PARAMETERS BART BASE
# ==============================================================================
VOCAB_SIZE = 52000
MAX_POSITION_EMBEDDINGS = 1024
ENCODER_LAYERS = 6
ENCODER_FFN_DIM = 3072
ENCODER_ATTENTION_HEADS = 12
DECODER_LAYERS = 6
DECODER_FFN_DIM = 3072
DECODER_ATTENTION_HEADS = 12
D_MODEL = 768
DROPOUT = 0.1
# ==============================================================================
# PARAMETERS 

# 手动划分训练集和验证集
def split_streaming_dataset(dataset, train_ratio=0.9):
    train_data = []
    validation_data = []
    for i, example in enumerate(dataset):
        if i % 10 < int(10 * train_ratio):  # 90% 训练集，10% 验证集
            train_data.append(example)
        else:
            validation_data.append(example)
    return train_data, validation_data


# 使用预训练好的分词器
tokenizer = AutoTokenizer.from_pretrained("fnlp/bart-base-chinese")

model = InfiniBART(
    BartConfig(
        vocab_size=VOCAB_SIZE,
        max_position_embeddings=MAX_POSITION_EMBEDDINGS,
        encoder_layers=ENCODER_LAYERS,
        encoder_ffn_dim=ENCODER_FFN_DIM,
        encoder_attention_heads=ENCODER_ATTENTION_HEADS,
        decoder_layers=DECODER_LAYERS,
        decoder_ffn_dim=DECODER_FFN_DIM,
        decoder_attention_heads=DECODER_ATTENTION_HEADS,
        d_model=D_MODEL,
        dropout=DROPOUT,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        is_encoder_decoder=True,
        decoder_start_token_id=tokenizer.eos_token_id,
    )
)

dataset = load_dataset(
    "./cleaned_datas", split="train", streaming=True
).with_format(type="torch")
train_dataset, eval_dataset = split_streaming_dataset(dataset)
# 使用工具进行分片，分片长度为1024
train_dataset = ShardedTextDataset(train_dataset, shard_length=1024)
eval_dataset = ShardedTextDataset(eval_dataset, shard_length=1024)

# 扰动方式
perturbations = [
    document_rotation,
    sentence_permutation,
    token_infilling,
    token_masking,
    token_deletion,
]
#文本扰动
perturbations_text_domain = [
    document_rotation,
    sentence_permutation,
]
#token扰动
perturbations_token_domain = [
    token_infilling,
    token_masking,
    token_deletion,
]

#扰动执行函数
def collate_fn(examples):
    # 按 original_id 和 shard_idx 排序，确保同一文本的分片连续且有序
    sorted_examples = sorted(examples, key=lambda x: (x["original_id"], x["shard_idx"]))
    
    # 该分片信息无法被嵌入从而传递到模型内部, 所以后续对传递的文本信息进行简化
    # shard_info = [{
    #     "original_id": ex["original_id"],
    #     "shard_idx": ex["shard_idx"],
    #     "total_shards": ex["total_shards"]
    # } for ex in sorted_examples]

    # 获取破坏前的原始文本的每个分片
    original_shard_texts = [ex["text"] for ex in sorted_examples]

    # 添加噪声（示例：随机替换Token）
    input_ids = None
    for text in original_shard_texts:
        # 随机选择扰动方法
        perturbation_function = random.choice(perturbations)

        if perturbation_function in perturbations_text_domain:
            # 分片级文本扰动（无需截断，分片已预分割）---
            # 直接处理分片文本（假设分片长度已适配模型）
            perturbed_text = perturbation_function(text)
            try:
                perturbed_input_ids = tokenizer(
                    perturbed_text, 
                    return_tensors="pt", 
                    padding="longest", 
                    truncation=True, 
                    max_length=1024  # 使用模型最大长度
                )["input_ids"][0]
            except Exception as e:
                print(f"触发异常的扰动文本: {perturbed_text}")
                print(f"异常原文本: {repr(text)}")
                raise e
        else:
            # 分片级Token扰动 ---
            # 编码原始分片（不截断）
            original_input_ids = tokenizer(
                text, 
                return_tensors="pt", 
                padding="longest", 
                truncation=True, 
                max_length=1024
            )["input_ids"][0]
            
            # 应用Token级扰动
            perturbed_input_ids = perturbation_function(
                tokenized_sequence=original_input_ids,
                mask_token_id=tokenizer.mask_token_id,
                mask_probability=0.15,
                list_special_tokens=tokenizer.all_special_ids,
            )

            # 动态填充到当前批次最大长度
            current_max_length = perturbed_input_ids.size(-1)
            if input_ids is not None:
                current_max_length = max(current_max_length, input_ids.size(-1))
            
            # 统一填充到当前批次最大长度
            pad_length = current_max_length - perturbed_input_ids.size(-1)
            if pad_length > 0:
                perturbed_input_ids = torch.cat([
                    perturbed_input_ids,
                    torch.full((pad_length,), tokenizer.pad_token_id, dtype=torch.long)
                ])
        # 累加批次
        if input_ids is None:
            input_ids = perturbed_input_ids.unsqueeze(0)
        else:
            # 对齐长度（填充或截断）
            if perturbed_input_ids.size(-1) < input_ids.size(-1):
                pad = torch.full((input_ids.size(-1) - perturbed_input_ids.size(-1),), 
                               tokenizer.pad_token_id, dtype=torch.long)
                perturbed_input_ids = torch.cat([perturbed_input_ids, pad])
            elif perturbed_input_ids.size(-1) > input_ids.size(-1):
                perturbed_input_ids = perturbed_input_ids[:input_ids.size(-1)]
            
            input_ids = torch.cat([input_ids, perturbed_input_ids.unsqueeze(0)], dim=0)

    # 生成注意力掩码，该掩码只是对填充部分的遮掩
    attention_mask = (input_ids != tokenizer.pad_token_id).int()
    
    # 标签生成：原始分片文本编码（无需扰动）
    labels = tokenizer(
        original_shard_texts,
        padding="longest",
        truncation=True,
        max_length=1024,
        return_tensors="pt"
    )["input_ids"]
    
    # original_id是由工具类返回的文本标识，仅将文本标识进行张量化
    original_ids = torch.tensor([ex["original_id"] for ex in sorted_examples], dtype=torch.long)

    return {
        "input_ids": input_ids, # 扰动后嵌入
        "original_input_ids": labels,# 扰动前嵌入，用于片段关注
        "attention_mask": attention_mask,
        "labels": labels,# 扰动前嵌入，用于自监督训练
        "shard_info": original_ids  # 分片信息只传递每段文本的唯一标识，以此来判断分片是否属于同一段文本
    }



def main():
    # 先用较短步数训练
    training_args = transformers.TrainingArguments(
        output_dir="./bart-zh-size-s",
        overwrite_output_dir=True,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=16,
        warmup_steps=3000,
        weight_decay=0.01,
        save_strategy="steps",
        eval_strategy="steps",
        max_steps=15_0000,
        logging_dir="./logs-bart-zh-size-s",
        logging_steps=10000,
        eval_steps=50000,
        save_steps=50000,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
        fp16=True,
        dataloader_num_workers=8,
        learning_rate=5e-5,
        gradient_checkpointing=True,
        max_grad_norm=1.0, 
        gradient_accumulation_steps = 4,
    )

    # 初始化自定义训练器
    trainer = ShardedBartTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

    # 评估
    print(trainer.evaluate(eval_dataset))

    trainer.save_model("./model")

if __name__ == '__main__':
    main()



