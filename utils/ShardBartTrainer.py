from transformers import Trainer

class ShardedBartTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_original_id = None

    def training_step(self, model, inputs, num_items_in_batch = None):
        # 传递shard_info到模型
        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            original_input_ids = inputs["original_input_ids"],
            labels=inputs["labels"],
            shard_info=inputs["shard_info"][0]  # 假设同一批次来自同一文本
        )
        return outputs.loss

    def evaluation_step(self, model, inputs, num_items_in_batch = None):
        # 评估时同理
        original_ids = [info for info in inputs.pop("shard_info")]
        if original_ids != self.current_original_id:
            model.reset_memory()
            self.current_original_id = original_ids
        return super().evaluation_step(model, inputs)