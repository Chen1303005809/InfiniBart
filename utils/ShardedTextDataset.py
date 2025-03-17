from torch.utils.data import IterableDataset
import random

class ShardedTextDataset(IterableDataset):
    def __init__(self, text_list, shard_length=1024, shuffle=False):
        self.text_list = text_list  # 输入的长文本列表
        self.shard_length = shard_length
        self.shuffle = shuffle  # 是否打乱文本顺序（但保持分片顺序）

    def normalize_punctuation(self, text):
        text['text'] = text['text'].replace('“', '"').replace('”', '"')  # 中文引号转英文
        text['text'] = text['text'].replace('‘', "'").replace('’', "'")
        text['text'] = text['text'].replace('—', '-')  # 统一破折号
        return text
    
    def __iter__(self):
        # 如果允许打乱，先随机排列文本顺序
        if self.shuffle:
            random.shuffle(self.text_list)
        
        for text in self.text_list:
            # 分割文本为多个分片
            self.normalize_punctuation(text)
            shards = [text['text'][i:i+self.shard_length] for i in range(0, len(text['text']), self.shard_length)]
            # 生成带元数据的分片
            for shard_idx, shard in enumerate(shards):
                yield {
                    "text": shard,
                    "original_id": id(text['text']),  # 唯一标识原始文本
                    "shard_idx": shard_idx,
                    "total_shards": len(shards)
                }