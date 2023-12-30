from torch.utils.data import Dataset

class emotionDataset(Dataset):
    """Class to load the dataset and get batches of paras"""    
    def __init__(self, list_data, 
                tokenizer, max_length):
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = list_data
        self.pad_token = 1
    
    def __len__(self):
        """Return length of dataset."""
        return self.data.__len__()

    def __getitem__(self, i):
        """Return sample from dataset at index i."""
        example = self.data[i]
        inputs = self.tokenizer.encode_plus(example['text'],
                                            add_special_tokens=True,
                                            truncation=True,
                                            max_length=self.max_length)
                
        input_ids = inputs["input_ids"]
        input_ids = input_ids[:self.max_length]
        attention_mask = [1] * len(input_ids)
        
        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([self.pad_token] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        
        assert len(input_ids) == self.max_length, "Error with input length {} vs {}".format(len(input_ids), self.max_length)
        
        nli_label = example['labels'][0]
        
        return_dict = {'input_ids':torch.LongTensor(input_ids),
                    'attention_mask':torch.LongTensor(attention_mask),
                    'labels': torch.LongTensor([nli_label])}
        
        return return_dict