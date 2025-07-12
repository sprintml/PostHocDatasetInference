import os
from datasets import load_dataset, concatenate_datasets

class HFDataloader:
    def __init__(self, data_dir, dataset_name, split, subset_name=None, text_length=None):
        '''
            Load dataset with huggingface API.
            Also keep in track the column name of texts content as self.text_column.
        '''
        self.dataset_name = dataset_name
        self.subset_name = subset_name
        self.split = split
        if dataset_name == 'pile':
            if split not in ['train', 'val', 'test', 'val+test']:
                # load from self-defined datasets
                self.dataset = load_dataset("json", data_files=os.path.join(data_dir, f'datasets/{dataset_name}_{subset_name}/{split}.jsonl'), split="train")
                self.text_column_name = 'original'
            else:
                # load from huggingface datasets
                if split == 'train':
                    # self.dataset = load_dataset("monology/pile-uncopyrighted", split='train')
                    dataset_a = load_dataset("json", data_files="https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/00.jsonl.zst", split="train")
                    dataset_b = load_dataset("json", data_files="https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/train/01.jsonl.zst", split="train")
                    self.dataset = concatenate_datasets([dataset_a, dataset_b])
                elif split == 'val':
                    self.dataset = load_dataset("json", data_files="https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst", split="train")
                elif split == 'test':
                    self.dataset = load_dataset("json", data_files="https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/test.jsonl.zst", split="train")
                elif split == 'val+test':
                    val_dataset = load_dataset("json", data_files="https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/val.jsonl.zst", split="train")
                    test_dataset = load_dataset("json", data_files="https://huggingface.co/datasets/monology/pile-uncopyrighted/resolve/main/test.jsonl.zst", split="train")
                    self.dataset = concatenate_datasets([val_dataset, test_dataset])
                # filter certain subset
                if subset_name is not None:
                    self.dataset = self.dataset.filter(lambda data: data['meta']['pile_set_name']==subset_name)
                self.text_column_name = 'text'
        elif dataset_name == 'wikimia':
            if text_length is not None:
                text_description = f'WikiMIA_length{text_length}'
                self.dataset = load_dataset("swj0419/WikiMIA", split=text_description)
            self.text_column_name = 'input'
        elif dataset_name == 'bookmia':
            if split == 'train':
                self.dataset = load_dataset("swj0419/BookMIA", split='train')
            else:
                raise Exception(f'Unknown split {split} for dataset BookMIA')
            self.text_column_name = 'snippet'
        elif dataset_name == 'cnn_dailymail':
            if split not in ['train', 'val', 'test', 'val+test']:
                # load from self-defined datasets
                self.dataset = load_dataset("json", data_files=os.path.join(data_dir, f'datasets/{dataset_name}/{split}.jsonl'), split="train")
                self.text_column_name = 'original'
            else:
                if split in ['train', 'test']:
                    self.dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
                elif split == 'val':
                    self.dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split='validation')
                else:
                    raise Exception(f'Unknown split {split} for dataset CNN/DailyMail')
        elif dataset_name == 'cc_news':
            if split not in ['train', 'val', 'test', 'val+test']:
                # load from self-defined datasets
                self.dataset = load_dataset("json", data_files=os.path.join(data_dir, f'datasets/{dataset_name}/{split}.jsonl'), split="train")
                self.text_column_name = 'original'
            else:
                if split in ['train', 'test']:
                    self.dataset = load_dataset("vblagoje/cc_news", split=split)
                elif split == 'val':
                    self.dataset = load_dataset("vblagoje/cc_news", split='validation')
                else:
                    raise Exception(f'Unknown split {split} for dataset CC News')
        elif dataset_name in ['timothy_sykes']:
            self.dataset = load_dataset("json", data_files=os.path.join(data_dir, f'datasets/{dataset_name}/{split}.jsonl'), split="train")
        elif dataset_name.startswith('dolma'):
            assert split not in ['train', 'val', 'test', 'val+test']
            # load from self-defined datasets
            self.dataset = load_dataset("json", data_files=os.path.join(data_dir, f'datasets/{dataset_name}_{subset_name}/{split}.jsonl'), split="train")
            self.text_column_name = 'original'
        else:
            raise Exception(f'Unknown dataset name: {dataset_name}')

    def get_dataset(self, n_sample=-1, data_offset_idx=0, shuffle=False):
        if n_sample < 0:
            return_dataset =  self.dataset
        elif n_sample+data_offset_idx > len(self.dataset):
            print(f'Requesting more data than the {self.dataset_name}-{self.subset_name}-{self.split} dataset size. Dataset size: {len(self.dataset)}')
            return_dataset = self.dataset.select(range(data_offset_idx, len(self.dataset)))
        else:
            return_dataset = self.dataset.select(range(data_offset_idx, n_sample+data_offset_idx))
        if shuffle:
            return return_dataset.shuffle()
        else:
            return return_dataset

if __name__ == '__main__':
    dataloader = HFDataloader('bookmia', 'train')
    dataset = dataloader.get_dataset(100)
    for i, data in enumerate(dataset):
        print(i, data)