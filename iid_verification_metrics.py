import os
# os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import json
from iid_metrics.char_ngram import CharNGramSelector
from iid_metrics.word_ngram import WordNGramSelector
from iid_metrics.bag_of_words import BoWClassification
from iid_metrics.gpt2_xl import GPT2Classification
from data_loader.hf_dataloader import HFDataloader


def main(args):
    # load datasets
    if args.dataset_name in ['wikimia', 'bookmia']:
        text_part_name_mapping = {
            'wikimia': 'input',
            'bookmia': 'snippet',
        }
        suspect_data = []
        test_data = []
        dataloader = HFDataloader(dataset_name=args.dataset_name, split=args.split, text_length=args.text_length)
        dataset = dataloader.get_dataset()
        for _, json_data in enumerate(dataset):
            text = json_data[text_part_name_mapping[args.dataset_name]]
            label = json_data['label']
            suspect_data.append(text) if label else test_data.append(text)

    elif args.dataset_name in ['pile', 'timothy_sykes']:
        # read original and synthetic data
        root = os.path.join(args.data_dir, "datasets")
        if args.dataset_name == 'pile':
            val_pair_dir = os.path.join(root, f'{args.dataset_name}_{args.subset_name}')
        else:
            val_pair_dir = os.path.join(root, f'{args.dataset_name}')
        os.makedirs(val_pair_dir, exist_ok=True)
        suspect_data = []
        test_data = []
        file_name = os.path.join(val_pair_dir, f'{args.split}.jsonl')
        with open(file_name, 'r', encoding="utf-8") as json_file:
            for line in json_file:
                json_data = json.loads(line)        
                suspect_data.append(json_data['original'])
                if args.aligned_type is None and args.test_split is None:
                    test_data.append(json_data['paraphrase'][0])
        # load test data from another file
        assert args.aligned_type is None or args.test_split is None # cannot assign both aligned file and test file
        # if assign aligned file
        if args.aligned_type is not None:
            aligned_suffix = args.aligned_type if args.aligned_type == 'aligned' else args.aligned_type+f'_{args.embed_model_name}'
            aligned_file_name = os.path.join(val_pair_dir, f'{args.split}_{aligned_suffix}.jsonl')
            with open(aligned_file_name, 'r', encoding="utf-8") as json_file:
                for line in json_file:
                    json_data = json.loads(line)        
                    test_data.append(json_data['text'])
        # if assign test file
        elif args.test_split is not None:
            test_file_name = os.path.join(val_pair_dir, f'{args.test_split}.jsonl')
            with open(test_file_name, 'r', encoding="utf-8") as json_file:
                for line in json_file:
                    json_data = json.loads(line)
                    if args.test_split_column == 'original':
                        test_data.append(json_data['original'])
                    else:
                        test_data.append(json_data['paraphrase'][0])
        

    else:
        raise Exception(f'Unknown dataset: {args.dataset_name}')

    # analyze IID
    if args.metric in ['char_ngram', 'word_ngram_member', 'word_ngram_non_member']:
        n_train = int(len(suspect_data) * 0.8)
        train_members = suspect_data[0:n_train]
        train_non_members = test_data[0:n_train]
        test_members = suspect_data[n_train:]
        test_non_members = test_data[n_train:]

        # Initialize the selector
        if args.metric == 'char_ngram':
            selector = CharNGramSelector(max_n=1, top_k=150)
        elif args.metric == 'word_ngram_member':
            selector = WordNGramSelector(max_n=1, top_k=10, use_member_word=True)
        elif args.metric == 'word_ngram_non_member':
            selector = WordNGramSelector(max_n=1, top_k=250, use_member_word=False)
        else:
            raise Exception(f'Unknown ngram type: {args.metric}')

        # Select the best n-grams
        selector.select_ngrams(train_members, train_non_members)

        # Evaluate on the test set
        tpr, fpr = selector.evaluate(test_members, test_non_members)
        print(f"TPR: {tpr}, FPR: {fpr}")

    elif args.metric == 'date':
        print('To be implemented..')

    elif args.metric == 'bag_of_words':

         # Initialize the model
        model = BoWClassification(hidden_dim=50, batch_size=args.batch_size, epochs=5, lr=0.001, threshold=0.999)

        # Train the model and perform 10-fold cross-validation
        average_tpr, average_fpr, average_auc = model.train(suspect_data, test_data)
        # average_tpr, average_fpr = model.train(suspect_data[0:int(len(suspect_data)/2)], suspect_data[int(len(suspect_data)/2):])
        # average_tpr, average_fpr = model.train(test_data[0:int(len(test_data)/2)], test_data[int(len(test_data)/2):])
        print(f'Average TPR: {average_tpr*100}%, Average FPR: {average_fpr*100}%, Average AUC: {average_auc*100}%')
    
    elif args.metric == 'gpt2_xl':
        classifier = GPT2Classification(model_name='gpt2-xl', num_labels=2, max_seq_len=args.max_seq_len, batch_size=args.batch_size)
        # Cross-validation
        average_tpr, average_fpr, average_auc = classifier.cross_validate(suspect_data, test_data, num_splits=5)
        # average_tpr, average_fpr, average_auc = classifier.cross_validate(suspect_data[0:len(suspect_data)//2], suspect_data[len(suspect_data)//2:], num_splits=5)
        print(f'Average TPR: {average_tpr*100}%, Average FPR: {average_fpr*100}%, Average AUC: {average_auc*100}%')

    elif args.metric == 'mauve':
        print('To be implemented..')

    else:
        raise Exception(f'Unimplemented metric: {args.metric}')
    



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="/storage2/bihe/llm_data_detect")
    parser.add_argument("--dataset_name", type=str, default="pile", help="for example, pile, wikimia', bookmia")
    parser.add_argument("--aligned_type", type=str, default=None, choices=['aligned', 'pair_aligned', None], help='Use the aligned synthesized data as test data. Set to None if you do not use aligned data.')
    parser.add_argument("--embed_model_name", type=str, default="all-MiniLM-L6-v2", help='For pairwise alignment, please indicate a sentence embedding model name.')
    parser.add_argument("--subset_name", type=str, default="Wikipedia (en)", help='Specify the subset name for PILE dataset')
    parser.add_argument("--text_length", type=int, default=64, choices=[128, 256, 32, 64], help='Specify the text snippet length for WikiMIA dataset')
    parser.add_argument("--max_seq_len", type=int, default=128, help='Length of the sequences used for classification analysis.')
    parser.add_argument("--split", type=str, default="train_original_16384_llama_pile_train_original_16384_100epoch_paraphrase", help='For aligned data, input the file name for the original data and indicate align type in the align domain')
    parser.add_argument("--test_split", type=str, default=None, help='Get another data file for evaluation. Have to assign test split column')
    parser.add_argument("--test_split_column", type=str, default='original', choices=['original', 'paraphrase'], help='the column for the test split')
    # parser.add_argument("--paraphrase_model", type=str, default="llama")
    # parser.add_argument("--n_sample", type=int, default=5000)
    parser.add_argument("--metric", type=str, default="gpt2_xl", choices=['gpt2_xl', 'char_ngram','word_ngram_member','word_ngram_non_member','bag_of_words'])
    parser.add_argument("--batch_size", type=int, default=12, help='Batch size for running classification evaluation.')
    # parser.add_argument("--n_val", type=int, default=600)
    # parser.add_argument("--device", type=int, default=3)
    args = parser.parse_args()

    print(args, flush=True)
    
    main(args)
