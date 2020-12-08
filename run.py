from fastNLP import Trainer, LossFunc
from data_loader import load_data
from model import CasRel
from utils import get_tokenizer
import argparse

parser = argparse.ArgumentParser(description='Model Controller')
parser.add_argument('--train', default=False, type=bool, help='to train the HBT model, python run.py --train=True')
parser.add_argument('--dataset', default='WebNLG', type=str,
                    help='specify the dataset from ["NYT","WebNLG"]')
args = parser.parse_args()

if __name__ == '__main__':
    # pre-trained bert model name
    bert_model_name = 'en-base-cased'

    vocab_path = 'data/vocab.txt'
    # load dataset
    # dataset = args.dataset
    dataset = "NYT"
    train_path = 'data/' + dataset + '/train_triples.json'
    dev_path = 'data/' + dataset + '/dev_triples.json'
    test_path = 'data/' + dataset + '/test_triples.json'  # overall test
    # test_path = 'data/' + dataset + '/test_split_by_num/test_triples_5.json' # ['1','2','3','4','5']
    # test_path = 'data/' + dataset + '/test_split_by_type/test_triples_seo.json' # ['normal', 'seo', 'epo']
    rel_dict_path = 'data/' + dataset + '/rel2id.json'
    save_weights_path = 'saved_weights/' + dataset + '/best_model.weights'
    save_logs_path = '/saved_logs/' + dataset + '/log'

    # parameters
    LR = 1e-5
    tokenizer = get_tokenizer(vocab_path)
    data_bundle, rel_vocab, num_rels = load_data(train_path, dev_path, test_path, rel_dict_path)
    model = CasRel(bert_model_name, num_rels)

    if args.train:
        BATCH_SIZE = 6
        EPOCH = 100
        MAX_LEN = 100
        STEPS = len(data_bundle.get_dataset('train')) // BATCH_SIZE
        metric = SpanFPreRecMetric()
        optimizer = Adam(lr=LR)
        trainer = Trainer(data_bundle.get_dataset('train'), model, optimizer=optimizer, batch_size=BATCH_SIZE,
                          metrics=metric)
