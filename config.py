class Config(object):
    def __init__(self, args):
        self.args = args

        # dataset
        self.dataset = args.dataset

        # train hyper parameters
        self.multi_gpu = args.multi_gpu
        self.learning_rate = args.lr
        self.batch_size = args.batch_size
        self.max_epoch = args.max_epoch
        self.max_len = args.max_len

        # path and name
        self.bert_model_name = args.bert_model_name
        self.train_path = 'data/' + self.dataset + '/train_triples.json'
        self.dev_path = 'data/' + self.dataset + '/dev_triples.json'
        self.test_path = 'data/' + self.dataset + '/test_triples.json'  # overall test
        self.rel_dict_path = 'data/' + self.dataset + '/rel2id.json'
        self.save_weights_dir = 'saved_weights/' + self.dataset + '/'
        self.save_logs_dir = 'saved_logs/' + self.dataset + '/'
        self.result_dir = 'results/' + self.dataset
        self.weights_save_name = args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(
            self.learning_rate) + "_BS_" + str(self.batch_size)
        self.log_save_name = 'LOG_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(
            self.learning_rate) + "_BS_" + str(self.batch_size)
        self.result_save_name = 'RESULT_' + args.model_name + '_DATASET_' + self.dataset + "_LR_" + str(
            self.learning_rate) + "_BS_" + str(self.batch_size) + ".json"

        # log setting
        self.period = args.period
        self.test_epoch = args.test_epoch
