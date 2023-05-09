def set_template(args):
    if args.template is None:
        return

    else:
        args.mode = 'train'

        args.dataloader_code = 'bert'
        batch = 64
        num_workers = 8
        if args.dataset_code=='UKB':
            args.bert_dropout = 0.3
            args.tau = 0.5
            seq_len = 50
            args.num_positive = 1
            args.decay_step = 100
            args.lambda_ = 1e-5
        else:
            args.decay_step = 50
            seq_len = 1440
            args.bert_dropout = 0.1

        args.train_batch_size = batch
        args.val_batch_size = batch
        args.test_batch_size = batch

        #args.train_negative_sampler_code = 'random'

        args.train_negative_sample_size = seq_len

        #args.train_negative_sampling_seed = 56789
        #args.test_negative_sampler_code = 'random'
        #args.test_negative_sample_size = seq_len
        #args.test_negative_sampling_seed = 98765

        args.trainer_code = 'bert'
        args.device = 'cuda'
        # args.device_idx = '0'
        args.num_gpu = 1
        args.optimizer = 'Adam'
        args.lr = 0.001
        args.enable_lr_schedule = True
   
        args.gamma = 1.0

        args.num_epochs = 100

        args.model_code = 'bert'
        args.model_init_seed = 0

        args.model_sample_seed=0
        
        args.bert_hidden_units = 64
        args.bert_mask_prob = 0.15
        args.bert_max_len = seq_len
        args.bert_num_blocks = 2
        args.bert_num_heads = 4 if 'UKB' in args.dataset_code else 2

        args.slide_window_step = 50



