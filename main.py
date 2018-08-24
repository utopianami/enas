
import torch

import data
import config
import utils
import trainer



def main(args):
    torch.manual_seed(args.random_seed)
    if args.num_gpu > 0:
        torch.cuda.manual_seed(args.random_seed)

    dataset = data.text.Corpus('data/ptb')

    trnr = trainer.Trainer(args, dataset)
    trnr.train()



if __name__ == "__main__":
    args, unparsed = config.get_args()
    args.shared_max_step = 1
    args.controller_max_step = 1
    args.max_epoch = 2
    args.save_epoch=1
    main(args)
    print('end')
    # dataset = data.text.Corpus('data/ptb')

