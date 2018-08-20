
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
    main(args)
    # dataset = data.text.Corpus('data/ptb')

