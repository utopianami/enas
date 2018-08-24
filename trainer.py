import controller
import numpy as np
import utils
import shared_rnn
import torch
import math
import os
from torch.autograd import Variable


class Trainer(object):
    """A class to wrap training code."""
    def __init__(self, args, dataset):

        self.args = args
        self.cuda = args.cuda
        self.dataset = dataset
        self.epoch = 0
        self.shared_step = 0
        self.controller_step = 0

        self.train_data = utils.batchify(dataset.train, args.batch_size, self.cuda)
        self.valid_data = utils.batchify(dataset.valid, args.batch_size, self.cuda)

        self.eval_data = utils.batchify(dataset.valid, args.test_batch_size, self.cuda)
        # self.test_data = utils.batchify(dataset.test, args.test_batch_size, self.cuda)

        self.max_length = self.args.shared_rnn_max_length # self.args.shared_rnn_max_length = 35

        self.shared = shared_rnn.RNN(self.args, self.dataset).to(self.args.device)
        self.controller = controller.Controller(self.args).to(self.args.device)

        self.shared_optim = torch.optim.SGD(self.shared.parameters(), lr=self.shared_lr, weight_decay=self.args.shared_l2_reg)
        self.controller_optim = torch.optim.Adam( self.controller.parameters(), lr=self.controller_lr)
        self.ce = torch.nn.CrossEntropyLoss()

    def train(self):
        for self.epoch in range(self.args.max_epoch):
            # 400 steps, each on a minibatch of 64 examples.
            print(f'train_shared, cur_shared_step: {self.shared_step}')
            # self.train_shared()


            #2000 steps, each on a minibatch of 1 examples.
            print(f'train_controller, cur_controller_step: {self.controller_step}')
            # self.train_controller()

            if self.epoch % self.args.save_epoch == self.args.save_epoch-1:
                with torch.no_grad():
                    best_dag = self.derive()
                    ppl = self.evaluate(self.eval_data, best_dag)
                    print(f'ppl: {ppl}')

            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)

    def get_loss(self, inputs, targets, dags):
        if not isinstance(dags, list):
            dags = [dags]

        loss = 0
        for dag in dags:
            output, _, extra_out = self.shared(inputs, dag, hidden=None)
            output_flat = output.view(-1, self.dataset.num_tokens)
            sample_loss = (self.ce(output_flat, targets) / self.args.shared_num_sample)
            loss += sample_loss
            ## 여기서 바로 업데이트 할 수 있음
        return loss, extra_out


    def train_shared(self):
        model = self.shared
        model.train()
        self.controller.eval()

        step = 0
        raw_total_loss = 0
        total_loss = 0
        train_idx = 0


        while train_idx < self.train_data.size(0) - 1 - 1:
            if step > self.args.shared_max_step:
                break

            dags = self.controller.sample(self.args.shared_num_sample) # args.shared_num_sample = M = 1
            inputs, targets = self.get_batch(self.train_data, train_idx, self.max_length)

            loss, extra_out = self.get_loss(inputs, targets, dags)
            raw_total_loss += loss.data

            # update
            self.shared_optim.zero_grad()
            loss.backward()

            # clip_grad: clip_coef * grad, max_norm=0.25
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.shared_grad_clip)

            self.shared_optim.step()
            total_loss += loss.data

            step += 1
            self.shared_step += 1
            train_idx += self.max_length

    def get_reward(self, dag, entropies, valid_idx=None):
        if valid_idx:
            valid_idx = 0

        inputs, targets = self.get_batch(self.valid_data, valid_idx, self.max_length)
        valid_loss, _ = self.get_loss(inputs, targets, dag)

        valid_ppl = math.exp(valid_loss.item())

        ### reward 계산 부분 다시 확인 reward_c=80
        if self.args.ppl_square:
            R = self.args.reward_c / valid_ppl ** 2
        else:
            R = self.args.reward_c / valid_ppl

        if self.args.entropy_mode == 'reward':
            rewards = R + self.args.entropy_coeff * entropies
        elif self.args.entropy_mode == 'regularizer':
            rewards = R * np.ones_like(entropies)
        else:
            raise NotImplementedError(f'Unkown entropy mode: {self.args.entropy_mode}')

        return rewards

    def train_controller(self):
        model = self.controller
        model.train()
        self.shared.eval()

        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []


        total_loss = 0
        valid_idx = 0
        for step in range(self.args.controller_max_step):
            dags, log_probs, entropies = self.controller.sample(with_details=True) # sample models
            np_entropies = entropies.data.cpu().numpy() # calculate reward

            with torch.no_grad():
                rewards = self.get_reward(dags, np_entropies, valid_idx)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay # 0.95
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline #
            adv_history.extend(adv)

            # policy loss
            loss = -log_probs * torch.Tensor(adv).to(self.args.device)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum(-1).mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            self.controller_optim.step()
            total_loss += loss.item()

            self.controller_step += 1

    def evaluate(self, source, dag, max_num=None):
        self.shared.eval()
        self.controller.eval()

        if max_num == None:
            max_num = source.size(0)
        else:
            max_num *=self.max_length
        data = source[:max_num]

        total_loss = 0
        hidden = None

        pbar = range(0, data.size(0) - 1, self.max_length)
        for count, idx in enumerate(pbar):
            inputs, targets = self.get_batch(data, idx)
            output, hidden, _ = self.shared(inputs, dag, hidden=hidden, is_train=False)
            output_flat = output.view(-1, self.dataset.num_tokens)
            total_loss += len(inputs) * self.ce(output_flat, targets).data

        val_loss = utils.to_item(total_loss) / len(data)
        ppl = math.exp(val_loss)

        return ppl

    def derive(self, sample_num=None, valid_idx=0):
        if sample_num is None:
            sample_num = self.args.derive_num_sample # args.derive_num_sample = 100

        # def sample(self, batch_size=1, with_details=False, save_dir=None):
        dags, _, entropies = self.controller.sample(with_details=True)

        max_R = 0
        best_dag = None
        for dag in dags:
            R = self.get_reward(dag, entropies, valid_idx)
            if R.sum() > max_R:
                max_R = R.sum()
                best_dag = dag

        fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                 f'{max_R:6.4f}-best.png')

        dir_path = 'sample_model/'
        path = os.path.join(dir_path, fname)
        utils.draw_network(best_dag, path)

        return best_dag

    @property
    def shared_lr(self):
        degree = max(self.epoch - self.args.shared_decay_after + 1, 0) # shared_decay_after
        return self.args.shared_lr * (self.args.shared_decay ** degree)

    @property
    def controller_lr(self):
        return self.args.controller_lr

    def get_batch(self, source, idx, length=None):
        length = min(length if length else self.max_length, len(source) - 1 - idx)
        data = source[idx:idx + length].to(self.args.device)
        target = source[idx + 1:idx + 1 + length].view(-1).to(self.args.device)
        return data, target