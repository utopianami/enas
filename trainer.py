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
        self.test_data = utils.batchify(dataset.test, args.test_batch_size, self.cuda)

        self.max_length = self.args.shared_rnn_max_length # self.args.shared_rnn_max_length = 35

        self.shared = shared_rnn.RNN(self.args, self.dataset).to(self.args.device)
        self.controller = controller.Controller(self.args).to(self.args.device)

        shared_optimizer = torch.optim.SGD
        controller_optimizer = torch.optim.Adam

        self.shared_optim = shared_optimizer( self.shared.parameters(), lr=self.shared_lr, weight_decay=self.args.shared_l2_reg)
        self.controller_optim = controller_optimizer( self.controller.parameters(), lr=self.args.controller_lr)
        self.ce = torch.nn.CrossEntropyLoss()

    def train(self):
        for self.epoch in range(self.args.max_epoch):
            # 1. Training the shared parameters omega of the child models
            # 400 steps, each on a minibatch of 64 examples.
            self.train_shared()

            # 2. Training the controller parameters theta
            #2000 steps, each on a minibatch of 1 examples.
            self.train_controller()

            # if self.epoch % self.args.save_epoch == 0:
            #     with torch.no_grad():
            #         best_dag = self.derive()
            #         self.evaluate(self.eval_data, best_dag, 'val_best', max_num=self.args.batch_size*100)
            #     self.save_model()

            if self.epoch >= self.args.shared_decay_after:
                utils.update_lr(self.shared_optim, self.shared_lr)

    def get_loss(self, inputs, targets, hidden, dags):
        if not isinstance(dags, list):
            dags = [dags]

        loss = 0
        for dag in dags:
            output, hidden, extra_out = self.shared(inputs, dag, hidden=hidden)
            output_flat = output.view(-1, self.dataset.num_tokens)
            sample_loss = (self.ce(output_flat, targets) / self.args.shared_num_sample)
            loss += sample_loss

        assert len(dags) == 1, 'there are multiple `hidden` for multple `dags`'
        return loss, hidden, extra_out


    def train_shared(self, max_step=400):
        model = self.shared
        model.train()
        self.controller.eval()

        abs_max_grad = 0
        abs_max_hidden_norm = 0
        step = 0
        raw_total_loss = 0
        total_loss = 0
        train_idx = 0


        while train_idx < self.train_data.size(0) - 1 - 1:
            if step > max_step:
                break

            dags = self.controller.sample(self.args.shared_num_sample) # args.shared_num_sample = M = 1
            inputs, targets = self.get_batch(self.train_data, train_idx, self.max_length)

            hidden = None # first
            loss, _, extra_out = self.get_loss(inputs, targets, hidden, dags)
            raw_total_loss += loss.data

            ## loss += _apply_penalties(extra_out, self.args) 추후 사용

            # update
            self.shared_optim.zero_grad()
            loss.backward()

            # h1tohT = extra_out['hiddens']
            # new_abs_max_hidden_norm = utils.to_item(
            #     h1tohT.norm(dim=-1).data.max())
            # if new_abs_max_hidden_norm > abs_max_hidden_norm:
            #     abs_max_hidden_norm = new_abs_max_hidden_norm
            #     logger.info(f'max hidden {abs_max_hidden_norm}')
            # abs_max_grad = _check_abs_max_grad(abs_max_grad, model)
            # torch.nn.utils.clip_grad_norm(model.parameters(),
            #                               self.args.shared_grad_clip)

            self.shared_optim.step()
            total_loss += loss.data

            step += 1
            self.shared_step += 1
            train_idx += self.max_length

    def get_reward(self, dag, entropies, hidden, valid_idx=None):
        """Computes the perplexity of a single sampled model on a minibatch of
        validation data.
        """
        if not isinstance(entropies, np.ndarray):
            entropies = entropies.data.cpu().numpy()

        if valid_idx:
            valid_idx = 0

        inputs, targets = self.get_batch(self.valid_data, valid_idx, self.max_length, volatile=True)
        valid_loss, hidden, _ = self.get_loss(inputs, targets, hidden, dag)
        valid_loss = utils.to_item(valid_loss.data)

        valid_ppl = math.exp(valid_loss)

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

        return rewards, hidden

    def train_controller(self):
        model = self.controller
        model.train()
        self.shared.eval()

        avg_reward_base = None
        baseline = None
        adv_history = []
        entropy_history = []
        reward_history = []


        total_loss = 0
        valid_idx = 0
        for step in range(self.args.controller_max_step):
            # sample models

            dags, log_probs, entropies = self.controller.sample(with_details=True)
            # calculate reward
            np_entropies = entropies.data.cpu().numpy()

            with torch.no_grad():
                hidden = None # first
                rewards, hidden = self.get_reward(dags, np_entropies, hidden, valid_idx)

            reward_history.extend(rewards)
            entropy_history.extend(np_entropies)

            # moving average baseline
            if baseline is None:
                baseline = rewards
            else:
                decay = self.args.ema_baseline_decay
                baseline = decay * baseline + (1 - decay) * rewards

            adv = rewards - baseline
            adv_history.extend(adv)

            # policy loss
            loss = -log_probs*utils.get_variable(adv, self.cuda, requires_grad=False)
            if self.args.entropy_mode == 'regularizer':
                loss -= self.args.entropy_coeff * entropies

            loss = loss.sum()  # or loss.mean()

            # update
            self.controller_optim.zero_grad()
            loss.backward()

            self.controller_optim.step()
            total_loss += utils.to_item(loss.data)

            self.controller_step += 1

            print(adv)

    def evaluate(self, source, dag, name, batch_size=1, max_num=None):
        self.shared.eval()
        self.controller.eval()

        data = source[:max_num*self.max_length]

        total_loss = 0
        hidden = None

        pbar = range(0, data.size(0) - 1, self.max_length)
        for count, idx in enumerate(pbar):
            inputs, targets = self.get_batch(data, idx, volatile=True)
            output, hidden, _ = self.shared(inputs,
                                            dag,
                                            hidden=hidden,
                                            is_train=False)
            output_flat = output.view(-1, self.dataset.num_tokens)
            total_loss += len(inputs) * self.ce(output_flat, targets).data
            ppl = math.exp(utils.to_item(total_loss) / (count + 1) / self.max_length)

        val_loss = utils.to_item(total_loss) / len(data)
        ppl = math.exp(val_loss)

    def derive(self, sample_num=None, valid_idx=0):
        hidden = None

        if sample_num is None:
            sample_num = self.args.derive_num_sample # args.derive_num_sample = 100

        dags, _, entropies = self.controller.sample(sample_num, with_details=True)

        max_R = 0
        best_dag = None
        for dag in dags:
            R, _ = self.get_reward(dag, entropies, hidden, valid_idx)
            if R.max() > max_R:
                max_R = R.max()
                best_dag = dag

        fname = (f'{self.epoch:03d}-{self.controller_step:06d}-'
                 f'{max_R:6.4f}-best.png')
        path = os.path.join(self.args.model_dir, 'networks', fname)
        utils.draw_network(best_dag, path)

        return best_dag

    @property
    def shared_lr(self):
        degree = max(self.epoch - self.args.shared_decay_after + 1, 0)
        return self.args.shared_lr * (self.args.shared_decay ** degree)

    @property
    def controller_lr(self):
        return self.args.controller_lr

    def get_batch(self, source, idx, length=None, volatile=False):
        length = min(length if length else self.max_length, len(source) - 1 - idx)
        data = Variable(source[idx:idx + length], volatile=volatile)
        target = Variable(source[idx + 1:idx + 1 + length].view(-1), volatile=volatile)
        return data, target