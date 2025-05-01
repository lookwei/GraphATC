

import torch
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class Evaluator:
    def __init__(self, train_loader, val_loader, record_epoch, log_path, writer, level, **kwargs):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = None
        self.epoch = None
        self.record_epoch = record_epoch
        self.device = train_loader.dataset.device
        self.writer = writer
        self.level = level
        self.kwargs = kwargs

        if writer:
            assert isinstance(kwargs['val_index'], list)
            if len(kwargs['val_index']) > 1:
                self.log_path = log_path + f"/{kwargs['val_index'][0]}-{kwargs['val_index'][-1]}"
            else:
                self.log_path = log_path + f"/{kwargs['val_index'][0]}"
            self.summary_writer = SummaryWriter(self.log_path)

        self.train_positive_num = len(self.train_loader.sampler)
        self.val_positive_num = len(self.val_loader.sampler)

        self.train_loss = None
        self.sample_num = None

        self.val_predict_opt = pd.DataFrame()

    def set_epoch(self, epoch):

        if self.epoch != epoch:
            self.epoch = epoch
            self.train_loss = 0.0
            self.sample_num = 0

    def receive(self, running_batch_size, train_loss):
        self.sample_num += running_batch_size
        self.train_loss += train_loss

    def write_train_loss(self):
        self.summary_writer.add_scalar('loss/train', self.train_loss / self.sample_num, self.epoch + 1)

    @classmethod
    def get_5_metric(cls, predict_multi_label, multi_label, multi_label_length, suffix=''):
        predict_multi_label = torch.tensor(predict_multi_label)
        multi_label = torch.tensor(multi_label)
        intersection = multi_label & predict_multi_label
        union = multi_label | predict_multi_label
        aiming = (intersection.sum() / (predict_multi_label.sum() + 1e-6)).item()
        coverage = (intersection.sum() / multi_label.sum()).item()
        accuracy = (intersection.sum() / union.sum()).item()
        abs_true = int(multi_label.equal(predict_multi_label))
        abs_false = float((union.sum() - intersection.sum()).item() / multi_label_length)
        return {
            "aiming" + suffix: aiming,
            "coverage" + suffix: coverage,
            "accuracy" + suffix: accuracy,
            "abs_true" + suffix: abs_true,
            "abs_false" + suffix: abs_false
        }

    def eval_model(self, model):
        model.eval()
        with torch.no_grad():
            eval_bar = tqdm(self.val_loader, total=len(self.val_loader), desc='[ EVAL]', unit='batch', position=4,
                            leave=False, colour='blue', ncols=60)
            val_index = self.kwargs['val_index']
            val_pre_list, val_true_list, drug_id_list = [], [], []
            for _, items in enumerate(eval_bar):
                if self.kwargs['model_type'] == "graph":
                    drug_id, drug_graph, multi_label, max_len, group_arr_list = items
                    node_feats = [drug_graph.ndata.pop('atomic_number'), drug_graph.ndata.pop('chirality_type')]
                    edge_feats = [drug_graph.edata.pop('bond_type'), drug_graph.edata.pop('bond_direction_type')]
                    output = model(drug_graph, node_feats, edge_feats, max_len, group_arr_list)
                else:
                    drug_id, drug_smiles, multi_label = items
                    output = model(drug_smiles)
                predict_multi_label = torch.where(output >= 0,
                                                  torch.tensor(1, device=self.device),
                                                  torch.tensor(0, device=self.device))
                predict_multi_label_list = predict_multi_label.tolist()
                multi_label_list = multi_label.tolist()
                val_pre_list.extend(predict_multi_label_list)
                val_true_list.extend(multi_label_list)
                drug_id_list.extend(drug_id)

            val_predict_df_list = []
            for j in range(len(val_true_list)):
                val_predict = pd.DataFrame({"index": val_index[j],
                                            "drug_id": drug_id_list[j],
                                            f"predict_l{self.level}": str(val_pre_list[j]),
                                            f"true_l{self.level}": str(val_true_list[j])},
                                           index=[val_index[j]])
                val_predict[[f"aiming_l{self.level}", f"coverage_l{self.level}", f"accuracy_l{self.level}",
                             f"abs_true_l{self.level}", f"abs_false_l{self.level}"]] = \
                    val_predict.apply(lambda x: self.get_5_metric(
                        val_pre_list[j], val_true_list[j], self.kwargs["n_class"]), axis=1).apply(pd.Series)
                val_predict_df_list.append(val_predict)

            return pd.concat(val_predict_df_list)

    def save_and_write_opt(self, val_predict):
        if self.epoch >= self.record_epoch - 1:
            if val_predict[f"aiming_l{self.level}"].values.mean() >= self.val_predict_opt[f"aiming_l{self.level}"].values.mean():
                self.val_predict_opt = val_predict
        else:
            self.val_predict_opt = val_predict
        if self.writer:
            self.write_opt_performance()

    def get_val_opt(self):
        return self.val_predict_opt

    def write_opt_performance(self):
        suffix = "_l" + self.level
        aiming = self.val_predict_opt['aiming' + suffix].values.mean()
        coverage = self.val_predict_opt['coverage' + suffix].values.mean()
        accuracy = self.val_predict_opt['accuracy' + suffix].values.mean()
        abs_true = self.val_predict_opt['abs_true' + suffix].values.mean()
        abs_false = self.val_predict_opt['abs_false' + suffix].values.mean()
        self.summary_writer.add_scalar(f'metric{suffix}/aiming', aiming, self.epoch + 1)
        self.summary_writer.add_scalar(f'metric{suffix}/coverage', coverage, self.epoch + 1)
        self.summary_writer.add_scalar(f'metric{suffix}/accuracy', accuracy, self.epoch + 1)
        self.summary_writer.add_scalar(f'metric{suffix}/abs_true', abs_true, self.epoch + 1)
        self.summary_writer.add_scalar(f'metric{suffix}/abs_false', abs_false, self.epoch + 1)

    def close(self):
        if self.writer:
            self.summary_writer.close()
