from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from tqdm import tqdm
from graphatc.evaluater import Evaluator

scaler = GradScaler()


class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer,
                 epochs, record_epoch, log_path, writer, level, use_amp, **kwargs):
        self.evaluator = Evaluator(train_loader, val_loader, record_epoch, log_path, writer, level, **kwargs)
        self.level = level
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.epochs = epochs
        self.device = train_loader.dataset.device
        self.kwargs = kwargs
        self.writer = writer
        self.use_amp = use_amp

    def train_one_epoch(self):
        self.model.train()
        batch_nums = len(self.train_loader)

        batch_bar = tqdm(self.train_loader, total=batch_nums, desc='[BATCH]',
                         unit='batch', position=3, leave=False, colour='green', ncols=60)
        for _, items in enumerate(batch_bar):

            if self.kwargs['model_type'] == "graph":
                _, drug_graph, multi_label, max_len, group_arr_list = items
                node_feats = [drug_graph.ndata.pop('atomic_number'), drug_graph.ndata.pop('chirality_type')]
                edge_feats = [drug_graph.edata.pop('bond_type'), drug_graph.edata.pop('bond_direction_type')]
            else:
                _, drug_smiles, multi_label = items
            
            running_batch_size = len(multi_label)

            self.optimizer.zero_grad()
            with autocast(enabled=self.use_amp):

                if self.kwargs['model_type'] == "graph":
                    output = self.model(drug_graph, node_feats, edge_feats, max_len, group_arr_list)
                else:
                    output = self.model(drug_smiles)

                loss = self.criterion(output, multi_label.float())

            self.evaluator.receive(running_batch_size, loss.item())

            if self.use_amp:
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

        batch_bar.close()
        if self.writer:
            self.evaluator.write_train_loss()

    def train(self):

        epoch_bar = tqdm(range(self.epochs), total=self.epochs, desc='[EPOCH]',
                         unit='epoch', position=2, leave=False, colour='yellow', ncols=60)
        for epoch in epoch_bar:
            self.evaluator.set_epoch(epoch)
            self.train_one_epoch()
            val_predict = self.evaluator.eval_model(self.model)
            self.evaluator.save_and_write_opt(val_predict)

            # early stop only for jf
            if self.kwargs["input_args"]["TRAIN_METHOD"] == "Jackknife" \
                    and val_predict[f"aiming_l{self.level}"].item() >= 0.95 \
                    and val_predict[f"accuracy_l{self.level}"].item() >= 0.95:
                break
        val_predict_opt = self.evaluator.get_val_opt()
        self.evaluator.close()
        return val_predict_opt
