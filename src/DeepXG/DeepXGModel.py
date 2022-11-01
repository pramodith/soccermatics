import pytorch_lightning as pl
import torch.nn as nn
import torch
import torchmetrics
from DeepXGDataModule import DeepXGDataModule
import torch.optim as optim
from constants import *
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import classification_report
import json

pl.seed_everything(42)


class DeepXG(pl.LightningModule):
    def __init__(self, embed_dim=10):
        super().__init__()
        with open("../../data/deepXG/cat2id.json") as f:
            self.cat2id = json.load(f)

        # All the embedding layers required.
        self.zone_embeds = nn.Embedding(num_embeddings=81, padding_idx=80, embedding_dim=embed_dim)
        self.position_embeds = nn.Embedding(num_embeddings=51, padding_idx=50, embedding_dim=embed_dim)
        self.embeddings = nn.ParameterDict(
            {feature: nn.Embedding(padding_idx=len(self.cat2id[feature]), num_embeddings=len(self.cat2id[feature]) + 1,
                                   embedding_dim=embed_dim) for feature in CATEGORICAL_VARIABLES})

        # Using Multihead attention to handle sequence data i.e passing sequences
        self.self_attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=1, dropout=0.1)

        # Projection Layers
        self.emb_proj = nn.ParameterList(
            [nn.Sequential(nn.ReLU(), nn.Dropout(0.1), nn.Linear(10, 1)) for i in range(len(CATEGORICAL_VARIABLES))])
        self.zone_proj = nn.Sequential(nn.ReLU(), nn.Dropout(0.1), nn.Linear(10, 1))

        # Final MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(
                len(CATEGORICAL_VARIABLES) + len(BOOLEAN_VARIABLES) + len(CONTINUOUS_VARIABLES) + len(ZONE_FEATURES),
                256),
            nn.ReLU(), nn.Linear(256, 1))

        # Metrics to keep track of
        self.classification_threshold = 0.5
        self.val_accuracy = torchmetrics.Accuracy()
        self.val_precision = torchmetrics.Precision()
        self.val_f1_score = torchmetrics.F1Score()
        self.val_precision_recall = torchmetrics.PrecisionRecallCurve()
        self.test_f1_score = torchmetrics.F1Score()

    def forward(self, batch):
        pass_seq, shot_zones, cat_x, cont_x, y = batch[0], batch[1], batch[2], batch[3], batch[4]
        shot_emb = self.zone_embeds(shot_zones)
        pass_emb = self.zone_embeds(pass_seq)
        positions_tensor = torch.range(0, pass_emb.shape[1] - 1).repeat(pass_seq.shape[0], 1) * torch.ne(pass_seq, 80)
        positions_tensor = positions_tensor.long()
        poistion_emb = self.position_embeds(positions_tensor)
        pass_emb = pass_emb + poistion_emb
        pass_tensor, _ = self.self_attention(pass_emb, pass_emb, pass_emb)
        pass_tensor = torch.sum(pass_tensor, 1)
        pass_tensor = self.zone_proj(pass_tensor)
        shot_tensor = self.zone_proj(shot_emb)
        # _, (h, c) = self.lstm(pass_emb)
        embeddings = [self.embeddings[v](cat_x[:, i]) for i, v in enumerate(CATEGORICAL_VARIABLES)]
        emb_proj = torch.cat([self.emb_proj[i](embeddings[i]) for i in range(len(CATEGORICAL_VARIABLES))], 1)
        all_vars = torch.cat((emb_proj, cont_x, pass_tensor, shot_tensor), 1)
        # logit = self.mlp(torch.cat((h.reshape(-1, 100), torch.stack(x)), 1))
        logit = self.mlp(all_vars)
        return logit.reshape(-1)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        logit = self.forward(batch)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, batch[-1].float(), pos_weight=torch.tensor(10.0))
        # loss = nn.functional.mse_loss(logit.reshape(-1), batch[2].float())
        # Logging to TensorBoard by default
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        logit = self.forward(batch)
        loss = nn.functional.binary_cross_entropy_with_logits(logit, batch[-1].float(), pos_weight=torch.tensor(10.0))
        # loss = nn.functional.mse_loss(logit.reshape(-1), batch[2].float())
        print(f"Number of goals is {torch.sum(batch[-1])}")
        self.val_accuracy(logit.reshape(-1), batch[-1].long())
        self.val_f1_score(logit.reshape(-1), batch[-1].long())
        self.val_precision(logit.reshape(-1), batch[-1].long())
        # self.val_precision_recall(logit.reshape(-1), batch[-1].long())
        # Logging to TensorBoard by default
        self.log("val_loss", loss)
        self.log('val_acc', self.val_accuracy, on_epoch=True, on_step=True)
        self.log('val_precision', self.val_precision, on_step=True, on_epoch=True)
        self.log('val_f1_score', self.val_f1_score, on_epoch=True, on_step=True)
        # self.log('val_precision_recall_score', self.val_precision_recall, on_epoch=True, on_step=True)
        return loss

    def validation_epoch_end(self, outputs):
        self.log('val_acc_epoch', self.val_accuracy.compute())
        self.log('val_f1_score_epoch', self.val_f1_score.compute())
        self.log('val_precision_epoch', self.val_precision.compute())
        # self.log('val_precision_recall_epoch', self.val_precision_recall.compute())
        self.val_accuracy.reset()
        self.val_f1_score.reset()
        self.val_precision.reset()
        self.val_precision_recall.reset()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-3)
        return optimizer

    def test_step(self, batch, batch_ind):
        logit = self.forward(batch)
        return logit > self.classification_threshold, batch[-1].long()

    def test_epoch_end(self, outputs):
        preds = []
        labels = []
        for i in range(len(outputs)):
            preds.extend(outputs[i][0].numpy())
            labels.extend(outputs[i][1].numpy())

        print(classification_report(labels, preds))


if __name__ == "__main__":
    dm = DeepXGDataModule(data_dir="../../data/deepXG/")
    early_stop_callback = EarlyStopping(monitor="val_f1_score_epoch", min_delta=0.00, patience=10, verbose=True,
                                        mode="max")
    train_dl = dm.train_dataloader()
    val_dl = dm.val_dataloader()
    trainer = pl.Trainer(max_epochs=50, log_every_n_steps=50, check_val_every_n_epoch=1,
                         callbacks=[early_stop_callback])
    trainer.fit(DeepXG(), dm)
    #model = DeepXG().load_from_checkpoint("lightning_logs/version_1/checkpoints/epoch=28-step=5075.ckpt")
    #trainer.test(model, dm, verbose=False)
