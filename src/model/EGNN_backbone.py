import torch

class EGNN(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        model_config = config['model']
        self.sigma_data = model_config['sigma_data']
        self.sigma_max = model_config['sigma_max']
        self.sigma_min = model_config['sigma_min']
        self.beta_d = model_config['beta_d']
        self.beta_min = model_config['beta_min']
        self.sigma_data_end = self.sigma_data
        self.cov_xy = model_config['cov_xy']
        self.c = 1
        self.weight_schedule = model_config['weight_schedule']
        self.bridge_type = model_config['bridge_type']
        self.rho = model_config['rho']
        self.num_timesteps = 40
        self.feature_size = model_config['feature_size']
        self.lr = model_config['learning_rate']
        self.ema_decay = model_config['ema_decay']
        self.ema_params = model_config['ema_params']

    def get_input(self, batch):
        x, x_T = batch.x, batch.target_x
        pos, pos_T = batch.pos, batch.target_pos
        return x, x_T, pos, pos_T

    def forward(self, x, x_T, pos, pos_T):
        return x

    def run_step(self, batch, batch_idx):
        x, x_T, pos, pos_T = self.get_input(batch)
        losses = self(x, x_T, pos, pos_T)
        return losses

    def training_step(self, batch, batch_idx):
        losses = self.run_step(batch, batch_idx)
        return losses

    def validation_step(self, batch, batch_idx):
        losses = self.run_step(batch, batch_idx)
        return losses

    def test_step(self, batch, batch_idx):
        losses = self.run_step(batch, batch_idx)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def validation_epoch_end(self, outputs):
        pass

    def test_epoch_end(self, outputs):
        pass

    def get_loss(self, x, x_T, pos, pos_T):
        return torch.tensor(0.0)