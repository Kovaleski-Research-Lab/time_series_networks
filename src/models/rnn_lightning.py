# Creates an RNN model using the PyTorch Lightning framework
# Author: Marshall Lindsay (mblgh6@missouri.edu)

import torch
import torch.nn as nn
import pytorch_lightning as pl

class RNNLightning(pl.LightningModule):
    def __init__(self, params:dict):
        super().__init__()
        self.params = self.parse_params(params)
        try:
            self.rnn = nn.RNN(input_size=self.params['input_size'],
                              hidden_size=self.params['hidden_size'],
                              num_layers=self.params['num_layers'],
                              batch_first=self.params['batch_first'],
                              nonlinearity=self.params['activation'])
        except ValueError:
            raise ValueError('Invalid activation function')

        self.fc = self.build_fully_connected()
        self.objective_function = self.create_objective_function()
        self.save_hyperparameters()

    def parse_params(self, params:dict) -> dict:
        params['input_size'] = int(params['input_size'])
        params['hidden_size'] = int(params['hidden_size'])
        params['num_layers'] = int(params['num_layers'])
        params['output_size'] = int(params['output_size'])
        params['learning_rate'] = float(params['learning_rate'])
        params['optimizer'] = str(params['optimizer'])
        params['batch_first'] = bool(params['batch_first'])
        params['activation'] = str(params['activation'])
        params['fc'] = dict(params['fc'])
        params['objective'] = str(params['objective'])

        for key in params['fc']:
            if 'activation' in key:
                params['fc'][key] = str(params['fc'][key])
            else:
                params['fc'][key] = int(params['fc'][key])
        return params

    def build_fully_connected(self) -> nn.Sequential:
        fc_layers = self.params['fc']

        # Define our activation function
        if fc_layers['activation'] == 'relu':
            activation = nn.ReLU()
        elif fc_layers['activation'] == 'sigmoid':
            activation = nn.Sigmoid()
        elif fc_layers['activation'] == 'tanh':
            activation = nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

        layers = []
        layer_keys = [key for key in fc_layers.keys() if 'activation' not in key]

        # Input layer
        layers.append(nn.Linear(self.params['hidden_size'], fc_layers[layer_keys[0]]))
        layers.append(activation)

        # Hidden layers
        for i in range(len(layer_keys) - 1):
            layers.append(nn.Linear(fc_layers[layer_keys[i]], fc_layers[layer_keys[i+1]]))
            layers.append(activation)

        # Output layer
        layers.append(nn.Linear(fc_layers[layer_keys[-1]], self.params['output_size']))

        return nn.Sequential(*layers)

    def create_objective_function(self) -> nn.Module:
        if self.params['objective'] == 'cross_entropy':
            return nn.CrossEntropyLoss()
        elif self.params['objective'] == 'mse':
            return nn.MSELoss()
        else:
            raise ValueError('Invalid objective function')

    def configure_optimizers(self) -> torch.optim.Optimizer:
        if self.params['optimizer'].lower() == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.params['learning_rate'])
        else:
            raise ValueError('Invalid optimizer')

    def objective(self, prediction:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return self.objective_function(prediction, target)

    def forward(self, x:torch.Tensor) -> list[torch.Tensor]:
        rnn_out, rnn_hidden = self.rnn(x)
        prediction = self.fc(rnn_out[:, -1, :])
        return [prediction, rnn_out, rnn_hidden]

    def training_step(self, batch:list[torch.Tensor], batch_idx:int) -> torch.Tensor:
        x, y = batch
        prediction, rnn_out, rnn_hidden = self.forward(x)
        loss = self.objective_function(prediction, y)
        self.log('train_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch:list[torch.Tensor], batch_idx:int) -> torch.Tensor:
        x, y = batch
        prediction, rnn_out, rnn_hidden = self.forward(x)
        loss = self.objective_function(prediction, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def test_step(self, batch:list[torch.Tensor], batch_idx:int) -> torch.Tensor:
        x, y = batch
        prediction, rnn_out, rnn_hidden = self.forward(x)
        loss = self.objective_function(prediction, y)
        self.log('test_loss', loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss

#-----------------------------#
#         Test Code           #
#-----------------------------#
if __name__ == "__main__":
    # Parameters for the RNN
    params = {
        'input_size': 1,
        'hidden_size': 128,
        'num_layers': 5,
        'output_size': 1,
        'learning_rate': 0.001,
        'optimizer': 'adam',
        'batch_first': True,
        'activation': 'relu',
        'objective': 'cross_entropy',
        'fc':
            {
                'activation': 'relu',
                'size0': 64,
                'size1': 64,
                'size2': 32
            }
    }

    # Create the model
    model = RNNLightning(params)

    # Create some fake data to test the model
    # Batch size, sequence length, input size
    # Simulates 32 sequences of length 28 with 1 input - like power consumption data
    fake_data = torch.randn(32, 28, 1)

    # Run the model
    prediction, rnn_out, rnn_hidden = model(fake_data)
    print(prediction.shape)
    print(rnn_out.shape)
    print(rnn_hidden.shape)

