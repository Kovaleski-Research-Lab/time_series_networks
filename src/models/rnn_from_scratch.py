# Creates an RNN model from scratch using the PyTorch Lightning framework
# Author: Marshall Lindsay (mblgh6@missouri.edu)
# References:
# 


import torch
import pytorch_lightning as pl

class RNNFromScratch(pl.LightningModule):
    def __init__(self, params:dict):
        super().__init__()
        self.params = self.parse_params(params)
        self.batch_first = self.params['batch_first']
        self.batch_size = self.params['batch_size']
        self.hidden_size = self.params['hidden_size']
        self.activation_function = self.get_activation_function()
        self.build_weight_matrices()
        self.build_fully_connected()

    def get_activation_function(self):
        if self.params['activation'] == 'relu':
            return torch.nn.ReLU()
        elif self.params['activation'] == 'sigmoid':
            return torch.nn.Sigmoid()
        elif self.params['activation'] == 'tanh':
            return torch.nn.Tanh()
        else:
            raise ValueError('Invalid activation function')

    def build_weight_matrices(self):
        input_size = self.params['input_size']
        hidden_size = self.params['hidden_size']
        self.w_ih = torch.nn.init.normal_(torch.zeros(hidden_size, input_size, requires_grad=True), mean=0.0, std=1.0)
        self.b_ih = torch.nn.init.normal_(torch.zeros(hidden_size, requires_grad=True), mean=0.0, std=1.0)
        self.w_hh = torch.nn.init.normal_(torch.zeros(hidden_size, hidden_size, requires_grad=True), mean=0.0, std=1.0)
        self.b_hh = torch.nn.init.normal_(torch.zeros(hidden_size, requires_grad=True), mean=0.0, std=1.0)

    def build_fully_connected(self):
        fc_layers = self.params['fc']
        layers = []
        layer_keys = [key for key in fc_layers.keys() if 'activation' not in key]
        if fc_layers['activation'] == 'relu':
            activation = torch.nn.ReLU()
        elif fc_layers['activation'] == 'sigmoid':
            activation = torch.nn.Sigmoid()
        elif fc_layers['activation'] == 'tanh':
            activation = torch.nn.Tanh()
        else:
            raise ValueError('Invalid activation function')
        layers.append(torch.nn.Linear(self.params['hidden_size'], fc_layers[layer_keys[0]]))
        layers.append(activation)
        for i in range(len(layer_keys) - 1):
            layers.append(torch.nn.Linear(fc_layers[layer_keys[i]], fc_layers[layer_keys[i + 1]]))
            layers.append(activation)
        self.fc = torch.nn.Sequential(*layers)

    def parse_params(self, params:dict) -> dict:
        params['input_size'] = int(params['input_size'])
        params['hidden_size'] = int(params['hidden_size'])
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

    def rnn_cell(self, x, h):
        A = x @ self.w_ih.T + self.b_ih
        B = h @ self.w_hh.T + self.b_hh
        h = self.activation_function(A + B)
        return h

    def rnn_forward(self, x, h_0 = None):
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()
        if h_0 is None:
            h_0 = torch.zeros(batch_size, self.hidden_size)

        h_t_minus_1 = h_0
        h_t = h_0
        output = []
        for t in range(seq_len):
            h_t = self.rnn_cell(x[t], h_t_minus_1)
            output.append(h_t)
            h_t_minus_1 = h_t
        output = torch.stack(output)
        if self.batch_first:
            output = output.transpose(0, 1)
        return output, h_t

    def forward(self, x):
        output, hidden_state = self.rnn_forward(x)
        prediction = self.fc(output[:, -1, :])
        return prediction, output, hidden_state



#-----------------------------#
#         Test Code           #
#-----------------------------#
if __name__ == "__main__":
    # Parameters for the RNN
    params = {
        'input_size': 1,
        'hidden_size': 128,
        'batch_size': 32,
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
                'size2': 32,
                'output_size': 1,
            }
    }

    # Create the model
    model = RNNFromScratch(params)

    # Create some fake data to test the model
    # Batch size, sequence length, input size
    # Simulates 32 sequences of length 28 with 1 input - like power consumption data
    fake_data = torch.randn(32, 28, 1)

    # Run the model
    prediction, rnn_out, rnn_hidden = model(fake_data)
    print(prediction.shape)
    print(rnn_out.shape)
    print(rnn_hidden.shape)

    from IPython import embed; embed()

