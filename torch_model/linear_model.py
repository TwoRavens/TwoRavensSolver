from torch import nn


# General form of FC-Layer only MLP
class NLayerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, back_steps, inner_blocks=None, activation='relu'):
        super(NLayerMLP, self).__init__()
        components = list()
        if not inner_blocks:
            components.append(nn.Linear(input_dim, input_dim * 4))
            if 'relu' == activation:
                components.append(nn.ReLU())
            elif 'leaky_relu' == activation:
                components.append(nn.LeakyReLU(0.2))
            elif 'sigmoid' == activation:
                components.append(nn.Sigmoid())
            elif 'tanh' == activation:
                components.append(nn.Tanh())
            else:
                raise NotImplementedError('Unsupported Activation Function {}'.format(activation))
            components.append(nn.Linear(input_dim * 4, output_dim))
        else:
            # inner_block is a list of layer configuration (input_dim & output_dim excluded)
            # E.G. (L1, L2, ..., Lk)
            pre_dim = input_dim
            for idx in range(len(inner_blocks)):
                current_dim = inner_blocks[idx]
                components.append(nn.Linear(pre_dim, current_dim))

                if 'relu' == activation:
                    components.append(nn.ReLU())
                elif 'leaky_relu' == activation:
                    components.append(nn.LeakyReLU(0.2))
                elif 'sigmoid' == activation:
                    components.append(nn.Sigmoid())
                elif 'tanh' == activation:
                    components.append(nn.Tanh())
                else:
                    raise NotImplementedError('Unsupported Activation Function {}'.format(activation))

                pre_dim = current_dim

            components.append(nn.Linear(pre_dim, output_dim))

        self.model = nn.Sequential(*components)
        self.back_steps = back_steps
        self.out_dim = output_dim
        self.last_record = list()

    def forward(self, x):
        return self.model(x)

    def set_last_record(self, tgt_list):
        self.last_record = tgt_list

    def forecast(self, x, steps):

        pass
