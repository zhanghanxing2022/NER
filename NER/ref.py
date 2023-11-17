import torch
import torch.nn as nn
import torch.nn.functional as F

class ChainCRF(nn.Module):
    def __init__(self, num_classes):
        super(ChainCRF, self).__init__()
        self.U = nn.Parameter(torch.rand(num_classes, num_classes))
        self.b_start = nn.Parameter(torch.zeros(num_classes))
        self.b_end = nn.Parameter(torch.zeros(num_classes))

    def path_energy(self, y, x, mask=None):
        x = self.add_boundary_energy(x, mask)
        return self.path_energy0(y, x, mask)

    def path_energy0(self, y, x, mask=None):
        n_classes = x.size(2)
        y_one_hot = F.one_hot(y, n_classes).float()

        # Tag path energy
        energy = torch.sum(x * y_one_hot, dim=2)
        energy = torch.sum(energy, dim=1)

        # Transition energy
        y_t = y[:, :-1]
        y_tp1 = y[:, 1:]
        U_flat = self.U.view(-1)
        flat_indices = y_t * n_classes + y_tp1
        U_y_t_tp1 = U_flat[flat_indices]

        if mask is not None:
            mask = mask.float()
            y_t_mask = mask[:, :-1]
            y_tp1_mask = mask[:, 1:]
            U_y_t_tp1 *= y_t_mask * y_tp1_mask

        energy += torch.sum(U_y_t_tp1, dim=1)

        return energy

    def sparse_chain_crf_loss(self, y, x, mask=None):
        x = self.add_boundary_energy(x, mask)
        energy = self.path_energy0(y, x, mask)
        energy -= self.free_energy0(x, mask)
        return -energy.unsqueeze(-1)

    def add_boundary_energy(self, x, mask=None):
        if mask is None:
            x = torch.cat([x[:, :1, :] + self.b_start, x[:, 1:, :]], dim=1)
            x = torch.cat([x[:, :-1, :], x[:, -1:, :] + self.b_end], dim=1)
        else:
            mask = mask.float()
            x *= mask
            start_mask = torch.cat([torch.zeros_like(mask[:, :1]), mask[:, :-1]], dim=1)
            start_mask = (mask > start_mask).float()
            x = x + start_mask.unsqueeze(-1) * self.b_start
            end_mask = torch.cat([mask[:, 1:], torch.zeros_like(mask[:, -1:])], dim=1)
            end_mask = (mask > end_mask).float()
            x = x + end_mask.unsqueeze(-1) * self.b_end
        return x

    def viterbi_decode(self, x, mask=None):
        x = self.add_boundary_energy(x, mask)

        alpha_0 = x[:, 0, :]
        gamma_0 = torch.zeros_like(alpha_0)
        initial_states = [gamma_0, alpha_0]
        _, gamma = self._forward(x, initial_states, mask)
        y = self._backward(gamma, mask)
        return y

    def free_energy(self, x, mask=None):
        x = self.add_boundary_energy(x, mask)
        return self.free_energy0(x, mask)

    def free_energy0(self, x, mask=None):
        initial_states = [x[:, 0, :]]
        last_alpha, _ = self._forward(x, initial_states, mask)
        return last_alpha[:, 0]

    def _forward(self, x, states, mask=None):
        def _forward_step(energy_matrix_t, states):
            alpha_tm1 = states[-1]
            new_states = [torch.logsumexp(alpha_tm1.unsqueeze(2) + energy_matrix_t, dim=1)]
            return new_states[0], new_states

        U_shared = self.U.unsqueeze(0).unsqueeze(0)

        if mask is not None:
            mask = mask.float()
            mask_U = (mask[:, :-1] * mask[:, 1:]).unsqueeze(2).unsqueeze(3)
            U_shared = U_shared * mask_U

        inputs = (x[:, 1:, :] + U_shared).contiguous()
        inputs = torch.cat([inputs, torch.zeros_like(inputs[:, -1:, :, :])], dim=1)

        last, values = self._rnn(_forward_step, inputs, states)
        return last, values

    def _rnn(self, fn, inputs, initial_states):
        def step(input, states):
            return fn(input, states)

        return torch.scan(step, inputs, initial_states)

    def _backward(self, gamma, mask):
        gamma = gamma.int()

        def _backward_step(gamma_t, states):
            y_tm1 = states[0].squeeze(0)
            y_t = gamma_t.gather(1, y_tm1.unsqueeze(1))
            return y_t, [y_t.unsqueeze(0)]

        initial_states = [torch.zeros_like(gamma[:, 0, 0]).unsqueeze(0)]
        _, y_rev = self._rnn(_backward_step, gamma, initial_states)
        y = torch.flip(y_rev, [1])

        if mask is not None:
            mask = mask.int()
            y *= mask
            y += -(1 - mask)
        return y

    def forward(self, x):
        # During training, return x; during testing, return viterbi decoding
        return F.in_train_mode(x, self.viterbi_decode(x))

# Example Usage
num_classes = 5
crf = ChainCRF(num_classes)
x = torch.rand(2, 4, num_classes)  # Batch size of 2, sequence length of 4
y_true = torch.tensor([[1, 2, 3, 4], [0, 2, 1, 3]])  # Example true tag sequences

# Training
loss = crf.sparse_chain_crf_loss(y_true, x)
print("Training Loss:", loss.item())

# Testing
y_pred = crf(x)
print("Predicted Tags:", y_pred.numpy())
