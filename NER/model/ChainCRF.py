# import ChainCRF
# import importlib
# importlib.reload(ChainCRF)
# from ChainCRF import ChainCRF
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class BLITM(nn.Module):
    def __init__(self, num_classes, vocab_length, embedding_dim, hidden_dim):
        super(BLITM, self).__init__()
        self.embed = nn.Embedding(vocab_length, embedding_dim)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden2tag = nn.Sequential(nn.Linear(hidden_dim, num_classes))

    def forward(self, sentence):
        embeds = self.embed(sentence)
        lstm_out, _ = self.lstm(embeds)
        emissions = self.hidden2tag(lstm_out)
        return emissions


class ChainCRF(nn.Module):
    def __init__(self, num_classes, init_value=None):
        super(ChainCRF, self).__init__()
        self.num_classes = num_classes
        # 定义转移矩阵 U、起始边界 b_start、结束边界 b_end
        if init_value:
            self.U = nn.Parameter(torch.full(
                (num_classes, num_classes), init_value))
        else:
            self.U = nn.Parameter(torch.rand(num_classes, num_classes))
        self.b_start = nn.Parameter(torch.zeros(num_classes))
        self.b_end = nn.Parameter(torch.zeros(num_classes))
        # sigmoidU = nn.Sigmoid()(sigmoidU)
        self.sigmoid = nn.Sigmoid()
        self.F = torch.sum

    def forward(self, emissions, true_tags, mask=None):
        """
        :param emissions: [batch_size, seq_len, num_classes] - emission scores
        :param true_tags: [batch_size, seq_len] - true tag indices
        :param mask: [batch_size, seq_len] - binary mask for sequence lengths
        :return: loss
        """
        # Calculate negative log-likelihood loss
        # sigmoidU = self.sigmoid(self.U)
        sigmoidU = self.U
        tag_energy = self.path_energy(sigmoidU, emissions, true_tags, mask)
        free_energy = self.free_energy(emissions, mask)
        loss = free_energy-tag_energy
        res = torch.mean(loss.unsqueeze(1))
        # print(res)
        return res

    def path_energy(self, sigmoidU, emissions, true_tags, mask=None):
        """
        : return [batch_size]
        """
        true_tags_mask = F.one_hot(
            (true_tags*mask).long(), self.num_classes).float()
        energy = true_tags_mask * emissions
        energy1 = self.F(energy, dim=2)

        if mask is not None:
            energy1 = energy1 * mask.float()

        energy2 = self.F(energy1, dim=1)

        prev = true_tags[:, :-1]
        next = true_tags[:, 1:]
        transitions = prev * self.num_classes + next
        # print("prev:", prev)
        # print("next:", next)
        # print("transitions:", transitions)
        U_flat = sigmoidU.reshape((-1))
        # print("U_flat:", U_flat)
        U_y_t_em = U_flat[transitions]
        # print(U_y_t_em.shape)
        if mask is not None:
            mask_transitions = mask[:, 1:]
            U_y_t_em *= mask_transitions.view(*U_y_t_em.shape).float()
            # print("U_y_t_em", U_y_t_em)
        U = self.F(U_y_t_em, dim=1)
        return energy2+U

    def free_energy(self, emissions, mask=None):
        """Compute the free energy using the forward algorithm.
        :param emissions: [batch_size, seq_len, num_classes] - emission scores
        :param mask: [batch_size, seq_len] - binary mask for sequence lengths
        :return: [batch_size] - free energy for each sequence in the batch
        """
        alpha = torch.zeros_like(emissions)
        alpha[:, 0, :] = emissions[:, 0, :]
        batch_size, seq_len, _ = emissions.size()
        for j in range(1, seq_len):
            # print(emissions[:, j, :].unsqueeze(2).shape)
            # # 32, 9, 1
            # print(self.U.unsqueeze(0).shape)
            # 1 ,9, 9
            new_score = emissions[:, j, :].unsqueeze(1)
            # print((new_score+self.U.unsqueeze(0)).shape)
            # print(mask[:, j].shape)
            # print("new_score",new_score)
            # print("self.U.unsqueeze(0)", self.U.unsqueeze(0))
            # print("mask[:, j].unsqueeze(1).unsqueeze(2)",
            #   mask[:, j].unsqueeze(1).unsqueeze(2))
            transition_energy = (new_score+self.U.unsqueeze(0)) * \
                mask[:, j].unsqueeze(1).unsqueeze(2)
            # print("transition_energy:", transition_energy)

            alpha_1 = alpha[:, j-1, :].unsqueeze(2)
            # print("alpha_1:", alpha_1)
            # print("alpha_1+transition_energy", alpha_1+transition_energy)
            alpha[:, j, :] = torch.logsumexp(alpha_1+transition_energy, dim=1)
            # print("alpha:", alpha)

        free_energy = torch.logsumexp(alpha[:, -1, :], dim=1)
        return free_energy

    def viterbi1(self, emission, mask=None):
        batch_size, seq_len, num_classes = emission.size()
        # sigmoidU = self.sigmoid(self.U)
        sigmoidU = self.U
        score = torch.zeros(emission.shape)
        path = torch.zeros(emission.shape, dtype=torch.long)
        score[:, 0, :] = emission[:, 0, :]

        for i in range(batch_size):
            for j in range(1, seq_len):
                for k in range(num_classes):
                    temp_emission = emission[i, j, k]
                    last_score = score[i, j - 1, :] + temp_emission
                    sigmoidU_ = (sigmoidU[:, k] + last_score).unsqueeze(1)
                    # print(sigmoidU_.shape)
                    score[i, j, k], path[i, j, k] = torch.max(sigmoidU_, dim=0)
        # print("score:", score)
        # print("path:", path)
        return score, path

    def viterbi2(self, emission, mask=None):
        _, seq_len, _ = emission.size()
        # sigmoidU = self.sigmoid(self.U)
        sigmoidU = self.U
        score = torch.zeros(emission.shape)
        path = torch.zeros(emission.shape, dtype=torch.long)
        score[:, 0, :] = emission[:, 0, :]

        for j in range(1, seq_len):

            prev = score[:, j - 1, :].unsqueeze(2)
            # 32,9,1
            temp_emission = emission[:, j, :].unsqueeze(1)
            # 32 ,1,9
            # print("prev", prev)
            # print("temp_emission:", temp_emission)

            last_score = prev + temp_emission
            # 32 ,9, 9
            # print("last_score:", last_score)
            # print("sigmoidU[ll.int(), :].unsqueeze(0):",
            #   sigmoidU[ll.int(), :].unsqueeze(0))
            sigmoidU_ = sigmoidU.unsqueeze(0) + last_score
            # 32, 9, 9
            # print("sigmoidU_-aaa:", sigmoidU_-temp_emission)

            # print("sigmoidU_:", sigmoidU_)
            score[:, j, :], path[:, j, :] = torch.max(sigmoidU_, dim=1)
        # print("score:", score)
        # print("path:", path)

        return score, path

    def viterbi_decode(self, emission, mask=None):
        """Decode the highest scoring sequence of tags using the Viterbi algorithm.
        :param emission: [batch_size, seq_len, num_classes] - emission scores
        :param mask: [batch_size, seq_len] - binary mask for sequence lengths
        :return: [batch_size, seq_len] - the tag indices of the highest scoring sequence
        """
        batch_size, seq_len, _ = emission.size()
        score, path = self.viterbi2(emission, mask)
        best_path = torch.zeros((batch_size, seq_len), dtype=torch.long)
        _, best_last_tag = torch.max(score[:, -1, :], dim=1)
        best_path[:, -1] = best_last_tag

        for j in range(seq_len - 2, -1, -1):
            for i in range(batch_size):
                best_path[i, j] = path[i, j + 1, best_path[i, j + 1]]

        if mask is not None:
            best_path *= mask.long()

        return best_path

    def viterbi_decode2(self, emission, mask=None):
        """Decode the highest scoring sequence of tags using the Viterbi algorithm.
        :param emission: [batch_size, seq_len, num_classes] - emission scores
        :param mask: [batch_size, seq_len] - binary mask for sequence lengths
        :return: [batch_size, seq_len] - the tag indices of the highest scoring sequence
        """
        batch_size, seq_len, _ = emission.size()
        score, path = self.viterbi1(emission, mask)
        best_path = torch.zeros((batch_size, seq_len), dtype=torch.long)
        _, best_last_tag = torch.max(score[:, -1, :], dim=1)
        best_path[:, -1] = best_last_tag

        for j in range(seq_len - 2, -1, -1):
            for i in range(batch_size):
                best_path[i, j] = path[i, j + 1, best_path[i, j + 1]]

        if mask is not None:
            best_path *= mask.long()

        return best_path
