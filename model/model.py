import torch.nn as nn
import torch
from torch.autograd import Variable
from model.AoA import AoA
from model.FCLayer import FCLayer


class HMN(nn.Module):
    def __init__(self, hidden_dim=200, embbeding_dim=200):
        super(HMN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embbeding_dim = embbeding_dim
        self.AoALayer = AoA()
        self.final_fc = FCLayer(hidden_dim * 2, 6, type="deep")
        self.embs = nn.Embedding(339503, embbeding_dim)
        self.gru = nn.GRU(embbeding_dim, hidden_dim, batch_first=True)

    def RSABlock(self, text_embedding, text_lens, law_repeat, law_len):
        source_raw = text_embedding
        target_raw = law_repeat
        avg_alpha, avg_beta = self.AoALayer.forward(text_embedding, text_lens,
                                                    law_repeat, law_len)

        # source_raw (B, L, H)
        # avg_alpha (B, L, H)
        FactAoA = source_raw + source_raw * avg_alpha
        LabelAoA = target_raw + target_raw * avg_beta

        # simple version
        # (B, L, H) -> (B, H)
        FactAoA_output = torch.mean(FactAoA, dim=1)
        # (B, LS, H) -> (B, H)
        LabelAoA_output = torch.mean(LabelAoA, dim=1)
        # (B, H) + (B, H) -> (B, 2H)
        output_feature = torch.cat((FactAoA_output, LabelAoA_output), dim=-1)

        # M (B, LS, L)
        return output_feature, avg_alpha

    def encoder(self, text):
        # text is a list of (S,W)
        # L = max_seg
        max_seg = max([len(i) for i in text])
        # (0, L, H) -> (B,L,H)
        text_embedding = torch.rand((0, max_seg, self.hidden_dim)).cuda()
        for t in text:
            seq_len = len(t)
            # (S,W) -> (S,W)
            t = torch.LongTensor(t).cuda()
            # (S,W) -> (S,W,embbeding_dim)
            t = self.embs(t)
            # (S,W,embbeding_dim) -> (1,S,hidden_dim)
            # _, t = self.gru(t)
            t = torch.mean(t, dim=1).unsqueeze(0)
            # (1,S,hidden_dim) -> (1,L,H)
            t = torch.cat(
                (t, torch.zeros(
                    (1, max_seg - seq_len, self.hidden_dim)).cuda()),
                dim=1)
            # (1,L,H) -> (B,L,H)
            text_embedding = torch.cat((text_embedding, t), dim=0)
        # (B,L,H)
        return text_embedding.cuda()

    def forward(self, text, text_lens, law):
        # list of (S,W) -> (B,L,H)
        text_embedding = self.encoder(text)

        # (1, LS, H)
        law_embedding = self.encoder([law])
        # (B, LS, H)
        law_repeat = law_embedding.repeat(
            (text_embedding.size(0), 1, 1)).cuda()
        # (B, 1)
        law_len = Variable(
            torch.LongTensor([law_repeat.size(1)] *
                             text_embedding.size(0))).cuda()

        output_feature, avg_alpha = self.RSABlock(text_embedding, text_lens,
                                                  law_repeat, law_len)

        predict = self.final_fc(output_feature)
        predict = torch.nn.functional.softmax(predict, dim=1)
        return predict, avg_alpha
