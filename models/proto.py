import sys
sys.path.append('..')
import fewshot_re_kit
import torch
from torch import autograd, optim, nn
from torch.autograd import Variable
from torch.nn import functional as F

class Proto(fewshot_re_kit.framework.FewShotREModel):
    
    def __init__(self, sentence_encoder, dot=False, relation_encoder=None, N=5, Q=1):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout()
        self.dot = dot
        self.relation_encoder = relation_encoder
        self.hidden_size = 768
    
    def __dist__(self, x, y, dim):
        if self.dot:
            return (x * y).sum(dim)
        else:
            return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, support, query, rel_txt, N, K, total_Q):
        rel_gol, rel_loc = self.sentence_encoder(rel_txt, cat=False)
        rel_loc = torch.mean(rel_loc, 1)
        rel_rep = torch.cat((rel_gol, rel_loc), -1)
        support_h, support_t,  s_loc = self.sentence_encoder(support)
        query_h, query_t,  q_loc = self.sentence_encoder(query)
        support = torch.cat((support_h, support_t), -1)
        query = torch.cat((query_h, query_t), -1)
        support = support.view(-1, N, K, self.hidden_size*2)
        sample = support
        query = query.view(-1, total_Q, self.hidden_size*2)
        support = torch.mean(support, 2)
        rel_rep = rel_rep.view(-1, N, rel_gol.shape[1]*2)
        rel_gate = torch.tanh(abs(rel_rep / support - 1))
        support = rel_rep * rel_gate + support
        proto = support


        logits = self.__batch_dist__(support, query)
        minn, _ = logits.min(-1)
        logits = torch.cat([logits, minn.unsqueeze(2) - 1], 2)
        _, pred = torch.max(logits.view(-1, N + 1), 1)

        return logits, pred, sample, proto
