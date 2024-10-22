import torch
from torch import nn
import numpy as np
from torch.autograd import Variable


class MyEmbedder(nn.Module):
    def __init__(self, vocab_size, emb_size, pad_id, max_len=5000,
                 dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size,
                                      embedding_dim=emb_size,
                                      padding_idx=pad_id)

        den = torch.exp(
            - torch.arange(0, emb_size, 2) * np.log(10000) / emb_size)
        pos = torch.arange(0, max_len).unsqueeze(1)
        pos_embedding = torch.zeros((max_len, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(0)

        self.register_buffer('pos_embedding', pos_embedding)

        self.emb_size = emb_size

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, tokens):
        embeddings = self.embedding(tokens.long()) * np.sqrt(self.emb_size)
        out = embeddings + Variable(self.pos_embedding[:, :embeddings.size(1)],
                                    requires_grad=False)
        return self.dropout(out)


class MyTransformer(nn.Module):
    def __init__(self, src_vocab, trg_vocab, nhead, pad_id, max_len,
                 d_model=512, num_enc_lay=3, num_dec_lay=3,
                 dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.embedding_src = MyEmbedder(src_vocab, d_model, pad_id,
                                        dropout=dropout)
        self.embedding_trg = MyEmbedder(trg_vocab, d_model, pad_id,
                                        dropout=dropout)

        self.trans = nn.Transformer(d_model=d_model, nhead=nhead,
                                    num_encoder_layers=num_enc_lay,
                                    num_decoder_layers=num_dec_lay,
                                    dim_feedforward=dim_feedforward,
                                    dropout=dropout, batch_first=True)
        self.linear = nn.Linear(d_model, trg_vocab)

        self.nhead = nhead

    def forward(self, src, trg, src_mask, trg_mask, src_padding_mask,
                trg_padding_mask, memory_key_padding_mask):
        embeddings_src = self.embedding_src(src)
        embeddings_trg = self.embedding_trg(trg)

        trans_out = self.trans(embeddings_src, embeddings_trg, src_mask,
                               trg_mask, None,
                               src_padding_mask, trg_padding_mask,
                               memory_key_padding_mask)
        return self.linear(trans_out)

    def encode(self, src, src_mask):
        return self.trans.encoder(self.embedding_src(src), src_mask)

    def decode(self, tgt, memory, tgt_mask):
        return self.trans.decoder(self.embedding_trg(tgt), memory, tgt_mask)
