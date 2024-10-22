from tqdm.notebook import tqdm
import torch
import torch.nn.functional as F

from train import generate_square_subsequent_mask


def translate_beam_s(model, src, start_symbol, train_Dataset, beam_size,
                     test_Dataset, en_vocab, device):
    model.eval()
    result = ""

    for ind, line in enumerate(tqdm(src)):
        line = torch.Tensor(line).unsqueeze(0).to(device)
        max_len = line.shape[1]
        src_mask = torch.zeros((line.shape[1], line.shape[1])).type(torch.bool).to(device)

        trans_line = beam_search(model, line, src_mask, max_len + 5, start_symbol, beam_size, en_vocab, device)

        result += train_Dataset.decode(trans_line.reshape(-1), lng='en') + ' ' + test_Dataset.get_last_word_in_str(ind)
    return result


def beam_search(model, src, src_mask, max_len, start_symbol, beam_size, en_vocab, device):
    memory = model.encode(src, src_mask).to(device)
    tokens = torch.tensor([start_symbol]).unsqueeze(0).to(device)

    sequences = [[tokens, 0, False]]

    for _ in range(max_len - 1):
        all_candidates = []
        eof_cnt = 0
        for seq, score, eof in sequences:
            if eof:
                eof_cnt += 1
                all_candidates.append([seq, score, eof])
                continue

            tgt_mask = generate_square_subsequent_mask(seq.shape[1]).type(torch.bool).to(device)

            out = model.decode(seq, memory, tgt_mask)
            logits = model.linear(out[:, -1])
            proba = F.softmax(logits, dim=1).view(-1)

            proba[en_vocab['<bos>']] = -1
            # proba[en_vocab['<unk>']] = -1
            proba[en_vocab['<pad>']] = -1

            top_scores, top_indices = torch.topk(proba, beam_size)

            for i in range(beam_size):
                is_eof = top_indices[i].item() == en_vocab['<eos>']
                candidate_seq = torch.cat(
                    [seq, top_indices[i].unsqueeze(0).unsqueeze(0)], dim=1)
                candidate_score = score + top_scores[i]
                all_candidates.append([candidate_seq, candidate_score, is_eof])

        ordered = sorted(all_candidates, key=lambda x: x[1] / x[0].shape[1], reverse=True)
        sequences = ordered[:beam_size]
        if eof_cnt == beam_size:
            break

    return sequences[0][0]
