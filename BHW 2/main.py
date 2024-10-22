import torch
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.optim.swa_utils import AveragedModel

from lang_dataset import LangDataset
from transformer import MyTransformer
from scheduler import MyScheduler
from train import train
from generate import translate_beam_s


def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(torch.Tensor(src_sample))
        tgt_batch.append(torch.Tensor(tgt_sample))

    return (pad_sequence(src_batch, batch_first=True, padding_value=0).type(torch.LongTensor),
            pad_sequence(tgt_batch, batch_first=True, padding_value=0).type(torch.LongTensor))


def create_vocab(files: list, t=1):
    text = []
    max_len = 0
    for file_source in files:
        with open(file_source) as file:
            for line in file.readlines():
                line = line.replace('\n', '')

                new_line = line.split(' ')
                text.append(new_line)
                if len(new_line) > max_len:
                    max_len = len(new_line)

    vocab = build_vocab_from_iterator(text, specials=['<pad>', '<bos>', '<eos>', '<unk>'], min_freq=t)
    vocab.set_default_index(vocab['<unk>'])
    return vocab, max_len + 2


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Наш код будет работать на', device)

    # создаем вокабуляр
    en_vocab, max_en = create_vocab(['data/train.de-en.en'], t=8)
    de_vocab, max_de = create_vocab(['data/train.de-en.de'], t=8)
    max_str_size = max(max_en, max_de)

    # создаем датасет
    train_Dataset = LangDataset('data/train.de-en.de', de_vocab, max_str_size,
                                'data/train.de-en.en', en_vocab, cut=False)
    val_Dataset = LangDataset('data/val.de-en.de', de_vocab,
                              max_str_size, 'data/val.de-en.en', en_vocab)
    test_Dataset = LangDataset('data/test1.de-en.de', de_vocab, max_str_size)

    # создаем уже лоадер для дальнейшей работы
    train_loader = DataLoader(train_Dataset, batch_size=64, shuffle=False,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_Dataset, batch_size=64, shuffle=False,
                            collate_fn=collate_fn)
    test_de_loader = DataLoader(test_Dataset, batch_size=1, shuffle=False)

    # Инициализируем
    num_epochs = 16
    warm_epoch = 5
    init_lr = 1e-8
    end_lr = 1e-3
    model = MyTransformer(len(de_vocab), len(en_vocab), 8,
                          train_Dataset.pad_index, max_str_size, 512, 3, 3,
                          512, 0.1)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    model = model.to(device)

    swa_model = AveragedModel(model)
    swa_start = 11

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-8, betas=(0.9, 0.98), eps=1e-9)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=train_Dataset.pad_index, label_smoothing=0.1)
    scheduler = MyScheduler(optimizer, warm_epoch, init_lr, end_lr, num_epochs - warm_epoch)

    train_losses, test_losses = train(
        model, optimizer, scheduler, criterion, train_loader, val_loader,
        num_epochs, swa_model, swa_start, device
    )
    result = translate_beam_s(model, test_Dataset, en_vocab['<bos>'], train_Dataset, 5, test_Dataset, en_vocab, device)
    with open('test.de-en.en', 'w') as file:
        file.write(result)

    torch.save(model.state_dict(), "model_weights_en8de8_basepar.pt")


if __name__ == '__main__':
    main()
