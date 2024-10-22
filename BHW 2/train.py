from tqdm.notebook import tqdm
import torch


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
        mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[1]
    tgt_seq_len = tgt.shape[1]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len)).type(torch.bool)
    PAD_IDX = 0
    src_padding_mask = (src == PAD_IDX)
    tgt_padding_mask = (tgt == PAD_IDX)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


def training_epoch(model, optimizer, criterion, train_loader, device, tqdm_desc):
    train_loss = 0.0
    model.train()

    for de_text, en_text in tqdm(train_loader, desc=tqdm_desc):
        de_text = de_text.to(device)
        en_text = en_text.to(device)

        tgt_input = en_text[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            de_text, tgt_input)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        src_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)

        optimizer.zero_grad()
        logits = model(de_text, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = en_text[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]),
                         tgt_out.reshape(-1))

        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(list(train_loader))
    return train_loss


@torch.no_grad()
def validation_epoch(model, criterion, test_loader, device, tqdm_desc):
    test_loss = 0.0
    model.eval()

    for de_text, en_text in tqdm(test_loader, desc=tqdm_desc):
        de_text = de_text.to(device)
        en_text = en_text.to(device)

        tgt_input = en_text[:, :-1]
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(
            de_text, tgt_input)
        src_mask = src_mask.to(device)
        tgt_mask = tgt_mask.to(device)
        src_padding_mask = src_padding_mask.to(device)
        tgt_padding_mask = tgt_padding_mask.to(device)

        logits = model(de_text, tgt_input, src_mask, tgt_mask,
                       src_padding_mask, tgt_padding_mask, src_padding_mask)

        tgt_out = en_text[:, 1:]
        loss = criterion(logits.reshape(-1, logits.shape[-1]),
                         tgt_out.reshape(-1))

        test_loss += loss.item()

    test_loss /= len(list(test_loader))
    return test_loss


def train(model, optimizer, scheduler, criterion, train_loader, test_loader,
          num_epochs, swa_model, swa_start, device):
    train_losses = []
    test_losses = []

    for epoch in range(1, num_epochs + 1):
        train_loss = training_epoch(
            model, optimizer, criterion, train_loader, device,
            tqdm_desc=f'Training {epoch}/{num_epochs}'
        )
        test_loss = validation_epoch(
            model, criterion, test_loader, device,
            tqdm_desc=f'Validating {epoch}/{num_epochs}'
        )

        if epoch >= swa_start:
            swa_model.update_parameters(model)

        if scheduler is not None:
            scheduler.step()

        train_losses += [train_loss]
        test_losses += [test_loss]
        # torch.save(model.state_dict(), "weights.pt")
        print(train_losses[-1], test_losses[-1])

    torch.optim.swa_utils.update_bn(train_loader, swa_model)

    return train_losses, test_losses
