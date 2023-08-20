def train(config, model, dl, opt, scheduler, epoch,mixup_fn=None,criterion=nn.CrossEntropyLoss()):
    model.train()
    model = model.cuda()
    for ep in tqdm(range(epoch)):
        model.train()
        model = model.cuda()
        # pbar = tqdm(dl)
        for i, batch in enumerate(dl):
            x, y = batch[0].cuda(), batch[1].cuda()
            if mixup_fn is not None:
                x,y=mixup_fn(x,y)
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        if scheduler is not None:
            scheduler.step(ep)
        if ep % 10 == 9:
            acc = test(model, test_dl)
            if acc > config['best_acc']:
                config['best_acc'] = acc
                save(config['method'], config['name'], model, acc, ep)
    model = model.cpu()
    return model

config = get_config(args.method, args.dataset,args.few_shot)
train_dl, test_dl = get_data(args.dataset,few_shot=args.few_shot)
