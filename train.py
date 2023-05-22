
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import dgl
from dataset import NpzData, collate
import datetime
from utils import EarlyStopping
from sklearn.metrics import accuracy_score
import numpy as np
from collections import OrderedDict
import random
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as multiprocessing
from torch.nn.parallel import DistributedDataParallel
from torch.cuda.amp import autocast, GradScaler
from network import GNNSAT, GNNSATAttention


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    dgl.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(args, proc_id, devices, n_gpus, stop):
    device = devices[proc_id]
    if n_gpus > 1:
        dist_init_method = 'tcp://{master_ip}:{master_port}'.format(
            master_ip='127.0.0.1', master_port=args.port)
        world_size = n_gpus
        torch.distributed.init_process_group(backend="nccl",
                                             init_method=dist_init_method,
                                             world_size=world_size,
                                             rank=proc_id)
    train_dataset = NpzData(args.train_path)
    sampler = DistributedSampler(
        train_dataset, num_replicas=n_gpus, rank=proc_id, shuffle=True) if args.is_distributed else None
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler, batch_size=args.batch_size, num_workers=args.num_workers,
        shuffle=(sampler is None), collate_fn=collate, pin_memory=args.pin_memory)

    # each relationship is completed by two edges, because DGL does not support undirected graphs
    # messages are delivered according to the order of relations
    relations = OrderedDict()
    relations[('var', 'clause')] = ['vcp', 'vcn']
    relations[('clause', 'var')] = ['cvp', 'cvn']

    net_args = (128, 2, ('var', 'clause'),
                ('cvp', 'cvn', 'vcp', 'vcn'),
                relations, args.step, args.assign, args.use_var)

    if args.attention:
        model = GNNSATAttention(*net_args)
    else:
        model = GNNSAT(*net_args)

    model = model.to(device)

    if n_gpus > 1:
        model = DistributedDataParallel(
            model, device_ids=[device], output_device=device, find_unused_parameters=True)

    start_epoch = 0
    if args.resume:
        model.load_state_dict(torch.load(
            args.resume_model, map_location=torch.device(device)))
        start_epoch = args.resume_epoch-1
    loss_func = nn.BCEWithLogitsLoss().to(device)

    weight_p, bias_p = [], []

    for name, p in model.named_parameters():
        if 'bias' in name:
            bias_p += [p]
        else:
            weight_p += [p]

    if args.optimizer == 'adam':
        optimizer = optim.Adam([
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ], lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = optim.SGD([
            {'params': weight_p, 'weight_decay': args.weight_decay},
            {'params': bias_p, 'weight_decay': 0}
        ],
            lr=args.learning_rate, momentum=0.9)
    else:
        raise RuntimeError('unsupport optimizer {}!'.format(args.optimizer))

    scheduler = ReduceLROnPlateau(
        optimizer, patience=3, factor=args.gamma, mode='max', verbose=True)

    if proc_id == 0:
        dt = datetime.datetime.now()
        prefix = 'satisfied_{}_{:02d}-{:02d}-{:02d}'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        stopper = EarlyStopping(args.save_dir, (prefix), 10)

    scaler = GradScaler()
    epoch_losses = []
    for epoch in range(start_epoch, args.epoch):
        model.train()

        if proc_id == 0:
            epoch_loss = 0

        if args.is_distributed:
            sampler.set_epoch(epoch)

        predictions = []
        labels = []
        for iter, (bg, label, assigns) in enumerate(train_dataloader):
            optimizer.zero_grad()

            bg = bg.to(torch.device(device))
            label = label.to(device)
            with autocast():
                if args.assign:
                    prediction, assign_prediction = model(bg)
                else:
                    prediction = model(bg)

                loss = loss_func(prediction, label)

                if args.assign:
                    assign_loss = 0
                    num = 0
                    for i, assign in enumerate(assigns):
                        if assign is not None:
                            assign = assign.to(device)
                            assign_loss += loss_func(
                                assign_prediction[i], assign)
                            num += 1
                    if num > 0:
                        loss += assign_loss/num

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()

            if proc_id == 0:
                predictions.append(torch.argmax(
                    prediction, dim=1).detach().cpu())
                labels.append(torch.argmax(label, dim=1).detach().cpu())
                epoch_loss += loss.detach().item()

                print('Epoch {}, iter {} loss {:.4f}'.format(
                    epoch+1, iter+1, loss.detach().item()))

        if proc_id == 0:
            acc = 0

            for prediction, label in zip(predictions, labels):
                acc += accuracy_score(prediction.numpy(),
                                      label.numpy())

            epoch_loss /= (iter + 1)
            print('Epoch {}, Avg loss {:.4f} Accuracy {:.4f}'.format(
                epoch+1, epoch_loss, acc/(iter+1)))
            epoch_losses.append(epoch_loss)

            val_dataset = NpzData(args.validation_path)
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset, batch_size=args.batch_size, num_workers=args.num_workers,
                shuffle=True, collate_fn=collate, pin_memory=args.pin_memory)

            avg_loss, val_acc = evaluate(
                model, val_dataloader, loss_func, args.assign, device)

            print('Validate Avg loss {:.4f}, Avg Accuracy {:.4f}'.format(
                avg_loss, val_acc))

            early_stop = stopper.step(avg_loss, val_acc, (model))

            if n_gpus > 1:
                stop.value = early_stop
            elif early_stop:
                break
            scheduler.step(val_acc)
        if n_gpus > 1:
            torch.distributed.barrier()
            if stop.value:
                break
    if n_gpus > 1:
        torch.distributed.barrier()


def evaluate(model, dataloader, loss_func, assign, device):
    with torch.no_grad():
        model.eval()
        loss_sum = 0
        acc_sum = 0
        for iter, (bg, label, assigns) in enumerate(dataloader):
            bg = bg.to(torch.device(device))
            label = label.to(device)
            if assign:
                prediction, assign_prediction = model(bg)
            else:
                prediction = model(bg)
            loss = loss_func(prediction, label)
            if assign:
                assign_loss = 0
                num = 0
                for i, target in enumerate(assigns):
                    if target is not None:
                        target = target.to(device)
                        assign_loss += loss_func(
                            assign_prediction[i], target)
                        num += 1
                if num > 0:
                    loss += assign_loss/num

            loss_sum += loss.detach().item()

            classifier = torch.argmax(prediction, dim=1).detach().cpu().numpy()
            label = torch.argmax(label, dim=1).detach().cpu().numpy()
            acc = accuracy_score(classifier, label)
            acc_sum += acc
            print('Iter {} Validate Accuracy {:.4f}'.format(iter+1, acc))
        avg_acc = acc_sum/(iter+1)
        avg_loss = loss_sum/(iter+1)
        return avg_loss, avg_acc


def main(args, devices):
    devices = list(map(int, args.gpu.split(',')))
    n_gpus = len(devices)

    if devices[0] == -1:
        train(args, 0, ['cpu'], n_gpus, None)
    if n_gpus == 1:
        train(args, 0, devices, n_gpus, None)
    else:
        mp = multiprocessing.get_context('spawn')
        stop = mp.Value('i', False)
        procs = []
        for proc_id in range(n_gpus):
            p = mp.Process(target=train,
                           args=(args, proc_id, devices, n_gpus, stop))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    set_seed(123456)
    import argparse
    import ast
    parser = argparse.ArgumentParser()
    parser.add_argument('-T', '--attention', type=ast.literal_eval, default=True,
                        help='use attention (default: False')
    parser.add_argument('-t', '--train_path', type=str, default='data/satlab/100/train.txt',
                        help='where you store train data (default: data/satlab/100/train.txt)')
    parser.add_argument('-v', '--validation_path', type=str, default='data/satlab/100/valid.txt',
                        help='where you store validation data (default: data/satlab/100/valid.txt)')
    parser.add_argument('-V', '--use_var', type=ast.literal_eval, default=False,
                        help='use var to predict (default: False)')
    parser.add_argument('-a', '--save_dir', type=str, default='model/100',
                        help='where you store model (default: model/100)')
    parser.add_argument('-l', '--learning_rate', type=float, default=1e-3,
                        help='learning_rate (default: 1e-3)')
    parser.add_argument('-d', '--weight_decay', type=float, default=1e-8,
                        help='weight_decay (default: 1e-8)')
    parser.add_argument('-e', '--epoch', type=int, default=100,
                        help='epoch number (default: 100)')
    parser.add_argument('-w', '--assign', type=ast.literal_eval, default=True,
                        help='assign (default: True)')
    parser.add_argument('-z', '--step', type=int, default=9,
                        help='step (default: 9)')
    parser.add_argument('-o', '--optimizer', type=str, default='adam',
                        help='optimizer, can be adam or sgd (default: adam)')
    parser.add_argument('-k', '--gamma', type=float, default=0.5,
                        help='learning rate decay (default: 0.5)')
    parser.add_argument('-s', '--embed_size', type=int, default=128,
                        help='embed_size (default: 128)')
    parser.add_argument('-b', '--batch_size', type=int, default=32,
                        help='batch_size (default: 32)')
    parser.add_argument('-j', '--pin_memory', type=ast.literal_eval, default=True,
                        help='pin_memory (default: True)')
    parser.add_argument('-n', '--num_workers', type=int, default=4,
                        help='how many workers you want to use (default: 4)')
    parser.add_argument('-P', '--port', type=int, default=12345,
                        help='master port (default: 12345)')
    parser.add_argument('-g', '--gpu', type=str, default='0',
                        help="GPU, can be a list of gpus for multi-gpu trianing, e.g., 0,1,2,3; -1 for CPU (default: 0)")
    parser.add_argument('-u', '--is_distributed', type=ast.literal_eval, default=False,
                        help="whether to use distributed (default: False)")
    parser.add_argument('-r', '--resume', type=ast.literal_eval, default=False,
                        help='whether to resume model (default: False)')
    parser.add_argument('-p', '--resume_model', type=str,
                        help='the model path you resume')
    parser.add_argument('-i', '--resume_epoch', type=int, default=1,
                        help='start epoch number (default: 1)')
    args = parser.parse_args()

    print(args)

    devices = list(map(int, args.gpu.split(',')))
    main(args, devices)
