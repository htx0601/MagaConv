import argparse
import os
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import CompletionDataset
import utils
from model.Network import Network
from loss import Loss_with_laplace
import torch.optim.lr_scheduler as lrs

parser = argparse.ArgumentParser(description='PyTorch MaGNet Training')
parser.add_argument('--data_root', default='/data/datasets', type=str, help='path to dataset root')
parser.add_argument('--project_root', default='/data/experiments', type=str, help='path to save logs/models')

parser.add_argument('--datasets', default='NYUDepth', type=str, help='dataset name (e.g., NYUDepth, DIML, SUNRGBD)')
parser.add_argument('--inner_channels', default=64, type=int, help='inner_channel')
parser.add_argument('--layers', default=4, type=int, help='layers')

parser.add_argument('--lr', default=0.001, type=float, help='learning rate')
parser.add_argument('--momentum', default=0.95, type=float, help='sgd momentum')
parser.add_argument('--weight_decay', default=1e-4, type=float, help='weight decay')
parser.add_argument('--dampening', default=0.0, type=float, help='dampening for momentum')
parser.add_argument('--nesterov', '-n', action='store_true', help='enables Nesterov momentum')
parser.add_argument('--num_epoch', default=100, type=int, help='number of epochs')

parser.add_argument('--batch_size_train', default=8, type=int, help='batch size for training')
parser.add_argument('--batch_size_eval', default=1, type=int, help='batch size for eval')
parser.add_argument('--resume', '-r', default=False, action='store_true', help='resume from checkpoint')

args = parser.parse_args()

save_dir = os.path.join(args.project_root, f"result_{args.datasets}")
save_img_dir = os.path.join(save_dir, "img")
utils.log_file_folder_make_lr(save_dir)

dataset_train = CompletionDataset(args.data_root, args.datasets, 'train')
dataset_test = CompletionDataset(args.data_root, args.datasets, 'test')
train_loader = DataLoader(dataset_train, batch_size=args.batch_size_train, shuffle=True, drop_last=True)
eval_loader = DataLoader(dataset_test, batch_size=args.batch_size_eval, shuffle=False)

model = Network().cuda()
loss_fn = Loss_with_laplace()

if args.resume:
    print("============> Load Pretrained Model <============")
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    if os.path.isfile(best_model_path):
        best_model_dict = torch.load(best_model_path)
        best_model_dict = utils.remove_moudle(best_model_dict)
        model.load_state_dict(utils.update_model(model, best_model_dict))
    else:
        print("Checkpoint not found, training from scratch.")

optimizer = torch.optim.SGD(model.parameters(),
                            lr=args.lr,
                            momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            nesterov=args.nesterov,
                            dampening=args.dampening)
scheduler = lrs.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=20, min_lr=1e-5)

best_delta = 1

def train(epoch):
    model.train()
    total_step_train = 0
    train_loss = 0.0
    error_sum_train = utils.init_error_metrics()

    tbar = tqdm(train_loader)
    for batch_idx, data in enumerate(tbar):
        color, depth, mask, gt = (Variable(data[k]).cuda() for k in ['rgb', 'depth', 'mask', 'gt'])
        optimizer.zero_grad()
        output = model(color, depth, mask)
        loss = loss_fn(output, gt, mask)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        tbar.set_description(f"Epoch {epoch}, loss={train_loss / (batch_idx + 1):.5f}")

        error_result = utils.evaluate_error(gt, output, mask, False)
        total_step_train += args.batch_size_train
        error_avg = utils.avg_error(error_sum_train, error_result, total_step_train, args.batch_size_train)

        if batch_idx % 50 == 0:
            utils.print_error('training_result: step(average)',
                              epoch, batch_idx, loss,
                              error_result, error_avg, print_out=True
                              )
            record_loss = utils.save_error(
                epoch, batch_idx, loss,
                error_result, error_avg, print_out=False
            )
            utils.log_loss_lr(save_dir, record_loss, 'train')

    error_avg = utils.avg_error(error_sum_train,
                                error_result,
                                total_step_train,
                                args.batch_size_train)

    for param_group in optimizer.param_groups:
        old_lr = float(param_group['lr'])

    utils.log_result_lr(save_dir, error_avg, epoch, old_lr, False, 'train')

    torch.save(model.state_dict(), os.path.join(save_dir, "latest_model.pth"))
    if epoch % 5 == 0:
        torch.save(model.state_dict(), os.path.join(save_dir, f"epoch_{epoch:02d}.pth"))

    scheduler.step(error_avg['RMSE'], epoch)

def eval(epoch):
    global best_delta
    model.eval()
    total_step_val = 0
    eval_loss = 0.0
    error_sum_val = utils.init_error_metrics()
    is_best_model = False

    tbar = tqdm(eval_loader)
    for batch_idx, data in enumerate(tbar):
        with torch.no_grad():
            color, depth, mask, gt = (Variable(data[k]).cuda() for k in ['rgb', 'depth', 'mask', 'gt'])
            output = model(color, depth, mask)
            loss = loss_fn(output, gt, mask)

        eval_loss += loss.item()
        tbar.set_description(f"Epoch {epoch}, loss={eval_loss / (batch_idx + 1):.4f}")

        error_result = utils.evaluate_error(gt, output, mask, False)
        utils.save_eval_img(save_img_dir, batch_idx, depth[0].cpu() / 10, gt[0].cpu() / 10, output[0].cpu() / 10)
        total_step_val += args.batch_size_eval
        error_avg = utils.avg_error(error_sum_val, error_result, total_step_val, args.batch_size_eval)

        if batch_idx % (args.batch_size_eval * 10) == 0:
            record_loss = utils.save_error(
                epoch, batch_idx, loss,
                error_result, error_avg, print_out=False
            )
            utils.log_loss_lr(save_dir, record_loss, 'eval')

    utils.print_error('eval_result: step(average)',
                      epoch, batch_idx, loss,
                      error_result, error_avg, print_out=False
                      )

    if utils.update_best_model(error_avg, best_delta):
        is_best_model = True
        best_delta = min(error_avg['RMSE'], 10)

    utils.log_result_lr(save_dir, error_avg, epoch, optimizer.param_groups[0]['lr'], is_best_model, 'eval')

    if is_best_model:
        print(f'==> saving best model at epoch {epoch}')
        torch.save(model.state_dict(), os.path.join(save_dir, 'best_model.pth'))

if __name__ == '__main__':
    for epoch in range(1, args.num_epoch):
        train(epoch)
        eval(epoch)
