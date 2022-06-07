import os
import torch
import torch.optim as optim


def load_solver(optimizer, lr_scheduler, solver_path):
    loaded_solver = torch.load(solver_path)
    loaded_optimizer = loaded_solver['optimizer']
    loaded_lr_scheduler = loaded_solver['lr_scheduler']
    iteration = loaded_solver['iteration']
    loss = loaded_solver['loss']
    optimizer.load_state_dict(loaded_optimizer)
    lr_scheduler.load_state_dict(loaded_lr_scheduler)

    del loaded_solver
    torch.cuda.empty_cache()

    return iteration, loss


def save_solver(optimizer, lr_scheduler, iteration, loss, solver_path):
    solver = dict()
    solver['optimizer'] = optimizer.state_dict()
    solver['lr_scheduler'] = lr_scheduler.state_dict()
    solver['iteration'] = iteration
    solver['loss'] = loss
    torch.save(solver, solver_path)


def make_optimizer(config_solver, model, logger, rank, num_gpu=None):
    if num_gpu is None:
        lr = config_solver.BASE_LR
    else:
        lr = config_solver.BASE_LR * num_gpu

    if config_solver.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                               lr=lr, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=config_solver.WEIGHT_DECAY)
    elif config_solver.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(filter(lambda x: x.requires_grad, model.parameters()),
                               lr=lr, betas=(0.9, 0.999), eps=1e-8,
                               weight_decay=config_solver.WEIGHT_DECAY)
    elif config_solver.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(filter(lambda x: x.requires_grad, model.parameters()),
                              lr=lr, momentum=config_solver.MOMENTUM,
                              weight_decay=config_solver.WEIGHT_DECAY)
    else:
        raise ValueError('Illegal optimizer.')

    if rank <= 0:
        logger.info('Optimizer: %s. Total params: %.2fM' %
                    (config_solver.OPTIMIZER, sum(p.numel() for p in model.parameters()) / 1000000.0))

    return optimizer


def make_lr_scheduler(config_solver, optimizer, logger, rank):
    if config_solver.STAGE == 0:
        steps = config_solver.PRETRAIN_ITERS
    elif config_solver.STAGE == 1:
        steps = config_solver.DAVIS_ITERS
    else:
        steps = config_solver.MAINTRAIN_ITERS
    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=steps, gamma=config_solver.GAMMA, last_epoch=-1)
    if rank <= 0:
        logger.info(f'MutiStepLR Scheduler with steps: {steps}, gamma: {config_solver.GAMMA}')

    return scheduler


def get_solver(config, model, logger, rank, num_gpu=None):
    optimizer = make_optimizer(config.SOLVER, model, logger, rank, num_gpu)
    lr_scheduler = make_lr_scheduler(config.SOLVER, optimizer, logger, rank)

    cur_iter = 0
    best_loss = 100000000

    if not config.FROM_SCRATCH and config.RESUME is not None:
        solver_path = os.path.join(config.RESUME, 'solvers', f'{config.MODEL.MODEL_NAME}.solver')
        cur_iter, best_loss = load_solver(optimizer, lr_scheduler, solver_path)

    return optimizer, lr_scheduler, cur_iter, best_loss


