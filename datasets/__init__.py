import torch.utils.data as data
from .DAVIS_Test import DAVIS_Test
from .YTVOS_Test import YTVOS_Test
from .video_dataset import VIDDEODataset
from .static_dataset import StaticTransformDataset
from .dataloader import train_loader


def get_vos_dataset(config, logger, rank, is_dist, phase='train', cur_iter=0):
    max_iter = 0
    if phase == 'train':
        if config.SOLVER.STAGE == 0:
            # PreTrain
            max_iter = config.SOLVER.PRETRAIN_ITERS[-1]
            dataset = StaticTransformDataset(config.DATASET, logger, rank)
            skip_iters = []
        elif config.SOLVER.STAGE == 1:
            # DAVIS 17
            max_iter = config.SOLVER.DAVIS_ITERS[-1]
            dataset = VIDDEODataset('DAVIS17', config.DATASET, logger, rank, max_iter)
            dataset.set_max_skip(cur_iter)
            skip_iters = dataset.skipper.skip_iters
        elif config.SOLVER.STAGE == 2:
            # YTVOS 19
            max_iter = config.SOLVER.MAINTRAIN_ITERS[-1]
            dataset = VIDDEODataset('YTVOS19', config.DATASET, logger, rank, max_iter)
            dataset.set_max_skip(cur_iter)
            skip_iters = dataset.skipper.skip_iters
        else:
            # DAVIS 17 + YTVOS 19
            max_iter = config.SOLVER.MAINTRAIN_ITERS[-1]
            davis_set = VIDDEODataset('DAVIS17', config.DATASET, logger, rank, max_iter)
            davis_set.set_max_skip(cur_iter)
            ytvos_set = VIDDEODataset('YTVOS19', config.DATASET, logger, rank, max_iter)
            ytvos_set.set_max_skip(cur_iter)
            skip_iters = davis_set.skipper.skip_iters + ytvos_set.skipper.skip_iters
            data_freq = config.DATASET.DATA_FREQ
            dataset = data.ConcatDataset([davis_set] * data_freq[0] + [ytvos_set] * data_freq[1])

        if rank <= 0:
            logger.info(f'Construction DataLoader, start iteration: {cur_iter}, max iteration: {max_iter}')

        data_loader = train_loader(dataset, config.DATALOADER, rank=rank, max_iter=max_iter,
                                   seed=config.DATASET.SEED, is_dist=is_dist,
                                   is_shuffle=True, start_iter=cur_iter)

        return data_loader, max_iter, sorted(list(set(skip_iters)))
    else:
        raise NameError(f'{phase} dataset is not supported.')


def renew_vos_dataset(data_loader, config, logger, rank, is_dist, cur_iter=0):
    if isinstance(data_loader.dataset, data.ConcatDataset):
        datasets = []
        for data_idx, dataset in enumerate(data_loader.dataset.datasets):
            dataset.set_max_skip(cur_iter)
            max_iter = dataset.max_iter
            datasets.append(dataset)
        dataset = data.ConcatDataset(datasets)
    else:
        dataset = data_loader.dataset
        dataset.set_max_skip(cur_iter)
        max_iter = dataset.max_iter

    if rank <= 0:
        logger.info(f'RENEW DataLoader, start iteration: {cur_iter}, max iteration: {max_iter}')
    data_loader = train_loader(dataset, config.DATALOADER, rank=rank, max_iter=max_iter,
                               seed=config.DATASET.SEED, is_dist=is_dist,
                               is_shuffle=True, start_iter=cur_iter)

    return data_loader







