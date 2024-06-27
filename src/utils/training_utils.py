import os

from utils import LOGGER, TQDM



def init_progress_bar(episodes, loss_names):
    header = tuple(['Epoch'] + loss_names)
    LOGGER.info(('\n' + '%15s' * (1 + len(loss_names))) % header)
    pbar = TQDM(range(episodes))
    return pbar


def choose_proper_resume_model(resume_dir, type):
    weights_dir = os.listdir(os.path.join(resume_dir, 'weights'))
    try:
        weight = list(filter(lambda x: type in x, weights_dir))[0]
        return os.path.join(resume_dir, 'weights', weight)
    except IndexError:
        raise IndexError(f"There's no model path in {weights_dir} of type {type}")
    