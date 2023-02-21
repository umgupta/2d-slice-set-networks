import logging

from box import Box
from torchvision.transforms import transforms

from src.common.data.adni import ADNI
from src.common.data.ukbb_brain_age import UKBBBrainAGE
from src.common.data_utils import frame_drop

logger = logging.getLogger()


def get_dataset(
        name, test_csv=None, train_csv=None, valid_csv=None, root_path=None,
        train_num_sample=-1, frame_keep_style="random", frame_keep_fraction=1.0,
        frame_dim=1, impute=False, **kwargs
):
    """
    train_num_sample: How many training samples to keep
    frame_dim : This is for using with adding noise and removing frame options,
                we use it to alter frame in particular direction.
                1 for sagittal, 2 for coronal, 3 for axial
    frame_keep_style: This is for using with frame_keep_fraction. Options are "ordered" or "random"
                    if ordered every kth frame is dropped, where k is computed from fraction
                    If random, frame are removed randomly
    impute: whether to impute data or not when removing frames.
                    if "zeros" : impute with zeros
                    if "noise" : impute with U(min, max)
                    if "fill" : find the nearest available frame and fill with it.
                    if anything else, we just remove the frames or make scan smaller (drop frames)

    """

    if name == "ukbb_brain_age":
        # Transformations to remove frames
        frame_drop_transform = lambda x: frame_drop(
            x, frame_keep_style=frame_keep_style,
            frame_keep_fraction=frame_keep_fraction,
            frame_dim=frame_dim, impute=impute
        )
        transform = transforms.Compose([frame_drop_transform])
        train_data = UKBBBrainAGE(
            root=root_path, metadatafile=train_csv,
            num_sample=train_num_sample, transform=transform
        )
        test_data = UKBBBrainAGE(root=root_path, metadatafile=test_csv, transform=transform)
        valid_data = UKBBBrainAGE(
            root=root_path, metadatafile=valid_csv if valid_csv else test_csv,
            transform=transform
        )
        return Box({"train": train_data, "test": test_data, "valid": valid_data}), {}

    if name == "adni_ad":
        # Transformations to remove frames
        frame_drop_transform = lambda x: frame_drop(
            x, frame_keep_style=frame_keep_style,
            frame_keep_fraction=frame_keep_fraction,
            frame_dim=frame_dim, impute=impute
        )
        transform = transforms.Compose([frame_drop_transform])
        train_data = ADNI(
            root=root_path, metadatafile=train_csv,
            num_sample=train_num_sample, transform=transform
        )
        test_data = ADNI(root=root_path, metadatafile=test_csv, transform=transform)
        valid_data = ADNI(
            root=root_path, metadatafile=valid_csv if valid_csv else test_csv,
            transform=transform
        )
        return Box({"train": train_data, "test": test_data, "valid": valid_data}), {}

    logger.error(f"Invalid data name {name} specified")
    raise Exception(f"Invalid data name {name} specified")
