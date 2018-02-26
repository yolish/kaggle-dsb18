import pandas as pd
import numpy as np
from skimage.io import imread
from dsbutils import collect_dataset
from dsbaugment import to_binary_mask
from torch.utils.data import Dataset
from dsbml import calc_expected_iou, try_add_weight_map


class NucleiDataset(Dataset):
    """Nuclei dataset."""

    def __init__(self, type, imgs_df=None, dataset=None,
                 labels_file=None, transform=None, img_channels=3):
        '''

        :param imgs_df: a Pandas dataframe with details about the images
        :param labels_file: a csv file with the labels, optional
        :param transform: a transformation to apply on the images, optional
        '''
        assert(imgs_df is not None or dataset is not None)
        if type == 'train':
            assert(labels_file is not None)

        if imgs_df is not None:
            self.dataset = collect_dataset(imgs_df, type, labels_file)
        else:
            self.dataset = dataset
        self.type = type
        self.transform = transform
        self.img_channels = img_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_record = self.dataset.iloc[idx,]
        img_path = img_record['ImagePath']
        img_id = img_record['ImageId']
        img = imread(img_path)[:,:,:self.img_channels]
        sample = {'id':img_id, 'img':img, 'size':img.shape}
        if self.type != 'test':
            mask_paths = img_record['MaskPaths']
            mask = self.combine_masks(mask_paths)
            sample['labelled_mask'] = mask # only used for evaluation and plotting
            binary_mask, borders = to_binary_mask(mask)
            sample['binary_mask'] = binary_mask
            sample['borders'] = borders
            if self.type == 'train':
                sample['expected_iou'] = calc_expected_iou(mask, binary_mask)


        if self.transform is not None:
            sample = self.transform(sample)

        try_add_weight_map(sample)

        return sample

    def split(self, frac, type, transform = None, filename=None):
        # split to 2 new datasets
        if filename is not None:
            split_out_df = pd.read_csv(filename)
            img_ids = split_out_df['ImageId'].values
            split_out_dataset = self.dataset.loc[self.dataset['ImageId'].isin(img_ids)]
            self.dataset = self.dataset.loc[~self.dataset['ImageId'].isin(img_ids)]
        else:
            total_size = self.__len__()
            sample_size = int(total_size*frac)
            sampled_idx = np.random.choice(total_size, replace=False, size=sample_size)
            remaining_idx = [idx for idx in xrange(total_size) if idx not in sampled_idx]
            split_out_dataset = self.dataset.iloc[sampled_idx]
            self.dataset = self.dataset.iloc[remaining_idx]
        return NucleiDataset(type, dataset=split_out_dataset, transform = transform)

    def combine_masks(self, masks_paths):
        # the combined mask has a different label for each mask
        combined_mask = imread(masks_paths[0])
        combined_mask[combined_mask > 0] = 1
        for i in xrange(1, len(masks_paths)):
            mask = imread(masks_paths[i])
            combined_mask[mask > 0] = i + 1
        return combined_mask






