
from datasets.mevis.mevis_dataset import MeViSDataset
from datasets.coco.refercoco import ModulatedDetection
import os
from misc import nested_tensor_from_videos_list


class Collator:
    def __call__(self, batch):
        samples, targets = list(zip(*batch))
        samples = nested_tensor_from_videos_list(samples, size_divisibility=32)   # [B, T, C, H, W]
        caption = [t.pop('caption') for t in targets]
        batch_dict = {
            'samples': samples,
            'targets': targets,
            'text_queries': caption
        }
        return batch_dict


def build_dataset(image_set, dataset_file, use_random_sample=None, **kwargs):
    if dataset_file == 'ref_youtube_vos':
        return ReferYouTubeVOSDataset(image_set, **kwargs)
    elif dataset_file == 'davis':
        return ReferDavisDataset(image_set, **kwargs)
    elif dataset_file == 'mevis':
        return MeViSDataset(image_set, **kwargs)
    elif dataset_file == 'refcoco' or dataset_file == 'refcoco+' or dataset_file == 'refcocog':
        kwargs['ann_file'] = os.path.join(kwargs['ann_file'], dataset_file, 'instances_{}_{}.json'.format(dataset_file, image_set))
        return ModulatedDetection(image_set, **kwargs)
    raise ValueError(f'dataset {dataset_file} not supported')
