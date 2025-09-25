from abc import abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

from ..extras import logging
from .data_utils import Role

if TYPE_CHECKING:
    from ..hparams import DataArguments
    from .mm_plugin import AudioInput, ImageInput, VideoInput
    from .parser import DatasetAttr

    MediaType = Union[ImageInput, VideoInput, AudioInput]

logger = logging.get_logger(__name__)

@dataclass
class DatasetConverter:
    dataset_attr: "DatasetAttr"
    data_args: "DataArguments"

    def _find_medias(self, medias: Union["MediaType", list["MediaType"], None]) -> Optional[list["MediaType"]]:
        r"""Optionally concatenate media path to media dir when loading from local disk."""
        if medias is None:
            return None
        elif not isinstance(medias, list):
            medias = [medias]
        elif len(medias) == 0:
            return None
        else:
            medias = medias[:]

        if self.dataset_attr.load_from in ["script", "file"] and isinstance(medias[0], str):
            for i in range(len(medias)):
                if os.path.isfile(os.path.join(self.data_args.media_dir, medias[i])):
                    medias[i] = os.path.join(self.data_args.media_dir, medias[i])
                else:
                    logger.warning_rank0_once(f"Media {medias[i]} does not exist in `media_dir`. Use original path.")

        return medias

    @abstractmethod
    def __call__(self, example: dict[str, Any]) -> dict[str, Any]:
        r"""Convert a single example in the dataset to the standard format."""
        ... 