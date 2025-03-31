from abc import abstractmethod
from pathlib import Path
from typing import Any, Generic

import torch

from lambeq.core.globals import VerbosityLevel
from lambeq.core.utils import (SentenceBatchType,
                               tokenised_batch_type_check,
                               untokenised_batch_type_check)
from lambeq.text2diagram.base import Reader
from lambeq.text2diagram.model_based_reader.model_downloader import (
    ModelDownloader,
    ModelDownloaderError,
    MODELS
)
from lambeq.typing import ModelT, StrPathT


class ModelBasedReader(Reader):
    """Base class for readers that use pre-trained models.

    This is an abstract base class that provides common functionality for
    model-based readers. Subclasses must implement the specific model
    initialization and inference logic.
    """

    def __init__(
        self,
        model_name_or_path: str | None = None,
        device: int = -1,
        cache_dir: StrPathT | None = None,
        force_download: bool = False,
        verbose: str = VerbosityLevel.PROGRESS.value,
    ) -> None:
        """Initialise the model-based reader.

        Parameters
        ----------
        model_name_or_path : str, default: 'bert'
            Can be either:
                - The path to a directory containing a model.
                - The name of a pre-trained model.
        device : int, default: -1
            The GPU device ID on which to run the model, if positive.
            If negative (the default), run on the CPU.
        cache_dir : str or os.PathLike, optional
            The directory to which a downloaded pre-trained model should
            be cached instead of the standard cache.
        force_download : bool, default: False
            Force the model to be downloaded, even if it is already
            available locally.
        verbose : str, default: 'progress'
            See :py:class:`VerbosityLevel` for options.
        """

        self.model_name_or_path = model_name_or_path
        self.device = device
        self.cache_dir = cache_dir
        self.force_download = force_download
        self.verbose = verbose
        self.model_dir: StrPathT | None = None

        # Prepare model artifacts
        self._prepare_model_artifacts()

    def _prepare_model_artifacts(self) -> None:
        """Download model artifacts to disk."""
        self.model_dir = Path(self.model_name_or_path)

        if not self.model_dir.is_dir():
            # Check for updates only if a local model path is not
            # specified in `self.model_name_or_path`
            downloader = ModelDownloader(self.model_name_or_path,
                                         self.cache_dir)
            self.model_dir = downloader.model_dir
            if (self.force_download
                    or not self.model_dir.is_dir()
                    or downloader.model_is_stale()):
                try:
                    downloader.download_model(self.verbose)
                except ModelDownloaderError as e:
                    local_model_version = downloader.get_local_model_version()

                    if (self.model_dir.is_dir()
                            and local_model_version is not None):
                        print('Failed to update model with '
                              f'exception: {e}')
                        print('Attempting to continue with version '
                              f'{local_model_version}')
                    else:
                        # No local version to fall back to
                        raise e

    @abstractmethod
    def _initialise_model(self, **kwargs: Any) -> None:
        """Initialise the model and put it into the appropriate device.

        Also handle required miscellaneous initialisation steps here."""

    def get_device(self) -> torch.device:
        return torch.device('cpu' if self.device < 0
                            else f'cuda:{self.device}')

    def validate_sentence_batch(
        self,
        sentences: SentenceBatchType,
        tokenised: bool = False,
        suppress_exceptions: bool = False,
    ) -> tuple[SentenceBatchType, list[int]]:
        """Prepare input sentences for parsing.

        Parameters
        ----------
        sentences : list of str, or list of list of str
            The sentences to be parsed, passed either as strings or as
            lists of tokens.
        suppress_exceptions : bool, default: False
            Whether to suppress exceptions. If :py:obj:`True`, then if a
            sentence fails to parse, instead of raising an exception,
            its return entry is :py:obj:`None`.
        tokenised : bool, default: False
            Whether each sentence has been passed as a list of tokens.
        verbose : str, optional
            See :py:class:`VerbosityLevel` for options. If set, takes
            priority over the :py:attr:`verbose` attribute of the
            parser.

        Returns
        -------
        SentenceBatchType
            List of (tokenised or untokenised) sentences
        """
        if tokenised:
            if not tokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `True`, but variable '
                                 '`sentences` does not have type '
                                 '`List[List[str]]`.')
        else:
            if not untokenised_batch_type_check(sentences):
                raise ValueError('`tokenised` set to `False`, but variable '
                                 '`sentences` does not have type '
                                 '`List[str]`.')
            sent_list: list[str] = [str(s) for s in sentences]
            sentences = [sentence.split() for sentence in sent_list]

        # Remove empty sentences
        empty_indices = []
        for i, sentence in enumerate(sentences):
            if not sentence:
                if suppress_exceptions:
                    empty_indices.append(i)
                else:
                    raise ValueError('sentence is empty.')

        for i in reversed(empty_indices):
            del sentences[i]

        return sentences, empty_indices

    @staticmethod
    def available_models() -> list[str]:
        """List the available models."""
        return [*MODELS]
