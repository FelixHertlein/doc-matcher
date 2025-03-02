from functools import partial
import string
from torch.nn.functional import one_hot
import math
from einops import rearrange
from typing import Any, Dict, List, Optional, Tuple
import torch
from torch.nn.init import trunc_normal_

from torch.nn.functional import interpolate


class RandomShuffle(object):

    def __call__(self, data: Dict) -> Dict:
        template_features = data["raw_data"]["template_features"]
        warpedgt_features = data["raw_data"]["warpedgt_features"]

        data["raw_data"]["template_features"] = self._random_shuffle(template_features)
        data["raw_data"]["warpedgt_features"] = self._random_shuffle(warpedgt_features)

        return data

    def _random_shuffle(
        self, features: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        perm = torch.randperm(len(features))
        feature_keys = list(features.keys())
        feature_vals = list(features.values())

        return {feature_keys[i]: feature_vals[i] for i in perm}


class RandomSubsampleCorrespondences(object):

    def __init__(self, num_features: int = 20, pairs_ratio: float = 0.666) -> None:
        self.num_features = num_features
        self.pairs_ratio = pairs_ratio

    def __call__(self, data: Dict) -> Dict:
        num_available_features = len(data["raw_data"]["template_features"])
        assert (
            num_available_features >= self.num_features
        ), f"Not enough features to subsample! Only {num_available_features} of {self.num_features} available."

        # calculate the minimum number of pairs such that the given amount of features can be reached
        min_num_pairs = self.num_features - (num_available_features - self.num_features)

        # calculate the number of pairs and singles
        num_pairs = max(min_num_pairs, math.ceil(self.num_features * self.pairs_ratio))
        num_singles = self.num_features - num_pairs

        # collect all line_ids
        raw_data = data["raw_data"]
        line_ids = list(raw_data["template_features"].keys())

        # subsample lines
        perm = torch.randperm(len(line_ids))

        pair_idx = perm[:num_pairs]

        singlesA_idx = perm[num_pairs : num_pairs + num_singles]
        singlesB_idx = perm[num_pairs + num_singles : num_pairs + num_singles * 2]

        template_idx = torch.cat([pair_idx, singlesA_idx])
        warpedgt_idx = torch.cat([pair_idx, singlesB_idx])

        # random shuffle the indicies
        template_idx = template_idx[torch.randperm(template_idx.shape[0])]
        warpedgt_idx = warpedgt_idx[torch.randperm(warpedgt_idx.shape[0])]

        # actually subsample the features
        template_features = raw_data["template_features"]
        warpedgt_features = raw_data["warpedgt_features"]

        template_features = {
            line_ids[i]: template_features[line_ids[i]] for i in template_idx
        }
        warpedgt_features = {
            line_ids[i]: warpedgt_features[line_ids[i]] for i in warpedgt_idx
        }

        data["raw_data"]["template_features"] = template_features
        data["raw_data"]["warpedgt_features"] = warpedgt_features

        return data


class ToLightGlueFormat(object):

    def __call__(self, data: Dict) -> Dict:
        template_features = data["raw_data"]["template_features"]
        warpedgt_features = data["raw_data"]["warpedgt_features"]

        line_ids0 = list(template_features.keys())
        line_ids1 = list(warpedgt_features.keys())

        descriptors0 = torch.stack(
            list([v["features"] for v in template_features.values()])
        )
        descriptors1 = torch.stack(
            list([v["features"] for v in warpedgt_features.values()])
        )

        texts0 = list([v["text"] for v in template_features.values()])
        texts1 = list([v["text"] for v in warpedgt_features.values()])

        descriptors0_ids = list(template_features.keys())
        descriptors1_ids = list(warpedgt_features.keys())

        # calculate ground truth matches
        # note: gt_matches0 and gt_mataches1 (from the original implementation):
        # if -1: unmatched point
        # if -2: ignore point

        def get_index(line_id: str, line_ids: List[str]):
            try:
                return line_ids.index(line_id)
            except ValueError:
                return -1

        gt_matches0 = map(
            partial(get_index, line_ids=descriptors1_ids), descriptors0_ids
        )
        gt_matches1 = map(
            partial(get_index, line_ids=descriptors0_ids), descriptors1_ids
        )

        gt_matches0 = torch.tensor(list(gt_matches0))
        gt_matches1 = torch.tensor(list(gt_matches1))

        # create gt_assignment matrix ( true -> match, false -> no match)
        num_features = gt_matches1.shape[0]
        gt_matches_copy = gt_matches0.clone()
        gt_matches_copy[gt_matches_copy < 0] = num_features
        gt_assignment = one_hot(gt_matches_copy, num_classes=num_features + 1)
        gt_assignment = gt_assignment[:, :num_features].type(torch.bool)

        data["line_ids0"] = line_ids0
        data["line_ids1"] = line_ids1
        data["descriptors0"] = descriptors0
        data["descriptors1"] = descriptors1
        data["texts0"] = texts0
        data["texts1"] = texts1
        data["gt_matches0"] = gt_matches0
        data["gt_matches1"] = gt_matches1
        data["gt_assignment"] = gt_assignment
        data.pop("raw_data")

        return data


class RandomLineCrop(object):
    def __init__(self) -> None:
        self.mean = 0
        self.std = 0.25

        self.min_crop = 0
        self.max_crop = 0.75

    def __call__(self, data: Dict[str, Any]) -> torch.Tensor:
        data["descriptors0"], data["texts0"] = self._random_crop_linefeatures(
            data["descriptors0"], data["texts0"]
        )
        data["descriptors1"], data["texts1"] = self._random_crop_linefeatures(
            data["descriptors1"], data["texts1"]
        )
        return data

    def _random_crop_linefeatures(
        self, features: torch.Tensor, texts: List[Optional[str]]
    ) -> Tuple[torch.Tensor, List[Optional[str]]]:

        # features: (n, c, h, w)
        num_features, channels, line_width, line_length = features.shape

        remove_length = torch.zeros(num_features)
        trunc_normal_(
            remove_length,
            mean=self.mean,
            std=self.std,
            a=self.min_crop,
            b=self.max_crop,
        )

        remain_length = 1 - remove_length
        crop_start = torch.rand(num_features) * remove_length
        crop_end = crop_start + remain_length

        crop_start_px = (crop_start * line_length).type(torch.int)
        crop_end_px = (crop_end * line_length).type(torch.int)

        new_line_features = []
        new_texts = []

        for i in range(num_features):
            line_features = features[i : i + 1]
            text = texts[i]

            # crop text accordingly to the crop ratio
            if text is not None:
                text_crop_start = int(crop_start[i] * len(text))
                text_crop_end = int(crop_end[i] * len(text))
                new_texts.append(text[text_crop_start:text_crop_end])
            else:
                new_texts.append(None)

            # crop line features
            line_crop_start = crop_start_px[i]
            line_crop_end = crop_end_px[i]
            line_features = line_features[..., line_crop_start:line_crop_end]

            # interpolate line features to match original length
            line_features = interpolate(
                line_features, size=(line_width, line_length), mode="bilinear"
            )

            new_line_features.append(line_features)

        return torch.cat(new_line_features, dim=0), new_texts


class RandomLineReverse(object):
    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["descriptors0"] = self._random_reverse_linefeatures(data["descriptors0"])
        data["descriptors1"] = self._random_reverse_linefeatures(data["descriptors1"])

        return data

    def _random_reverse_linefeatures(self, features: torch.Tensor) -> torch.Tensor:
        num_features, channels, line_width, line_length = features.shape

        reverse = torch.rand(num_features) > 0.5
        reverse = rearrange(reverse, "b -> b () () ()")

        reversed_features = features.flip(3)

        return torch.where(reverse, reversed_features, features)


class RandomTextMutation(object):

    def __init__(
        self,
        prob_insert: float = 0.05,
        prob_delete: float = 0.05,
        prob_substitute: float = 0.05,
        alphabet: str = string.printable,
    ) -> None:
        self.alphabet = alphabet
        self.prob_insert = prob_insert
        self.prob_delete = prob_delete
        self.prob_substitute = prob_substitute

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["texts0"] = self._random_mutate_text(data["texts0"])
        data["texts1"] = self._random_mutate_text(data["texts1"])

        return data

    def _random_mutate_text(self, texts: List[Optional[str]]) -> List[Optional[str]]:
        new_texts = []

        for text in texts:
            if text is not None:
                text = "".join(map(self._random_insert_char, text + " "))[:-1]
                text = "".join(map(self._random_delete_char, text))
                text = "".join(map(self._random_substitue_char, text))
            new_texts.append(text)

        return new_texts

    def _random_insert_char(self, character: str) -> str:
        if float(torch.rand((1,))) > self.prob_insert:
            return character

        new_char_idx = int(torch.randint(len(self.alphabet), (1,)))
        new_char = self.alphabet[new_char_idx]

        return new_char + character

    def _random_delete_char(self, character: str) -> str:
        if float(torch.rand((1,))) > self.prob_delete:
            return character

        return ""

    def _random_substitue_char(self, character: str) -> str:
        if float(torch.rand((1,))) > self.prob_insert:
            return character

        new_char_idx = int(torch.randint(len(self.alphabet), (1,)))
        new_char = self.alphabet[new_char_idx]

        return new_char


class EmbeddTexts(object):
    def __init__(
        self, alphabet: str = string.printable, max_text_length: int = 32
    ) -> None:
        self.alphabet = alphabet
        self.max_text_length = max_text_length

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        data["texts_str0"] = data["texts0"]
        data["texts_str1"] = data["texts1"]
        data["texts0"] = self._embedd_texts(data["texts0"])
        data["texts1"] = self._embedd_texts(data["texts1"])

        return data

    def _embedd_texts(self, texts: List[Optional[str]]) -> torch.Tensor:
        embeddings = []

        for text in texts:
            embedding = torch.zeros(
                self.max_text_length, len(self.alphabet), dtype=torch.float32
            )

            if text is not None:
                text = text[: self.max_text_length]
                for i, c in enumerate(text):
                    try:
                        index = self.alphabet.index(c)
                        embedding[i][index] = 1
                    except ValueError:
                        pass  # ignore unknown characters

            embeddings.append(embedding)

        return torch.stack(embeddings)
