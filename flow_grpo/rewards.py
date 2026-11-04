"""Rewards logged during VATDM training and evaluation."""
from PIL import Image
import torch


def pickscore_score(device):
    from flow_grpo.pickscore_scorer import PickScoreScorer

    scorer = PickScoreScorer(dtype=torch.float32, device=device)

    def _fn(images, prompts, metadata):
        if isinstance(images, torch.Tensor):
            images = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images = images.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            images = [Image.fromarray(image) for image in images]
        scores = scorer(prompts, images)
        return scores, {}

    return _fn


def multi_score(device, score_dict):
    score_functions = {
        'pickscore': pickscore_score,
    }
    score_fns = {name: score_functions[name](device) for name in score_dict}

    def _fn(images, prompts, metadata):
        total_scores = []
        score_details = {}

        for score_name, weight in score_dict.items():
            scores, _ = score_fns[score_name](images, prompts, metadata)
            score_details[score_name] = scores
            weighted_scores = [weight * score for score in scores]

            if not total_scores:
                total_scores = weighted_scores
            else:
                total_scores = [total + weighted for total, weighted in zip(total_scores, weighted_scores)]

        score_details['avg'] = total_scores
        return score_details, {}

    return _fn
