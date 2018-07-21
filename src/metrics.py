import torch
import pdb


class Metrics:
    def __init__(self):
        self.name = 'Metric Name'

    def reset(self):
        pass

    def update(self, predicts, batch):
        pass

    def get_score(self):
        pass


class Accuracy(Metrics):
    def __init__(self, ats=[1, 5, 10, 50]):
        self.ats = ats
        self.n = 0
        self.n_correct_ats = [0] * len(self.ats)
        self.name = 'Accuracy@' + ', '.join(map(str, ats))
        self.noise = 1e-6

    def reset(self):
        self.n = 0
        self.n_correct_ats = [0] * len(self.ats)

    def update(self, predicts, batch):
        """
        Args:
            predicts (FloatTensor): with size (batch, n_samples).
            batch (dict): batch.
        """
        # add noise to deal with cases where all predict score are identical
        predicts += torch.rand_like(predicts) * self.noise
        self.n += predicts.size(0)
        sorted_predicts = torch.sort(predicts)[0]
        for i, at in enumerate(self.ats):
            if predicts.size(1) <= at:
                break

            score_at = sorted_predicts[:, -at]

            # assume that the 0th option is answer
            self.n_correct_ats[i] += (predicts[:, 0] >= score_at).sum().item()

    def get_score(self):
        return [n_correct / self.n for n_correct in self.n_correct_ats]
