import itertools

import numpy as np
from typing import List, Optional

from numpy import ndarray


def intervene_on_concept(c_pred: ndarray,
                         c_true: ndarray,
                         intervention_mask: List[bool]) -> ndarray:
    c_pred_intervened = c_pred.copy()
    # mask = np.random.choice([True, False], n, p=[intervention_mask[c_i], 1 - intervention_mask[c_i]])
    c_pred_intervened[intervention_mask] = c_true[intervention_mask]
    assert not np.isnan(c_pred_intervened).any(), "c_pred_intervened contains NaN values."
    return c_pred_intervened


def intervene(C_pred: ndarray,
              C_true: ndarray,
              intervention_masks: ndarray) -> ndarray:
    assert C_pred.shape == C_true.shape
    C_pred_intervened = np.empty(shape=C_pred.shape)
    for c_i in range(C_pred.shape[1]):
        C_pred_intervened[:, c_i] = intervene_on_concept(C_pred[:, c_i], C_true[:, c_i],
                                                         intervention_masks[:, c_i])
    assert not np.isnan(C_pred_intervened).any(), "C_pred_intervened contains NaN values."
    return C_pred_intervened


class InterventionStrategy(object):
    def __init__(self, costs: List[float], f, h, **kwargs):
        self.name = ''
        # list of length n_concepts representing costs associated with intervening at each concept
        self.costs = costs
        self.f = f
        self.h = h
        self.seed = kwargs.get('seed', 1)
        np.random.seed(self.seed)

    def fit(self, concept_probas: ndarray, budget: float):
        """
        Fit the intervention strategy to the data
        :param concept_probas:
        :param budget:
        :return:
        """
        raise NotImplementedError

    def select_interventions(self, concept_probas: ndarray, budget=np.infty, abs_inds: Optional[ndarray] = None,
                             exhaust_intervention_budget: bool = False):
        raise NotImplementedError


class NoIntervention(InterventionStrategy):
    """
    No intervention
    """

    def __init__(self, costs: List[float], f, h, **kwargs):
        super().__init__(costs, f, h, **kwargs)
        self.name = 'no intervention'

    def fit(self, concept_probas: ndarray, budget: float):
        pass

    def select_interventions(self, concept_probas: ndarray, budget=np.infty, abs_inds: Optional[ndarray] = None,
                             exhaust_intervention_budget: bool = False):
        return np.zeros_like(concept_probas, dtype=bool)


class InterveneAllAbstentions(InterventionStrategy):
    def __init__(self, costs: List[float], f, h, **kwargs):
        super().__init__(costs, f, h, **kwargs)
        self.name = 'all abstentions intervention'

    def fit(self, concept_probas: ndarray, budget: float):
        """
        Set the fraction of samples to intervene on
        :param concept_probas:
        :param budget:
        :return:
        """
        pass

    def select_interventions(self, concept_probas: ndarray, budget=np.infty, abs_inds: Optional[ndarray] = None,
                             exhaust_intervention_budget: bool = False):
        y_proba = self.f.concept_probas_to_y_proba(concept_probas)
        abs_mask = self.h.get_abstension_mask(y_proba)
        return np.tile(abs_mask, (concept_probas.shape[1], 1)).T


class RandomAgnosticIntervention(InterventionStrategy):
    """
    Randomly intervene regardless of abstention
    """

    def __init__(self, costs: List[float], f, h, p_intervene=0.0, **kwargs):
        super().__init__(costs, f, h, **kwargs)
        self.name = f'random agnostic intervention p{p_intervene}'
        self.p_intervene = p_intervene

    def fit(self, concept_probas: ndarray, budget: float):
        """
        Set the fraction of samples to intervene on
        :param concept_probas:
        :param budget:
        :return:
        """
        pass

    def select_interventions(self, concept_probas: ndarray, budget=np.infty, abs_inds: Optional[ndarray] = None,
                             exhaust_intervention_budget: bool = False):
        # ignore budget
        n_samples, n_concepts = concept_probas.shape
        random_values = np.random.rand(n_samples, n_concepts)
        interventions = random_values < self.p_intervene
        return interventions


class RandomIntervention(InterventionStrategy):
    """
    Randomly intervene on a fraction of samples with the maximum probability possible given an intervention budget
    """

    def __init__(self, costs: List[float], f, h, **kwargs):
        super().__init__(costs, f, h, **kwargs)
        self.name = 'random intervention'
        self.p_intervene = 0.0

    def fit(self, concept_probas: ndarray, budget: float):
        """
        Set the fraction of samples to intervene on
        :param concept_probas:
        :param budget:
        :return:
        """
        pass
        # ignore p_intervene for now
        # n_samples, n_concepts = concept_probas.shape
        # n_abstentions = np.sum(np.isnan(self.h.predict_proba(self.f.concept_probas_to_y_proba(concept_probas))))
        # if n_abstentions == 0:
        #     self.p_intervene = 0.0
        # else:
        #     self.p_intervene = min(1.0, (budget / (n_abstentions * np.mean(self.costs) * n_concepts)))

    def select_interventions(self, concept_probas: ndarray, budget=np.infty, abs_inds: Optional[ndarray] = None,
                             exhaust_intervention_budget: bool = False):
        np.random.seed(self.seed)
        if budget == np.infty:
            return np.ones_like(concept_probas, dtype=bool)
        if budget == 0:
            return np.zeros_like(concept_probas, dtype=bool)

        n_samples, n_concepts = concept_probas.shape
        random_values = np.random.rand(n_samples, n_concepts)
        masks = np.zeros_like(concept_probas, dtype=bool)

        if abs_inds is None:
            y_proba = self.f.concept_probas_to_y_proba(concept_probas)
            abs_inds = np.nonzero(self.h.get_abstension_mask(y_proba))[0]

        # First, prioritize the abstention indices
        b = 0
        if abs_inds.size > 0:
            restricted_random_values = random_values[abs_inds, :]
            sorted_indices = np.unravel_index(np.argsort(-restricted_random_values, axis=None),
                                              restricted_random_values.shape)

            for i, k in zip(*sorted_indices):
                real_i = abs_inds[i]
                if b + self.costs[k] <= budget:
                    masks[real_i, k] = True
                    b += self.costs[k]
                    if b >= budget:
                        return masks

        # Then, try to exhaust the budget for non-abstention indices
        if exhaust_intervention_budget:
            non_abs_inds = np.setdiff1d(np.arange(n_samples), abs_inds)
            remaining_random_values = random_values[non_abs_inds, :]
            remaining_sorted_indices = np.unravel_index(np.argsort(-remaining_random_values, axis=None),
                                                        remaining_random_values.shape)

            for i, k in zip(*remaining_sorted_indices):
                real_i = non_abs_inds[i]
                if b + self.costs[k] <= budget:
                    masks[real_i, k] = True
                    b += self.costs[k]
                    if b >= budget:
                        break

        return masks


class VarianceIntervention(InterventionStrategy):
    """
    Intervene concepts greedily according to the expected increase in confidence
    """

    def __init__(self, costs, f, h, **kwargs):
        super().__init__(costs, f, h, **kwargs)
        self.name = 'greedy variance intervention'
        self.score_threshold = np.infty
        self.scores = {}

    def get_score(self, concept_proba: ndarray, k: int):
        # Memoization
        rounded_concept_proba = np.round(concept_proba, 5)
        key = (tuple(rounded_concept_proba), k)
        if key in self.scores:
            return self.scores[key]

        # score components
        p_k = concept_proba[k]
        one_minus_p_k = 1.0 - p_k
        # c_int if the result of intervention is c_k=0 or c_k=1
        cint_0 = rounded_concept_proba.copy()
        cint_1 = rounded_concept_proba.copy()
        cint_0[k] = 0
        cint_1[k] = 1
        # pred_proba if c_k=0 or c_k=1
        cint_batch = np.vstack([cint_0, cint_1])
        f_cint_batch = self.f.concept_probas_to_y_proba(cint_batch)
        f_cint_0, f_cint_1 = f_cint_batch[0], f_cint_batch[1]
        # expected value of r_k (outcome of intervention)
        E_r_k = np.dot([p_k, one_minus_p_k], [f_cint_1, f_cint_0])
        E_r_k_squared = np.dot([p_k, one_minus_p_k], [f_cint_1 ** 2, f_cint_0 ** 2])
        # variance of r_k
        score = np.mean(E_r_k_squared - (E_r_k ** 2))
        self.scores[key] = score
        return score

    def fit(self, concept_probas: ndarray, budget: float):
        """
        Fit the intervention strategy to the data
        :param concept_probas:
        :param budget:
        :return:
        """
        pass
        # ignore score_threshold for now
        # b, S = 0, []
        # n_samples, n_concepts = concept_probas.shape
        # # only iterate through samples that are yield abstentions
        # abs_inds = np.where(np.isnan(self.h.predict_proba(self.f.concept_probas_to_y_proba(concept_probas))))[
        #     0].tolist()
        # for i in abs_inds:
        #     for k in range(n_concepts):
        #         # increase in certainty from intervention on concept k
        #         cp = concept_probas[i, :].copy()
        #         score = self.get_score(cp, k)
        #         b += self.costs[k]
        #         heapq.heappush(S, (score, i, k))
        #         # remove lowest scoring interventions until budget is satisfied
        #         while b > budget:
        #             _, i_min, k_min = heapq.heappop(S)
        #             b -= self.costs[k_min]
        # # score threshold is the lowest score in the heap
        # if S:
        #     s_thresh, _, _ = heapq.heappop(S)
        #     self.score_threshold = s_thresh

    def select_interventions(self, concept_probas: ndarray, budget=np.infty, abs_inds: Optional[ndarray] = None,
                             exhaust_intervention_budget: bool = False):
        if budget == np.infty:
            return np.ones_like(concept_probas, dtype=bool)
        if budget == 0:
            return np.zeros_like(concept_probas, dtype=bool)

        n_samples, n_concepts = concept_probas.shape
        masks = np.zeros_like(concept_probas, dtype=bool)

        if abs_inds is None:
            y_proba = self.f.concept_probas_to_y_proba(concept_probas)
            abs_inds = np.nonzero(self.h.get_abstension_mask(y_proba))[0]

        # First, prioritize the abstention indices
        b = 0
        all_scores = np.random.rand(n_samples, n_concepts)
        if abs_inds.size > 0:
            for i, k in itertools.product(abs_inds, range(n_concepts)):
                all_scores[i, k] = self.get_score(concept_probas[i, :].copy(), k)
            abs_scores = all_scores[abs_inds, :]
            sorted_indices_abs = np.unravel_index(np.argsort(-abs_scores, axis=None),
                                                  abs_scores.shape)

            for i, k in zip(*sorted_indices_abs):
                if b + self.costs[k] <= budget:
                    real_i = abs_inds[i]
                    masks[real_i, k] = True
                    b += self.costs[k]
                    if b >= budget:
                        return masks

        # Then, try to exhaust the budget for non-abstention indices
        if exhaust_intervention_budget:
            non_abs_inds = np.setdiff1d(np.arange(n_samples), abs_inds)
            for i, k in itertools.product(non_abs_inds, range(n_concepts)):
                all_scores[i, k] = self.get_score(concept_probas[i, :].copy(), k)
            remaining_scores = all_scores[non_abs_inds, :]
            remaining_sorted_score_indices = np.unravel_index(np.argsort(-remaining_scores, axis=None),
                                                              remaining_scores.shape)

            for i, k in zip(*remaining_sorted_score_indices):
                if b + self.costs[k] <= budget:
                    real_i = non_abs_inds[i]
                    masks[real_i, k] = True
                    b += self.costs[k]
                    if b >= budget:
                        break

        return masks

    # def select_interventions(self, concept_probas: ndarray, budget=np.infty, abs_inds: Optional[ndarray] = None,
    #                          exhaust_intervention_budget: bool = False):
    #     n_samples, n_concepts = concept_probas.shape
    #     masks = np.zeros_like(concept_probas, dtype=bool)
    #
    #     if budget == np.infty:
    #         return np.ones_like(concept_probas, dtype=bool)
    #     if budget == 0:
    #         return np.zeros_like(concept_probas, dtype=bool)
    #
    #     if abs_inds is None:
    #         y_proba = self.f.concept_probas_to_y_proba(concept_probas)
    #         abs_inds = np.nonzero(self.h.get_abstension_mask(y_proba))[0]
    #
    #     start_time = time.time()
    #     b = 0
    #     # First loop through abstention indices
    #     all_abs_scores = np.zeros_like(concept_probas, dtype=float)
    #     for i, k in itertools.product(abs_inds, range(n_concepts)):
    #         all_abs_scores[i, k] = self.get_score(concept_probas[i, :].copy(), k)
    #     # print('score computation took {:.2f} seconds'.format(time.time() - start_time))
    #     sorted_abs_indices = np.unravel_index(np.argsort(-all_abs_scores, axis=None), all_abs_scores.shape)
    #     for i, k in zip(*sorted_abs_indices):
    #         if i in abs_inds and b + self.costs[k] <= budget:
    #             masks[i, k] = True
    #             b += self.costs[k]
    #             if b >= budget:
    #                 return masks
    #
    #     # Second loop through non-abstention indices to exhaust the remaining budget
    #     if exhaust_intervention_budget and b < budget:
    #         non_abs_inds = np.setdiff1d(np.arange(n_samples), abs_inds)
    #         all_non_abs_scores = np.zeros_like(concept_probas, dtype=float)
    #         for i, k in itertools.product(non_abs_inds, range(n_concepts)):
    #             all_non_abs_scores[i, k] = self.get_score(concept_probas[i, :].copy(), k)
    #         # print('score computation took {:.2f} seconds'.format(time.time() - start_time))
    #         sorted_non_abs_indices = np.unravel_index(np.argsort(-all_non_abs_scores, axis=None),
    #                                                   all_non_abs_scores.shape)
    #         for i, k in zip(*sorted_non_abs_indices):
    #             if i in non_abs_inds and b + self.costs[k] <= budget:
    #                 masks[i, k] = True
    #                 b += self.costs[k]
    #                 if b >= budget:
    #                     return masks
    #
    #     return masks

# class EnumerationIntervention(VarianceIntervention):
