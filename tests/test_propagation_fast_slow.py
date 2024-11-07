import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from models.concept_bottleneck_model import PropagationCBM as up_cbm
from utils import generate_all_binary_vectors


def test_c_probability():
    c = [0, 1, 0]
    c_pred_probas = np.array([[0.5, 0.1, 0.2],
                              [0.1, 0.5, 0.9]])
    expected_val = np.array([0.04, 0.045])
    assert np.allclose(up_cbm.c_probability(c, c_pred_probas), expected_val)


test_c_probability()

n_train_samples = 100
n_concepts = 3
multiclass = False
# create binary numpy array of size (n_train_samples, n_concepts)
concepts_train = np.random.randint(2, size=(n_train_samples, n_concepts))
y_train = np.random.randint(2, size=(n_train_samples,))

y_model = LogisticRegression().fit(concepts_train, y_train)

concept_probas = np.array([[0.55582076, 0.34586252, 0.66859962],
                           [0.73845586, 0.79936703, 0.88832295],
                           [0.45831525, 0.16680123, 0.60681941],
                           [0.02593659, 0.03799343, 0.51283525],
                           [0.22280347, 0.48819178, 0.73270701],
                           [0.12621145, 0.05686961, 0.4807942],
                           [0.62129846, 0.85428359, 0.30566356],
                           [0.3458411, 0.19549518, 0.37792467],
                           [0.5898568, 0.78177266, 0.37864355],
                           [0.36676077, 0.06756824, 0.829159]])

all_cs = generate_all_binary_vectors(n_concepts)
pred_probas = y_model.predict_proba(all_cs)[:, 1]
all_cs_probas = {tuple(c): proba for c, proba in zip(all_cs, pred_probas)}

slow_res = up_cbm.propagation(concept_probas, all_cs, all_cs_probas, multiclass=multiclass)
fast_res = up_cbm.propagation_monte_carlo(concept_probas, y_model, n_mc=10000)

print('original: ', slow_res)
print('monte carlo: ', fast_res)

# print absoluate differences between slow and fast
print('difference: ', np.abs(slow_res - fast_res))


# print bootstrap confidence intervals

def bootstrap_analysis(concept_proba, ym, n_mc=10000, n_bootstrap=1000):
    concept_proba = concept_proba.reshape(1, -1)
    original_samples = [up_cbm.propagation_monte_carlo(concept_proba, ym, n_mc=n_mc) for _ in
                        range(100)]

    bootstrap_samples = []
    for _ in range(n_bootstrap):
        resample_indices = np.random.choice(len(original_samples), len(original_samples), replace=True)
        resampled_estimates = np.array(original_samples)[resample_indices]
        bootstrap_sample_mean = np.mean(resampled_estimates)
        bootstrap_samples.append(bootstrap_sample_mean)

    # Compute statistics from bootstrap samples
    bootstrap_mean = np.mean(bootstrap_samples)
    bootstrap_std = np.std(bootstrap_samples)
    bootstrap_ci = np.percentile(bootstrap_samples, [2.5, 97.5])

    # Calculate standard error and bias
    standard_error = bootstrap_std / np.sqrt(n_bootstrap)
    bias = bootstrap_mean - np.mean(original_samples)

    print(f"Bootstrap Mean: {bootstrap_mean}")
    print(f"Bootstrap Standard Deviation: {bootstrap_std}")
    print(f"95% Confidence Interval: {bootstrap_ci}")
    print(f"Standard Error: {standard_error}")
    print(f"Bias: {bias}")
    return bootstrap_mean, bootstrap_std, bootstrap_ci, standard_error, bias


def plot_convergence(cps, ym, min_n=10, max_n=10000, step=100):
    for i in range(cps.shape[0]):
        N_values = []
        estimates = []
        current_concept_probas = cps[i, :].reshape(1, -1)

        for n in range(min_n, max_n + 1, step):
            estimate = up_cbm.propagation_monte_carlo(current_concept_probas, y_model, n_mc=n)
            N_values.append(n)
            estimates.append(estimate)

        plt.plot(N_values, estimates, marker='o', linestyle='-', label=f'Point {i + 1}')

    plt.xlabel('Number of Samples (N)')
    plt.ylabel('Monte Carlo Estimate')
    plt.title('Convergence of Monte Carlo Estimates for Multiple Points')
    plt.legend()
    plt.grid(True)
    plt.show()

std_devs = []
for i in range(concept_probas.shape[0]):
    std_dev = bootstrap_analysis(concept_probas[i, :], y_model, n_mc=10000, n_bootstrap=1000)[1]
    std_devs.append(std_dev)
print('Average Bootstrap Standard Deviation: ', np.mean(std_devs))
bootstrap_analysis(concept_probas[0], y_model, n_mc=10000, n_bootstrap=1000)

plot_convergence(concept_probas[:5, :], y_model)
