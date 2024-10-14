import pytest
import numpy as np

from lambeq import BinaryCrossEntropyLoss, CrossEntropyLoss, MSELoss


bce_w_jax = BinaryCrossEntropyLoss(use_jax=True)
bce_wo_jax = BinaryCrossEntropyLoss(use_jax=False)
bce_w_jax_sparse = BinaryCrossEntropyLoss(sparse=True, use_jax=True)
bce_wo_jax_sparse = BinaryCrossEntropyLoss(sparse=True, use_jax=False)
ce_wo_jax = CrossEntropyLoss(use_jax=False)
ce_w_jax = CrossEntropyLoss(use_jax=True)
mse = MSELoss()

@pytest.mark.parametrize('loss_fn, y_hat, y, expected_output, expected_error, use_jax', [
    (bce_wo_jax, [[0.1, 0.9], [0.8, 0.2]], [[0, 1], [1, 0]], 0.16425203230546248, None, False),
    (bce_wo_jax, [[0.0, 1.0], [0.8, 0.2]], [[0, 1], [1, 0]], 0.11157177503210487, None, False),
    (bce_wo_jax, [[0.0, 1.0], [1.0, 0.0]], [[0, 1], [1, 0]], 0.0, None, False),
    (bce_wo_jax, [[0.0, 1.0], ], [[1, 0], [0, 1]], None, ValueError, False),
    (bce_wo_jax_sparse, [0.9, 0.2], [1, 0], 0.16425203230546248, None, False),
    (bce_wo_jax_sparse, [1.0, 0.2], [1, 0], 0.11157177503210487, None, False),
    (bce_wo_jax_sparse, [1.0, 0.0], [1, 0], 0.0, None, False),
    (bce_wo_jax_sparse, [1.0, ], [0, 1], None, ValueError, False),
    (bce_w_jax, [[0.1, 0.9], [0.8, 0.2]], [[0, 1], [1, 0]], 0.16425203230546248, None, True),
    (bce_w_jax, [[0.0, 1.0], [0.8, 0.2]], [[0, 1], [1, 0]], 0.11157177503210487, None, True),
    (bce_w_jax, [[0.0, 1.0], [1.0, 0.0]], [[0, 1], [1, 0]], 0.0, None, True),
    (bce_w_jax, [[0.0, 1.0], ], [[1, 0], [0, 1]], None, ValueError, True),
    (bce_w_jax_sparse, [0.9, 0.2], [1, 0], 0.16425203230546248, None, True),
    (bce_w_jax_sparse, [1.0, 0.2], [1, 0], 0.11157177503210487, None, True),
    (bce_w_jax_sparse, [1.0, 0.0], [1, 0], 0.0, None, True),
    (bce_w_jax_sparse, [1.0, ], [0, 1], None, ValueError, True),
    (mse, [0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], 0.01, None, False),
    (mse, [[0.1, 0.2], [0.3, 0.4]], [[0.2, 0.3], [0.4, 0.5]], 0.01, None, False),
    (mse, [0.0, 0.0], [1.0, 1.0], 1.0, None, False),
    (mse, [0.0, 1.0], [0.0, 1.0], 0.0, None, False),
    (mse, [0.0, 0.0], [1.0, ], None, ValueError, False),
    (mse, [0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5], 0.01, None, True),
    (mse, [[0.1, 0.2], [0.3, 0.4]], [[0.2, 0.3], [0.4, 0.5]], 0.01, None, True),
    (mse, [0.0, 0.0], [1.0, 1.0], 1.0, None, True),
    (mse, [0.0, 1.0], [0.0, 1.0], 0.0, None, True),
    (mse, [0.0, 0.0], [1.0, ], None, ValueError, True),
    (ce_wo_jax, [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], [[0, 0, 1], [0, 1, 0]], 0.7803238741323342, None, False),
    (ce_wo_jax, [[0.1, 0.2, 0.7], [0.6, 0.4, 0]], [[0, 0, 1], [0, 1, 0]], 0.6364828379064438, None, False),
    (ce_wo_jax, [[0, 0, 0, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [1, 0, 0, 0]], 0, None, False),
    (ce_w_jax, [[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]], [[0, 0, 1], [0, 1, 0]], 0.7803238741323342, None, True),
    (ce_w_jax, [[0.1, 0.2, 0.7], [0.6, 0.4, 0]], [[0, 0, 1], [0, 1, 0]], 0.6364828379064438, None, True),
    (ce_w_jax, [[0, 0, 0, 1], [1, 0, 0, 0]], [[0, 0, 0, 1], [1, 0, 0, 0]], 0, None, True),
])
def test_loss_functions(loss_fn, y_hat, y, expected_output, expected_error, use_jax):

    if use_jax:
        from jax import numpy as jnp

        y_p = jnp.array(y_hat)
        y_t = jnp.array(y)
    else:
        y_p = np.array(y_hat)
        y_t = np.array(y)

    if expected_error:
        with pytest.raises(expected_error):
            loss_fn(y_p, y_t)
    else:
        assert loss_fn(y_p, y_t) == pytest.approx(expected_output, abs=1e-7)
