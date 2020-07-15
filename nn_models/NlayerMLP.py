from sklearn.neural_network._multilayer_perceptron import BaseMultilayerPerceptron
from sklearn.neural_network._base import LOSS_FUNCTIONS, DERIVATIVES
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.base import RegressorMixin
from sklearn.utils import check_X_y, column_or_1d
from sklearn.utils.validation import check_is_fitted

import numpy as np
import pandas as pd
import copy

NEW_LOSS_FUNCTIONS = copy.copy(LOSS_FUNCTIONS)


''' Additional Loss function'''


def rooted_squared_loss(y_true, y_pred):
    return np.sqrt(((y_true - y_pred) ** 2).mean())


def mean_absolute_loss(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


NEW_LOSS_FUNCTIONS['rooted_squared_loss'] = rooted_squared_loss
NEW_LOSS_FUNCTIONS['mean_absolute_loss'] = mean_absolute_loss


'''Modified MLP model'''


class ModBaseMLP(BaseMultilayerPerceptron):
    """
    Modified BaseMultilayerPerceptron that support different loss function -- MSE|RMSE|MAE.

    LBFGS won't be supported
    """

    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, loss='squared_loss', shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, max_fun=15000):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, loss=loss, shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change, max_fun=max_fun)

        if 'squared_loss' != self.loss and 'rooted_squared_loss' != self.loss and 'mean_absolute_loss' != self.loss:
            raise NotImplementedError('Unsupported Loss specified: {}'.format(self.loss))

    def _backprop(self, X, y, activations, deltas, coef_grads,
                  intercept_grads):
        """Compute the MLP loss function and its corresponding derivatives
        with respect to each parameter: weights and bias vectors.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        y : ndarray of shape (n_samples,)
            The target values.
        activations : list, length = n_layers - 1
             The ith element of the list holds the values of the ith layer.
        deltas : list, length = n_layers - 1
            The ith element of the list holds the difference between the
            activations of the i + 1 layer and the backpropagated error.
            More specifically, deltas are gradients of loss with respect to z
            in each layer, where z = wx + b is the value of a particular layer
            before passing through the activation function
        coef_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            coefficient parameters of the ith layer in an iteration.
        intercept_grads : list, length = n_layers - 1
            The ith element contains the amount of change used to update the
            intercept parameters of the ith layer in an iteration.
        Returns
        -------
        loss : float
        coef_grads : list, length = n_layers - 1
        intercept_grads : list, length = n_layers - 1
        """
        n_samples = X.shape[0]

        # Forward propagate
        activations = self._forward_pass(activations)

        # Get loss
        loss_func_name = self.loss
        if loss_func_name == 'log_loss' and self.out_activation_ == 'logistic':
            loss_func_name = 'binary_log_loss'
        raw_loss = NEW_LOSS_FUNCTIONS[loss_func_name](y, activations[-1])

        # Add L2 regularization term to loss
        values = np.sum(
            np.array([np.dot(s.ravel(), s.ravel()) for s in self.coefs_]))
        loss = raw_loss + (0.5 * self.alpha) * values / n_samples

        # Backward propagate
        last = self.n_layers_ - 2

        # The calculation of delta[last] here works with following
        # combinations of output activation and loss function:
        # sigmoid and binary cross entropy, softmax and categorical cross
        # entropy, and identity with squared loss -- ONLY this one is supported
        if 'squared_loss' == self.loss:
            deltas[last] = activations[-1] - y
        elif 'rooted_squared_loss' == self.loss:
            if -1e-8 < raw_loss < 1e-8:
                # Safe divide
                deltas[last] = 0. * activations[-1]
            else:
                deltas[last] = (activations[-1] - y) / (2 * raw_loss)
        else:
            # Mean Absolute Error
            tmp_holder = activations[-1] - y
            pos = (tmp_holder > 0.).astype(activations[-1].dtype)
            neg = (tmp_holder < 0.).astype(activations[-1].dtype)
            deltas[last] = pos - neg

        # Compute gradient for the last layer
        coef_grads, intercept_grads = self._compute_loss_grad(
            last, n_samples, activations, deltas, coef_grads, intercept_grads)

        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 2, 0, -1):
            deltas[i - 1] = safe_sparse_dot(deltas[i], self.coefs_[i].T)
            inplace_derivative = DERIVATIVES[self.activation]
            inplace_derivative(activations[i], deltas[i - 1])

            coef_grads, intercept_grads = self._compute_loss_grad(
                i - 1, n_samples, activations, deltas, coef_grads,
                intercept_grads)

        return loss, coef_grads, intercept_grads


class ModMLPForecaster(RegressorMixin, ModBaseMLP):
    def __init__(self, hidden_layer_sizes=(100,), activation="relu",
                 solver='adam', alpha=0.0001,
                 batch_size='auto', learning_rate="constant",
                 learning_rate_init=0.001,
                 power_t=0.5, max_iter=200, shuffle=True,
                 random_state=None, tol=1e-4,
                 verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False,
                 validation_fraction=0.1, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, n_iter_no_change=10, loss='squared_loss', num_tgt=1):
        super().__init__(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation, solver=solver, alpha=alpha,
            batch_size=batch_size, learning_rate=learning_rate,
            learning_rate_init=learning_rate_init, power_t=power_t,
            max_iter=max_iter, loss=loss, shuffle=shuffle,
            random_state=random_state, tol=tol, verbose=verbose,
            warm_start=warm_start, momentum=momentum,
            nesterovs_momentum=nesterovs_momentum,
            early_stopping=early_stopping,
            validation_fraction=validation_fraction,
            beta_1=beta_1, beta_2=beta_2, epsilon=epsilon,
            n_iter_no_change=n_iter_no_change)
        self.history = None
        self.back_step, self.length_per_point = -1, -1
        self._index = None
        self.num_tgt = num_tgt

    def predict(self, X):
        """Predict using the multi-layer perceptron model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data.
        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self, "coefs_")
        y_pred = self._predict(X)
        if y_pred.shape[1] == 1:
            return y_pred.ravel()
        return y_pred

    def forecast(self, X, steps, real_value=False):
        """Forecast k-steps ahead using fiited MLP model.
        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input data. -- Not necessary

        steps : Integer, number of steps forecast ahead
        real_value : Boolean, flag of the execution mode, True for prediction using real value  (train split)

        Returns
        -------
        y : ndarray of shape (n_samples, n_outputs)
            The predicted values.
        """
        check_is_fitted(self, "coefs_")
        res = list()

        # Current input seems unfair in the real_value model
        current_input = self.history
        if not real_value:
            for s in range(steps):
                tmp_pred = self._predict(current_input.reshape(1, -1))
                res.append([tmp_pred.item(i) for i in range(self.num_tgt)])
                current_input = np.concatenate((tmp_pred, current_input), axis=None)[:-self.length_per_point]
        else:
            if isinstance(X, pd.DataFrame):
                X = X.astype(float).to_numpy()

            for s in range(X.shape[0]):
                tmp_pred = self._predict(current_input.reshape(1, -1))
                res.append([tmp_pred.item(i) for i in range(self.num_tgt)])
                current_input = np.concatenate((X[s], current_input), axis=None)[:-self.length_per_point]

        res = np.asarray(res)
        if res.shape[1] == 1:
            return res.ravel()
        return res

    def _validate_input(self, X, y, incremental):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True, y_numeric=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=False)
        return X, y

    def set_history(self, data, index):
        self.history = data.to_numpy()  # Ndarray
        self.back_step = self.history.shape[0]
        self.history = self.history.flatten()
        self.length_per_point = self.history.shape[0] // self.back_step
        self._index = index
