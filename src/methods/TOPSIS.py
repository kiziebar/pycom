class Method:

    def __init__(self, criteria: list, weight: list, alternatives: list, normalize_type: str = 'min_max'):
        """
        :param criteria: List of criteria, where 1 is benefit (profit or max), 0 is cost (min)
        :param weight: List of weights
        :param alternatives: List of lists alternatives
        :param normalize_type: Str type normalization method (sum, min_max, max, sqrt or log)
        """

        self.criteria = np.array(criteria)
        self.weight = np.array(weight)
        self.alternatives = np.array(alternatives, dtype = float)
        self.normalize_type = normalize_type

    def _normalize(self):
        if self.normalize_type == 'sum':
            self.alternatives = sum_norm(self.alternatives, self.criteria)
        elif self.normalize_type == 'min_max':
            self.alternatives = min_max(self.alternatives, self.criteria)
        elif self.normalize_type == 'max':
            self.alternatives = maximum(self.alternatives, self.criteria)
        elif self.normalize_type == 'sqrt':
            self.alternatives = square_sum(self.alternatives, self.criteria)
        elif self.normalize_type == 'log':
            self.alternatives = log_norm(self.alternatives, self.criteria)
        else:
            raise TypeError('Bad normalization type: %s', self.normalize_type)

    @staticmethod
    def rank(preference):
        return (len(preference) + 1) - rankdata(preference)


class Topsis(Method):

    def __init__(self, criteria: list, weight: list, alternatives: list, normalize_type: str = 'min_max',
                 distance='euclidean'):
        super(Topsis, self).__init__(criteria, weight, alternatives, normalize_type)
        self.distance = distance

    def _neg_pos(self):
        pis = np.zeros(self.alternatives.shape[1])
        nis = np.zeros(self.alternatives.shape[1])

        for j, crit in enumerate(self.criteria):
            if crit == 'cost':
                pis[j] = np.min(self.alternatives[:, j])
                nis[j] = np.max(self.alternatives[:, j])
            else:
                pis[j] = np.max(self.alternatives[:, j])
                nis[j] = np.min(self.alternatives[:, j])
        return pis, nis

    def fit(self):
        self._normalize()
        self.alternatives = self.alternatives * self.weight
        pis, nis = self._neg_pos()
        dp = np.zeros(self.alternatives.shape[0])
        dn = np.zeros(self.alternatives.shape[0])
        for i in range(self.alternatives.shape[0]):
            if self.distance == 'euclidean':
                dn[i] = np.sqrt(np.sum((self.alternatives[i, :] - nis) ** 2))
                dp[i] = np.sqrt(np.sum((self.alternatives[i, :] - pis) ** 2))
            if self.distance == 'manhattan':
                dn[i] = np.abs(np.sum((self.alternatives[i, :] - nis)))
                dp[i] = np.abs(np.sum((self.alternatives[i, :] - pis)))
            if self.distance == 'czybyszew':
                dn[i] = np.abs(np.max(self.alternatives[i, :] - nis))
                dp[i] = np.abs(np.max(self.alternatives[i, :] - pis))
            if self.distance == 'cosinus':
                dn[i] = self.__cos(self.alternatives[i, :], nis)
                dp[i] = self.__cos(self.alternatives[i, :], pis)
            if self.distance == 'corr':
                dn[i] = self.__corr(self.alternatives[i, :], nis)
                dp[i] = self.__corr(self.alternatives[i, :], pis)
        return dn / (dn + dp)