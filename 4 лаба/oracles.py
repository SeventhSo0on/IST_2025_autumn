import numpy as np
import scipy
import scipy.sparse
from scipy.special import expit
import warnings


class BaseSmoothOracle(object):
    """
    Base class for implementation of oracles.
    """
    def func(self, x):
        """
        Computes the value of function at point x.
        """
        raise NotImplementedError('Func oracle is not implemented.')

    def grad(self, x):
        """
        Computes the gradient at point x.
        """
        raise NotImplementedError('Grad oracle is not implemented.')
    
    def hess(self, x):
        """
        Computes the Hessian matrix at point x.
        """
        raise NotImplementedError('Hessian oracle is not implemented.')
    
    def func_directional(self, x, d, alpha):
        """
        Computes phi(alpha) = f(x + alpha*d).
        """
        return np.squeeze(self.func(x + alpha * d))

    def grad_directional(self, x, d, alpha):
        """
        Computes phi'(alpha) = (f(x + alpha*d))'_{alpha}
        """
        return np.squeeze(self.grad(x + alpha * d).dot(d))


class QuadraticOracle(BaseSmoothOracle):
    """
    Oracle for quadratic function:
       func(x) = 1/2 x^TAx - b^Tx.
    """

    def __init__(self, A, b):
        if not scipy.sparse.isspmatrix_dia(A) and not np.allclose(A, A.T):
            warnings.warn('A should be a symmetric matrix.')
        self.A = A
        self.b = b

    def func(self, x):
        return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)

    def grad(self, x):
        return self.A.dot(x) - self.b

    def hess(self, x):
        return self.A 


class CustomQuadraticOracle(QuadraticOracle):  # НАША МОДИФИКАЦИЯ
    """
    Extended quadratic oracle with additional linear term.
    func(x) = 1/2 x^TAx - b^Tx + c^Tx + d
    """
    def __init__(self, A, b, c=None, d=0):
        super().__init__(A, b)
        self.c = c if c is not None else np.zeros_like(b)
        self.d = d
        
    def func(self, x):
        return super().func(x) + self.c.dot(x) + self.d
        
    def grad(self, x):
        return super().grad(x) + self.c


class LogRegL2Oracle(BaseSmoothOracle):
    """
    Oracle for logistic regression with l2 regularization:
         func(x) = 1/m sum_i log(1 + exp(-b_i * a_i^T x)) + regcoef / 2 ||x||_2^2.

    Let A and b be parameters of the logistic regression (feature matrix
    and labels vector respectively).
    For user-friendly interface use create_log_reg_oracle()

    Parameters
    ----------
        matvec_Ax : function
            Computes matrix-vector product Ax, where x is a vector of size n.
        matvec_ATx : function of x
            Computes matrix-vector product A^Tx, where x is a vector of size m.
        matmat_ATsA : function
            Computes matrix-matrix-matrix product A^T * Diag(s) * A,
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        self.matvec_Ax = matvec_Ax
        self.matvec_ATx = matvec_ATx
        self.matmat_ATsA = matmat_ATsA
        self.b = b
        self.regcoef = regcoef
        self.m = len(b)

    def func(self, x):
        """
        Compute f(x) = 1/m * sum(log(1 + exp(-b_i * a_i^T x))) + regcoef/2 * ||x||^2
        """
        Ax = self.matvec_Ax(x)
        b_Ax = self.b * Ax
        
        # Используем logaddexp для численной устойчивости
        # log(1 + exp(-t)) = logaddexp(0, -t)
        log_loss = np.logaddexp(0, -b_Ax).mean()
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        
        return log_loss + reg_term

    def grad(self, x):
        """
        Compute ∇f(x) = 1/m * A^T * (-b * σ(-b * Ax)) + regcoef * x
        where σ(t) = 1/(1+exp(-t))
        """
        Ax = self.matvec_Ax(x)
        b_Ax = self.b * Ax
        
        # σ(-b * Ax) = 1/(1+exp(b * Ax)) = expit(-b * Ax)
        # Но более устойчиво: σ(-b * Ax) = expit(-b * Ax)
        sigma = expit(-b_Ax)
        
        # -b * σ(-b * Ax)
        weights = -self.b * sigma
        
        grad_loss = self.matvec_ATx(weights) / self.m
        grad_reg = self.regcoef * x
        
        return grad_loss + grad_reg

    def hess(self, x):
        """
        Compute ∇²f(x) = 1/m * A^T * D * A + regcoef * I
        where D = diag(σ(-b * Ax) * (1 - σ(-b * Ax)))
        """
        Ax = self.matvec_Ax(x)
        b_Ax = self.b * Ax
        
        # σ(-b * Ax)
        sigma = expit(-b_Ax)
        
        # D = diag(σ * (1 - σ))
        # Для численной устойчивости: σ(1-σ) = expit(t) * expit(-t)
        D = sigma * (1 - sigma)
        # Альтернативно: D = expit(b_Ax) * expit(-b_Ax)
        
        hess_loss = self.matmat_ATsA(D) / self.m
        hess_reg = self.regcoef * np.eye(len(x))
        
        # Если hess_loss разреженная матрица, нужно соответствующим образом сложить
        if scipy.sparse.issparse(hess_loss):
            return hess_loss + hess_reg
        else:
            return hess_loss + hess_reg


class LogRegL2OptimizedOracle(LogRegL2Oracle):
    """
    Oracle for logistic regression with l2 regularization
    with optimized *_directional methods (are used in line_search).

    For explanation see LogRegL2Oracle.
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self._cache = {}  # Кэш для оптимизации
        
    def _clear_cache(self):
        """Очистка кэша"""
        self._cache.clear()
        
    def func(self, x):
        # Кэшируем Ax для возможного повторного использования
        x_key = tuple(x) if x.size < 100 else None  # Для больших векторов не кэшируем
        
        if x_key not in self._cache:
            self._cache['x'] = x.copy()
            self._cache['Ax'] = self.matvec_Ax(x)
        
        Ax = self._cache['Ax']
        b_Ax = self.b * Ax
        
        log_loss = np.logaddexp(0, -b_Ax).mean()
        reg_term = 0.5 * self.regcoef * np.dot(x, x)
        
        return log_loss + reg_term
        
    def grad(self, x):
        x_key = tuple(x) if x.size < 100 else None
        
        if x_key not in self._cache:
            self._cache['x'] = x.copy()
            self._cache['Ax'] = self.matvec_Ax(x)
            
        Ax = self._cache['Ax']
        b_Ax = self.b * Ax
        sigma = expit(-b_Ax)
        weights = -self.b * sigma
        
        grad_loss = self.matvec_ATx(weights) / self.m
        grad_reg = self.regcoef * x
        
        return grad_loss + grad_reg

    def func_directional(self, x, d, alpha):
        """
        Optimized version with pre-computation of Ax and Ad
        """
        # Ключи для кэша
        x_key = tuple(x) if x.size < 100 else None
        d_key = tuple(d) if d.size < 100 else None
        
        # Получаем или вычисляем Ax
        if x_key not in self._cache or 'Ax' not in self._cache:
            self._cache['x'] = x.copy()
            self._cache['Ax'] = self.matvec_Ax(x)
            
        # Получаем или вычисляем Ad  
        if d_key not in self._cache or 'Ad' not in self._cache:
            self._cache['d'] = d.copy()
            self._cache['Ad'] = self.matvec_Ax(d)
            
        Ax = self._cache['Ax']
        Ad = self._cache['Ad']
        
        # Вычисляем A(x + alpha*d) = Ax + alpha*Ad
        Ax_alpha = Ax + alpha * Ad
        b_Ax_alpha = self.b * Ax_alpha
        
        log_loss = np.logaddexp(0, -b_Ax_alpha).mean()
        x_alpha = x + alpha * d
        reg_term = 0.5 * self.regcoef * np.dot(x_alpha, x_alpha)
        
        return log_loss + reg_term

    def grad_directional(self, x, d, alpha):
        """
        Optimized version with pre-computation of Ax and Ad
        """
        x_key = tuple(x) if x.size < 100 else None
        d_key = tuple(d) if d.size < 100 else None
        
        if x_key not in self._cache or 'Ax' not in self._cache:
            self._cache['x'] = x.copy()
            self._cache['Ax'] = self.matvec_Ax(x)
            
        if d_key not in self._cache or 'Ad' not in self._cache:
            self._cache['d'] = d.copy()
            self._cache['Ad'] = self.matvec_Ax(d)
            
        Ax = self._cache['Ax']
        Ad = self._cache['Ad']
        
        Ax_alpha = Ax + alpha * Ad
        b_Ax_alpha = self.b * Ax_alpha
        sigma = expit(-b_Ax_alpha)
        weights = -self.b * sigma
        
        # ∇f(x+αd)^T d = (A^T * weights/m + regcoef*(x+αd))^T d
        grad_loss_directional = np.dot(weights, Ad) / self.m
        x_alpha = x + alpha * d
        grad_reg_directional = self.regcoef * np.dot(x_alpha, d)
        
        return grad_loss_directional + grad_reg_directional


class CustomLogRegOracle(LogRegL2Oracle):  # НАША МОДИФИКАЦИЯ
    """
    Custom logistic regression oracle with additional features:
    - Elastic net regularization
    - Different loss functions
    """
    def __init__(self, matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef, 
                 l1_ratio=0.0, loss_type='logistic'):
        super().__init__(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)
        self.l1_ratio = l1_ratio  # Для elastic net
        self.loss_type = loss_type  # 'logistic' или 'modified_huber'
        
    def func(self, x):
        Ax = self.matvec_Ax(x)
        b_Ax = self.b * Ax
        
        if self.loss_type == 'logistic':
            loss = np.logaddexp(0, -b_Ax).mean()
        elif self.loss_type == 'modified_huber':
            # Modified Huber loss для большей устойчивости
            loss = np.where(b_Ax >= 1, 0, 
                           np.where(b_Ax <= -1, -4 * b_Ax, (1 - b_Ax)**2)).mean()
        else:
            loss = np.logaddexp(0, -b_Ax).mean()
            
        # Elastic net regularization
        l2_term = 0.5 * self.regcoef * (1 - self.l1_ratio) * np.dot(x, x)
        l1_term = self.regcoef * self.l1_ratio * np.linalg.norm(x, 1)
        
        return loss + l2_term + l1_term


def create_log_reg_oracle(A, b, regcoef, oracle_type='usual'):
    """
    Auxiliary function for creating logistic regression oracles.
        `oracle_type` must be either 'usual' or 'optimized'
    """
    # Определяем функции для матрично-векторных произведений
    if scipy.sparse.issparse(A):
        # Для разреженных матриц
        matvec_Ax = lambda x: A.dot(x)
        matvec_ATx = lambda x: A.T.dot(x)
        
        def matmat_ATsA(s):
            # A^T * diag(s) * A
            if scipy.sparse.issparse(A):
                # Эффективное вычисление для разреженных матриц
                S = scipy.sparse.diags(s)
                return A.T.dot(S.dot(A))
            else:
                return A.T.dot(A * s.reshape(-1, 1))
    else:
        # Для плотных матриц
        matvec_Ax = lambda x: np.dot(A, x)
        matvec_ATx = lambda x: np.dot(A.T, x)
        
        def matmat_ATsA(s):
            # A^T * diag(s) * A
            if A.shape[0] * A.shape[1] > 1000000:  # Для больших матриц
                # Вычисляем поэтапно для экономии памяти
                return np.dot(A.T, A * s.reshape(-1, 1))
            else:
                return np.dot(A.T * s, A)

    if oracle_type == 'usual':
        oracle = LogRegL2Oracle
    elif oracle_type == 'optimized':
        oracle = LogRegL2OptimizedOracle
    elif oracle_type == 'custom':  # НАША МОДИФИКАЦИЯ
        oracle = CustomLogRegOracle
    else:
        raise ValueError('Unknown oracle_type=%s' % oracle_type)
        
    return oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, regcoef)


def grad_finite_diff(func, x, eps=1e-8):
    """
    Returns approximation of the gradient using finite differences:
        result_i := (f(x + eps * e_i) - f(x)) / eps,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    grad = np.zeros(n)
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_plus = func(x + eps * e_i)
        grad[i] = (f_plus - f_x) / eps
        
    return grad


def hess_finite_diff(func, x, eps=1e-5):
    """
    Returns approximation of the Hessian using finite differences:
        result_{ij} := (f(x + eps * e_i + eps * e_j)
                               - f(x + eps * e_i) 
                               - f(x + eps * e_j)
                               + f(x)) / eps^2,
        where e_i are coordinate vectors:
        e_i = (0, 0, ..., 0, 1, 0, ..., 0)
                          >> i <<
    """
    n = len(x)
    hess = np.zeros((n, n))
    f_x = func(x)
    
    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1.0
        f_i = func(x + eps * e_i)
        
        for j in range(n):
            e_j = np.zeros(n)
            e_j[j] = 1.0
            f_j = func(x + eps * e_j)
            f_ij = func(x + eps * e_i + eps * e_j)
            
            hess[i, j] = (f_ij - f_i - f_j + f_x) / (eps * eps)
            
    return hess


def check_oracle_gradients(oracle, x, eps=1e-6, tol=1e-4):
    """
    НАША МОДИФИКАЦИЯ: Функция для проверки корректности градиента и гессиана
    """
    print("=== Checking Oracle Gradients ===")
    
    # Проверка градиента
    analytic_grad = oracle.grad(x)
    finite_diff_grad = grad_finite_diff(oracle.func, x, eps)
    
    grad_error = np.linalg.norm(analytic_grad - finite_diff_grad)
    grad_relative_error = grad_error / (np.linalg.norm(analytic_grad) + 1e-10)
    
    print(f"Gradient error: {grad_error:.2e}")
    print(f"Relative gradient error: {grad_relative_error:.2e}")
    
    if grad_relative_error < tol:
        print("✓ Gradient implementation is CORRECT")
    else:
        print("✗ Gradient implementation might be INCORRECT")
    
    # Проверка гессиана (только для небольших размерностей)
    if len(x) <= 10:
        print("\n=== Checking Oracle Hessian ===")
        analytic_hess = oracle.hess(x)
        finite_diff_hess = hess_finite_diff(oracle.func, x, eps)
        
        hess_error = np.linalg.norm(analytic_hess - finite_diff_hess)
        hess_relative_error = hess_error / (np.linalg.norm(analytic_hess) + 1e-10)
        
        print(f"Hessian error: {hess_error:.2e}")
        print(f"Relative hessian error: {hess_relative_error:.2e}")
        
        if hess_relative_error < tol:
            print("✓ Hessian implementation is CORRECT")
        else:
            print("✗ Hessian implementation might be INCORRECT")
    
    return grad_relative_error < tol


# Тестовые функции для проверки
def test_oracles():
    """
    Test function for our oracle implementations
    """
    print("=== Testing Quadratic Oracle ===")
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    oracle = QuadraticOracle(A, b)
    x_test = np.array([1.0, -1.0])
    
    print(f"f(x) = {oracle.func(x_test):.6f}")
    print(f"∇f(x) = {oracle.grad(x_test)}")
    print(f"∇²f(x) = \n{oracle.hess(x_test)}")
    
    # Проверка конечными разностями
    check_oracle_gradients(oracle, x_test)
    
    print("\n=== Testing Logistic Regression Oracle ===")
    # Простая задача классификации
    A_small = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -1.0]])
    b_small = np.array([1.0, 1.0, -1.0])
    regcoef = 0.1
    
    logreg_oracle = create_log_reg_oracle(A_small, b_small, regcoef)
    x_test_lr = np.array([0.5, -0.5])
    
    print(f"LogReg f(x) = {logreg_oracle.func(x_test_lr):.6f}")
    print(f"LogReg ∇f(x) = {logreg_oracle.grad(x_test_lr)}")
    
    # Проверка конечными разностями
    check_oracle_gradients(logreg_oracle, x_test_lr)
    
    print("\n=== Testing Custom Oracle ===")
    custom_oracle = CustomQuadraticOracle(A, b, c=np.array([0.1, 0.2]), d=0.5)
    print(f"Custom f(x) = {custom_oracle.func(x_test):.6f}")
    print(f"Custom ∇f(x) = {custom_oracle.grad(x_test)}")


if __name__ == "__main__":
    test_oracles()