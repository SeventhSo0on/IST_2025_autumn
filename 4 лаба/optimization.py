import numpy as np
from numpy.linalg import LinAlgError
import scipy
import scipy.linalg
from datetime import datetime
from collections import defaultdict

try:
    from scipy.optimize.linesearch import scalar_search_wolfe2
    SCIPY_WOLFE_AVAILABLE = True
except ImportError:
    try:
        from scipy.optimize import scalar_search_wolfe2
        SCIPY_WOLFE_AVAILABLE = True
    except ImportError:
        SCIPY_WOLFE_AVAILABLE = False
        print("Warning: scalar_search_wolfe2 not available. Using fallback Armijo method for Wolfe conditions.")


class LineSearchTool(object):
    """
    Line search tool for adaptively tuning the step size of the algorithm.

    method : String containing 'Wolfe', 'Armijo' or 'Constant'
        Method of tuning step-size.
        Must be be one of the following strings:
            - 'Wolfe' -- enforce strong Wolfe conditions;
            - 'Armijo" -- adaptive Armijo rule;
            - 'Constant' -- constant step size.
    kwargs :
        Additional parameters of line_search method:

        If method == 'Wolfe':
            c1, c2 : Constants for strong Wolfe conditions
            alpha_0 : Starting point for the backtracking procedure
                to be used in Armijo method in case of failure of Wolfe method.
        If method == 'Armijo':
            c1 : Constant for Armijo rule
            alpha_0 : Starting point for the backtracking procedure.
        If method == 'Constant':
            c : The step size which is returned on every step.
    """
    
    def __init__(self, method='Wolfe', **kwargs):
        self._method = method
        if self._method == 'Wolfe':
            self.c1 = kwargs.get('c1', 1e-4)
            self.c2 = kwargs.get('c2', 0.9)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Armijo':
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
        elif self._method == 'Constant':
            self.c = kwargs.get('c', 1.0)
        elif self._method == 'AdaptiveArmijo':  # НАША МОДИФИКАЦИЯ
            self.c1 = kwargs.get('c1', 1e-4)
            self.alpha_0 = kwargs.get('alpha_0', 1.0)
            self.adaptation_factor = kwargs.get('adaptation_factor', 0.8)
        else:
            raise ValueError('Unknown method {}'.format(method))

    @classmethod
    def from_dict(cls, options):
        if type(options) != dict:
            raise TypeError('LineSearchTool initializer must be of type dict')
        return cls(**options)

    def to_dict(self):
        return self.__dict__

    def line_search(self, oracle, x_k, d_k, previous_alpha=None):
        """
        Finds the step size alpha for a given starting point x_k
        and for a given search direction d_k that satisfies necessary
        conditions for phi(alpha) = oracle.func(x_k + alpha * d_k).

        Parameters
        ----------
        oracle : BaseSmoothOracle-descendant object
            Oracle with .func_directional() and .grad_directional() methods implemented for computing
            function values and its directional derivatives.
        x_k : np.array
            Starting point
        d_k : np.array
            Search direction
        previous_alpha : float or None
            Starting point to use instead of self.alpha_0 to keep the progress from
             previous steps. If None, self.alpha_0, is used as a starting point.

        Returns
        -------
        alpha : float or None if failure
            Chosen step size
        """
        if self._method == 'Constant':
            return self.c
            
        elif self._method == 'Armijo':
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            phi_0 = oracle.func_directional(x_k, d_k, 0.0)
            phi_grad_0 = oracle.grad_directional(x_k, d_k, 0.0)
            
            while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * phi_grad_0:
                alpha /= 2
                if alpha < 1e-14:  # Защита от слишком маленьких шагов
                    break
            return alpha
            
        elif self._method == 'AdaptiveArmijo':  # НАША МОДИФИКАЦИЯ
            alpha = previous_alpha if previous_alpha is not None else self.alpha_0
            phi_0 = oracle.func_directional(x_k, d_k, 0.0)
            phi_grad_0 = oracle.grad_directional(x_k, d_k, 0.0)
            
            # Адаптивное дробление с разными коэффициентами
            iteration = 0
            while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * phi_grad_0:
                if iteration < 5:
                    alpha *= 0.5  # Быстрое уменьшение в начале
                else:
                    alpha *= self.adaptation_factor  # Медленное уменьшение позже
                iteration += 1
                if iteration > 50 or alpha < 1e-14:  # Защита от бесконечного цикла
                    break
            return alpha
            
        elif self._method == 'Wolfe':
            # Если scipy.wolfe недоступен, используем Armijo
            if not SCIPY_WOLFE_AVAILABLE:
                alpha = previous_alpha if previous_alpha is not None else self.alpha_0
                phi_0 = oracle.func_directional(x_k, d_k, 0.0)
                phi_grad_0 = oracle.grad_directional(x_k, d_k, 0.0)
                
                while oracle.func_directional(x_k, d_k, alpha) > phi_0 + self.c1 * alpha * phi_grad_0:
                    alpha /= 2
                    if alpha < 1e-14:
                        break
                return alpha
            
            # Используем scipy Wolfe если доступен
            phi = lambda a: float(oracle.func_directional(x_k, d_k, a))
            phi_grad = lambda a: float(oracle.grad_directional(x_k, d_k, a))
            
            try:
                result = scalar_search_wolfe2(
                    phi, phi_grad, 
                    phi(0.0), phi_grad(0.0),
                    c1=self.c1, c2=self.c2
                )
                
                if result is not None:
                    alpha, phi_alpha, phi_grad_alpha = result
                    if alpha is not None:
                        return alpha
            except:
                pass  # В случае ошибки переходим к Armijo
            
            # Fallback to standard Armijo
            alpha = self.alpha_0
            phi_0 = phi(0.0)
            phi_grad_0 = phi_grad(0.0)
            
            while phi(alpha) > phi_0 + self.c1 * alpha * phi_grad_0:
                alpha /= 2
                if alpha < 1e-14:
                    break
            return alpha


def get_line_search_tool(line_search_options=None):
    if line_search_options:
        if type(line_search_options) is LineSearchTool:
            return line_search_options
        else:
            return LineSearchTool.from_dict(line_search_options)
    else:
        return LineSearchTool()


def gradient_descent(oracle, x_0, tolerance=1e-5, max_iter=10000,
                     line_search_options=None, trace=False, display=False,
                     additional_stopping=False):  # НАША МОДИФИКАЦИЯ
    """
    Gradien descent optimization method with enhanced stopping criteria.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively.
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.
    additional_stopping : bool
        If True, enables additional stopping criterion based on function value stagnation.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        "success" or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'computational_error': in case of getting Infinity or None value during the computations.
            - 'function_stagnation': if function value doesn't improve for several iterations (additional_stopping=True)
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time in seconds passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
            - history['alphas'] : list of step sizes (additional)
            - history['grad_dots'] : list of cosine similarities between consecutive gradients (additional)
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    # Проверка входных данных
    if np.any(np.isnan(x_0)) or np.any(np.isinf(x_0)):
        return x_0, 'computational_error', history
    
    # Initial values
    try:
        grad_0 = oracle.grad(x_0)
        grad_norm_0_sq = np.dot(grad_0, grad_0)
        tolerance_sq = tolerance * grad_norm_0_sq
    except:
        return x_0, 'computational_error', history
    
    # Для дополнительного критерия остановки
    prev_func_value = oracle.func(x_0)
    stagnation_count = 0
    
    # ДОПОЛНИТЕЛЬНОЕ ЛОГИРОВАНИЕ - наша модификация
    if trace:
        history['alphas'] = []  # Сохраняем размеры шагов
        history['grad_dots'] = []  # Сохраняем скалярные произведения градиентов
    
    start_time = datetime.now()
    prev_grad = grad_0
    
    for k in range(max_iter):
        try:
            # Compute gradient at current point
            grad_k = oracle.grad(x_k)
            grad_norm_sq = np.dot(grad_k, grad_k)
            current_func_value = oracle.func(x_k)
        except:
            return x_k, 'computational_error', history
        
        # Store history if needed
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(current_func_value)
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
            
            # Сохраняем косинус угла между последовательными градиентами
            if k > 0:
                norm_prev = np.linalg.norm(prev_grad)
                norm_curr = np.linalg.norm(grad_k)
                if norm_prev > 1e-10 and norm_curr > 1e-10:
                    cos_angle = np.dot(prev_grad, grad_k) / (norm_prev * norm_curr)
                else:
                    cos_angle = 1.0
                history['grad_dots'].append(cos_angle)
            else:
                history['grad_dots'].append(1.0)
        
        # Основной критерий остановки
        if grad_norm_sq <= tolerance_sq:
            return x_k, 'success', history
        
        # ДОПОЛНИТЕЛЬНЫЙ КРИТЕРИЙ ОСТАНОВКИ - наша модификация
        if additional_stopping:
            # Проверяем стагнацию функции
            if abs(prev_func_value - current_func_value) < 1e-12:
                stagnation_count += 1
            else:
                stagnation_count = 0
                
            if stagnation_count >= 5:  # Если 5 итераций без улучшения
                return x_k, 'function_stagnation', history
            
            prev_func_value = current_func_value
        
        if display and k % 100 == 0:  # Выводим каждые 100 итераций
            print(f"Iteration {k}: f(x) = {current_func_value:.6f}, ||grad|| = {np.sqrt(grad_norm_sq):.6f}")
        
        # Compute search direction (negative gradient)
        d_k = -grad_k
        
        # Choose step size
        try:
            alpha_k = line_search_tool.line_search(oracle, x_k, d_k)
        except:
            alpha_k = 1e-8  # Минимальный шаг в случае ошибки
        
        # ДОПОЛНИТЕЛЬНОЕ ЛОГИРОВАНИЕ
        if trace:
            history['alphas'].append(alpha_k)
        
        # Update point
        x_k = x_k + alpha_k * d_k
        prev_grad = grad_k
        
        # Check for computational errors
        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            return x_k, 'computational_error', history
    
    # Если достигли максимального числа итераций
    try:
        grad_final = oracle.grad(x_k)
        grad_norm_final_sq = np.dot(grad_final, grad_final)
    except:
        return x_k, 'computational_error', history
    
    if grad_norm_final_sq <= tolerance_sq:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


def newton(oracle, x_0, tolerance=1e-5, max_iter=100,
           line_search_options=None, trace=False, display=False):
    """
    Newton's optimization method.

    Parameters
    ----------
    oracle : BaseSmoothOracle-descendant object
        Oracle with .func(), .grad() and .hess() methods implemented for computing
        function value, its gradient and Hessian respectively. If the Hessian
        returned by the oracle is not positive-definite method stops with message="newton_direction_error"
    x_0 : np.array
        Starting point for optimization algorithm
    tolerance : float
        Epsilon value for stopping criterion.
    max_iter : int
        Maximum number of iterations.
    line_search_options : dict, LineSearchTool or None
        Dictionary with line search options. See LineSearchTool class for details.
    trace : bool
        If True, the progress information is appended into history dictionary during training.
        Otherwise None is returned instead of history.
    display : bool
        If True, debug information is displayed during optimization.

    Returns
    -------
    x_star : np.array
        The point found by the optimization procedure
    message : string
        'success' or the description of error:
            - 'iterations_exceeded': if after max_iter iterations of the method x_k still doesn't satisfy
                the stopping criterion.
            - 'newton_direction_error': in case of failure of solving linear system with Hessian matrix (e.g. non-invertible matrix).
            - 'computational_error': in case of getting Infinity or None value during the computations.
    history : dictionary of lists or None
        Dictionary containing the progress information or None if trace=False.
        Dictionary has to be organized as follows:
            - history['time'] : list of floats, containing time passed from the start of the method
            - history['func'] : list of function values f(x_k) on every step of the algorithm
            - history['grad_norm'] : list of values Euclidian norms ||g(x_k)|| of the gradient on every step of the algorithm
            - history['x'] : list of np.arrays, containing the trajectory of the algorithm. ONLY STORE IF x.size <= 2
    """
    history = defaultdict(list) if trace else None
    line_search_tool = get_line_search_tool(line_search_options)
    x_k = np.copy(x_0)
    
    # Проверка входных данных
    if np.any(np.isnan(x_0)) or np.any(np.isinf(x_0)):
        return x_0, 'computational_error', history
    
    # Initial values
    try:
        grad_0 = oracle.grad(x_0)
        grad_norm_0_sq = np.dot(grad_0, grad_0)
        tolerance_sq = tolerance * grad_norm_0_sq
    except:
        return x_0, 'computational_error', history
    
    start_time = datetime.now()
    
    for k in range(max_iter):
        try:
            # Compute gradient and Hessian at current point
            grad_k = oracle.grad(x_k)
            hess_k = oracle.hess(x_k)
            grad_norm_sq = np.dot(grad_k, grad_k)
        except:
            return x_k, 'computational_error', history
        
        # Store history if needed
        if trace:
            current_time = (datetime.now() - start_time).total_seconds()
            history['time'].append(current_time)
            history['func'].append(oracle.func(x_k))
            history['grad_norm'].append(np.sqrt(grad_norm_sq))
            if x_k.size <= 2:
                history['x'].append(x_k.copy())
        
        # Check stopping criterion
        if grad_norm_sq <= tolerance_sq:
            return x_k, 'success', history
        
        if display:
            print(f"Iteration {k}: f(x) = {oracle.func(x_k):.6f}, ||grad|| = {np.sqrt(grad_norm_sq):.6f}")
        
        try:
            # Solve Newton system: H_k * d_k = -grad_k using Cholesky
            # For Newton direction we want d_k = -H_k^{-1} * grad_k
            L, lower = scipy.linalg.cho_factor(hess_k)
            d_k = -scipy.linalg.cho_solve((L, lower), grad_k)
        except (LinAlgError, ValueError) as e:
            # Hessian is not positive definite
            return x_k, 'newton_direction_error', history
        
        # Choose step size (for Newton, we prefer to start with alpha=1)
        try:
            alpha_k = line_search_tool.line_search(oracle, x_k, d_k, previous_alpha=1.0)
        except:
            alpha_k = 1e-8
        
        # Update point
        x_k = x_k + alpha_k * d_k
        
        # Check for computational errors
        if np.any(np.isnan(x_k)) or np.any(np.isinf(x_k)):
            return x_k, 'computational_error', history
    
    # If we reached max_iter
    try:
        grad_final = oracle.grad(x_k)
        grad_norm_final_sq = np.dot(grad_final, grad_final)
    except:
        return x_k, 'computational_error', history
    
    if grad_norm_final_sq <= tolerance_sq:
        return x_k, 'success', history
    else:
        return x_k, 'iterations_exceeded', history


# Функция для тестирования наших модификаций
def test_our_optimization_modifications():
    """
    Test function for our custom modifications
    """
    # Квадратичная функция
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    
    class QuadraticOracle:
        def __init__(self, A, b):
            self.A = A
            self.b = b
            
        def func(self, x):
            return 0.5 * np.dot(self.A.dot(x), x) - self.b.dot(x)
            
        def grad(self, x):
            return self.A.dot(x) - self.b
            
        def hess(self, x):
            return self.A
            
        def func_directional(self, x, d, alpha):
            return self.func(x + alpha * d)
            
        def grad_directional(self, x, d, alpha):
            return np.dot(self.grad(x + alpha * d), d)
    
    oracle = QuadraticOracle(A, b)
    x0 = np.array([5.0, 5.0])
    
    print("=== Testing AdaptiveArmijo ===")
    line_search_options = {'method': 'AdaptiveArmijo', 'c1': 0.1, 'adaptation_factor': 0.7}
    
    x_opt, msg, history = gradient_descent(
        oracle, x0, 
        line_search_options=line_search_options,
        trace=True, display=True,
        additional_stopping=True
    )
    
    print(f"Result: {msg}")
    print(f"Optimal point: {x_opt}")
    print(f"Final gradient norm: {np.linalg.norm(oracle.grad(x_opt))}")
    
    return history


if __name__ == "__main__":
    # Запускаем тесты наших модификаций
    history = test_our_optimization_modifications()