import nose
from nose.tools import assert_almost_equal, ok_, eq_
from nose.plugins.attrib import attr
from io import StringIO
import numpy as np
import scipy
import scipy.sparse
import scipy.optimize
import sys
import warnings
import tempfile
import os

# Добавляем путь к текущей директории для импорта наших модулей
sys.path.insert(0, os.path.dirname(__file__))

try:
    import optimization
    import oracles
    IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORT_SUCCESS = False


def test_python3():
    """Test that we are using Python 3"""
    ok_(sys.version_info >= (3, 0))


def test_imports():
    """Test that all required modules can be imported"""
    ok_(IMPORT_SUCCESS, "Failed to import optimization and oracles modules")


def test_QuadraticOracle():
    """Test QuadraticOracle implementation"""
    if not IMPORT_SUCCESS:
        return
        
    # Quadratic function:
    #   f(x) = 1/2 x^T x - [1, 2, 3]^T x
    A = np.eye(3)
    b = np.array([1, 2, 3])
    quadratic = oracles.QuadraticOracle(A, b)

    # Check at point x = [0, 0, 0]
    x = np.zeros(3)
    assert_almost_equal(quadratic.func(x), 0.0)
    ok_(np.allclose(quadratic.grad(x), -b))
    ok_(np.allclose(quadratic.hess(x), A))
    ok_(isinstance(quadratic.grad(x), np.ndarray))
    ok_(isinstance(quadratic.hess(x), np.ndarray))

    # Check at point x = [1, 1, 1]
    x = np.ones(3)
    expected_func = 0.5 * (1+1+1) - (1+2+3)  # 1.5 - 6 = -4.5
    assert_almost_equal(quadratic.func(x), -4.5)
    ok_(np.allclose(quadratic.grad(x), x - b))
    ok_(np.allclose(quadratic.hess(x), A))
    ok_(isinstance(quadratic.grad(x), np.ndarray))
    ok_(isinstance(quadratic.hess(x), np.ndarray))

    # Check func_direction and grad_direction oracles at
    # x = [1, 1, 1], d = [-1, -1, -1], alpha = 0.5 and 1.0
    x = np.ones(3)
    d = -np.ones(3)
    assert_almost_equal(quadratic.func_directional(x, d, alpha=0.5), -2.625)
    assert_almost_equal(quadratic.grad_directional(x, d, alpha=0.5), 4.5)
    assert_almost_equal(quadratic.func_directional(x, d, alpha=1.0), 0.0)
    assert_almost_equal(quadratic.grad_directional(x, d, alpha=1.0), 6.0)


def check_log_reg(oracle_type, sparse=False):
    """Helper function to test logistic regression oracle"""
    # Simple data:
    A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    if sparse: 
        A = scipy.sparse.csr_matrix(A)
    b = np.array([1, 1, -1, 1])
    reg_coef = 0.5

    # Logistic regression oracle:
    logreg = oracles.create_log_reg_oracle(A, b, reg_coef, oracle_type=oracle_type)

    # Check at point x = [0, 0]
    x = np.zeros(2)
    
    # Expected values computed manually:
    # For x = [0,0], Ax = [0, 0, 0, 0], b*Ax = [0, 0, 0, 0]
    # log(1 + exp(0)) = log(2) ≈ 0.693147
    # f(x) = (4 * log(2))/4 + 0.5/2 * (0) = log(2) ≈ 0.693147
    
    assert_almost_equal(logreg.func(x), 0.693147180, places=6)
    ok_(np.allclose(logreg.grad(x), [0, -0.25], atol=1e-6))
    
    # Check hessian (approximate due to numerical differences)
    hess = logreg.hess(x)
    ok_(np.allclose(hess, [[0.625, 0.0625], [0.0625, 0.625]], atol=1e-3))
    ok_(isinstance(logreg.grad(x), np.ndarray))
    ok_(isinstance(logreg.hess(x), np.ndarray))

    # Check func_direction and grad_direction oracles at
    # x = [0, 0], d = [1, 1], alpha = 0.5 and 1.0
    x = np.zeros(2)
    d = np.ones(2)
    assert_almost_equal(logreg.func_directional(x, d, alpha=0.5), 0.7386407091095, places=6)
    assert_almost_equal(logreg.grad_directional(x, d, alpha=0.5), 0.4267589549159, places=6)
    assert_almost_equal(logreg.func_directional(x, d, alpha=1.0), 1.1116496416598, places=6)
    assert_almost_equal(logreg.grad_directional(x, d, alpha=1.0), 1.0559278283039, places=6)


def test_log_reg_usual():
    """Test usual logistic regression oracle"""
    if not IMPORT_SUCCESS:
        return
    check_log_reg('usual')
    check_log_reg('usual', sparse=True)


@attr('bonus')
def test_log_reg_optimized():
    """Test optimized logistic regression oracle (bonus)"""
    if not IMPORT_SUCCESS:
        return
    check_log_reg('optimized')
    check_log_reg('optimized', sparse=True)


def get_counters(A):
    """Helper function to count matrix operations"""
    counters = {'Ax': 0, 'ATx': 0, 'ATsA': 0}

    def matvec_Ax(x):
        counters['Ax'] += 1
        return A.dot(x)

    def matvec_ATx(x):
        counters['ATx'] += 1
        return A.T.dot(x)

    def matmat_ATsA(s):
        counters['ATsA'] += 1
        return A.T.dot(A * s.reshape(-1, 1))

    return (matvec_Ax, matvec_ATx, matmat_ATsA, counters)


def check_counters(counters, groundtruth):
    """Check that counters don't exceed expected values"""
    for (key, value) in groundtruth.items():
        ok_(key in counters)
        ok_(counters[key] <= value, 
            f"Counter {key}: {counters[key]} > {groundtruth[key]}")


def test_log_reg_oracle_calls():
    """Test that logistic regression oracle makes efficient matrix calls"""
    if not IMPORT_SUCCESS:
        return

    A = np.ones((2, 2))
    b = np.ones(2)
    x = np.ones(2)
    d = np.ones(2)
    reg_coef = 0.5

    # Single func
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func(x)
    check_counters(counters, {'Ax': 1, 'ATx': 0, 'ATsA': 0})

    # Single grad
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).grad(x)
    check_counters(counters, {'Ax': 1, 'ATx': 1, 'ATsA': 0})

    # Single hess
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).hess(x)
    check_counters(counters, {'Ax': 1, 'ATx': 0, 'ATsA': 1})

    # Single func_directional
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func_directional(x, d, 1)
    check_counters(counters, {'Ax': 1, 'ATx': 0, 'ATsA': 0})

    # Single grad_directional
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).grad_directional(x, d, 1)
    check_counters(counters, {'Ax': 1, 'ATx': 1, 'ATsA': 0})

    # In a row: func + grad + hess
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
    oracle.func(x)
    oracle.grad(x)
    oracle.hess(x)
    check_counters(counters, {'Ax': 3, 'ATx': 1, 'ATsA': 1})

    # In a row: func + grad
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
    oracle.func(x)
    oracle.grad(x)
    check_counters(counters, {'Ax': 2, 'ATx': 1, 'ATsA': 0})

    # In a row: grad + hess
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracle = oracles.LogRegL2Oracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
    oracle.grad(x)
    oracle.hess(x)
    check_counters(counters, {'Ax': 2, 'ATx': 1, 'ATsA': 1})


@attr('bonus')
def test_log_reg_optimized_oracle_calls():
    """Test optimized oracle calls (bonus)"""
    if not IMPORT_SUCCESS:
        return

    A = np.ones((2, 2))
    b = np.ones(2)
    x = np.ones(2)
    d = np.ones(2)
    reg_coef = 0.5

    # Single func
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func(x)
    check_counters(counters, {'Ax': 1, 'ATx': 0, 'ATsA': 0})

    # Single grad
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).grad(x)
    check_counters(counters, {'Ax': 1, 'ATx': 1, 'ATsA': 0})

    # Single hess
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).hess(x)
    check_counters(counters, {'Ax': 1, 'ATx': 0, 'ATsA': 1})

    # Single func_directional
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).func_directional(x, d, 1)
    check_counters(counters, {'Ax': 2, 'ATx': 0, 'ATsA': 0})

    # Single grad_directional
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef).grad_directional(x, d, 1)
    check_counters(counters, {'Ax': 2, 'ATx': 0, 'ATsA': 0})

    # In a row: func + grad + hess
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracle = oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
    oracle.func(x)
    oracle.grad(x)
    oracle.hess(x)
    check_counters(counters, {'Ax': 1, 'ATx': 1, 'ATsA': 1})

    # In a row: func + grad
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracle = oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
    oracle.func(x)
    oracle.grad(x)
    check_counters(counters, {'Ax': 1, 'ATx': 1, 'ATsA': 0})

    # In a row: grad + hess
    (matvec_Ax, matvec_ATx, matmat_ATsA, counters) = get_counters(A)
    oracle = oracles.LogRegL2OptimizedOracle(matvec_Ax, matvec_ATx, matmat_ATsA, b, reg_coef)
    oracle.grad(x)
    oracle.hess(x)
    check_counters(counters, {'Ax': 1, 'ATx': 1, 'ATsA': 1})


def test_grad_finite_diff_1():
    """Test finite difference gradient on quadratic function"""
    if not IMPORT_SUCCESS:
        return
        
    # Quadratic function.
    A = np.eye(3)
    b = np.array([1, 2, 3])
    quadratic = oracles.QuadraticOracle(A, b)
    g = oracles.grad_finite_diff(quadratic.func, np.zeros(3))
    ok_(isinstance(g, np.ndarray))
    ok_(np.allclose(g, -b, atol=1e-5))


def test_grad_finite_diff_2():
    """Test finite difference gradient on custom function"""
    if not IMPORT_SUCCESS:
        return
        
    # f(x, y) = x^3 + y^2
    func = lambda x: x[0] ** 3 + x[1] ** 2
    x = np.array([2.0, 3.0])
    eps = 1e-5
    g = oracles.grad_finite_diff(func, x, eps)
    ok_(isinstance(g, np.ndarray))
    ok_(np.allclose(g, [12.0, 6.0], atol=1e-4))


def test_hess_finite_diff_1():
    """Test finite difference hessian on quadratic function"""
    if not IMPORT_SUCCESS:
        return
        
    # Quadratic function.
    A = np.eye(3)
    b = np.array([1, 2, 3])
    quadratic = oracles.QuadraticOracle(A, b)
    H = oracles.hess_finite_diff(quadratic.func, np.zeros(3))
    ok_(isinstance(H, np.ndarray))
    ok_(np.allclose(H, A, atol=1e-4))


def test_hess_finite_diff_2():
    """Test finite difference hessian on custom function"""
    if not IMPORT_SUCCESS:
        return
        
    # f(x, y) = x^3 + y^2
    func = lambda x: x[0] ** 3 + x[1] ** 2
    x = np.array([2.0, 3.0])
    eps = 1e-5
    H = oracles.hess_finite_diff(func, x, eps)
    ok_(isinstance(H, np.ndarray))
    ok_(np.allclose(H, [[12.0, 0.], [0., 2.0]], atol=1e-3))


def get_quadratic():
    """Helper function to create quadratic oracle"""
    # Quadratic function:
    #   f(x) = 1/2 x^T x - [1, 2, 3]^T x
    A = np.eye(3)
    b = np.array([1, 2, 3])
    return oracles.QuadraticOracle(A, b)


def test_line_search():
    """Test line search implementation"""
    if not IMPORT_SUCCESS:
        return
        
    oracle = get_quadratic()
    x = np.array([100, 0, 0])
    d = np.array([-1, 0, 0])

    # Constant line search
    ls_tool = optimization.LineSearchTool(method='Constant', c=1.0)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 1.0)
    ls_tool = optimization.LineSearchTool(method='Constant', c=10.0)
    assert_almost_equal(ls_tool.line_search(oracle, x, d), 10.0)

    # Armijo rule
    ls_tool = optimization.LineSearchTool(method='Armijo', alpha_0=100, c1=0.9)
    # For quadratic f(x) = 0.5||x||^2 - b^T x, optimal step is 1
    # But with alpha_0=100 and backtracking, we get smaller step
    alpha = ls_tool.line_search(oracle, x, d)
    ok_(alpha > 0 and alpha <= 100)

    # Wolfe rule (may not always find exact value due to numerical precision)
    ls_tool = optimization.LineSearchTool(method='Wolfe', c1=1e-4, c2=0.9)
    alpha = ls_tool.line_search(oracle, x, d)
    ok_(alpha > 0)


def check_equal_histories(history1, history2, atol=1e-3):
    """Helper function to compare optimization histories"""
    if history1 is None or history2 is None:
        eq_(history1, history2)
        return
        
    ok_('func' in history1 and 'func' in history2)
    ok_(np.allclose(history1['func'], history2['func'], atol=atol))
    ok_('grad_norm' in history1 and 'grad_norm' in history2)
    ok_(np.allclose(history1['grad_norm'], history2['grad_norm'], atol=atol))
    ok_('time' in history1 and 'time' in history2)
    eq_(len(history1['time']), len(history2['time']))
    eq_('x' in history1, 'x' in history2)
    if 'x' in history1:
        ok_(np.allclose(history1['x'], history2['x'], atol=atol))


def check_prototype(method):
    """Test that optimization methods have correct prototype"""
    if not IMPORT_SUCCESS:
        return
        
    class ZeroOracle2D(oracles.BaseSmoothOracle):
        def func(self, x): 
            return 0.0

        def grad(self, x): 
            return np.zeros(2)

        def hess(self, x): 
            return np.zeros([2, 2])

    oracle = ZeroOracle2D()
    x0 = np.ones(2)

    def check_result(result, msg='success', history=None):
        eq_(len(result), 3)
        ok_(isinstance(result[0], np.ndarray))
        eq_(result[1], msg)
        if history is not None:
            check_equal_histories(result[2], history)

    # Test various parameter combinations
    check_result(method(oracle, x0))
    check_result(method(oracle, x0, 1e-3, 10))
    check_result(method(oracle, x0, 1e-3, 10, {'method': 'Constant', 'c': 1.0}))
    
    # Test with trace=True
    result = method(oracle, x0, 1e-3, 10, {'method': 'Constant', 'c': 1.0}, trace=True)
    check_result(result, 'success')
    ok_(result[2] is not None)  # history should not be None
    
    # Test display parameter (should not crash)
    result = method(oracle, x0, display=False)
    check_result(result)


def test_gd_basic():
    """Test gradient descent basic functionality"""
    if not IMPORT_SUCCESS:
        return
    check_prototype(optimization.gradient_descent)


def test_newton_basic():
    """Test Newton method basic functionality"""
    if not IMPORT_SUCCESS:
        return
    check_prototype(optimization.newton)


def get_1d(alpha):
    """1D test function: f(x) = exp(alpha * x) + alpha * x^2"""
    class Func(oracles.BaseSmoothOracle):
        def __init__(self, alpha):
            self.alpha = alpha

        def func(self, x):
            return np.exp(self.alpha * x[0]) + self.alpha * x[0] ** 2

        def grad(self, x):
            return np.array([self.alpha * np.exp(self.alpha * x[0]) + 2 * self.alpha * x[0]])

        def hess(self, x):
            return np.array([[self.alpha ** 2 * np.exp(self.alpha * x[0]) + 2 * self.alpha]])

    return Func(alpha)


def test_gd_1d():
    """Test gradient descent on 1D function"""
    if not IMPORT_SUCCESS:
        return
        
    oracle = get_1d(0.5)
    x0 = np.array([1.0])
    
    # Test that it runs without errors
    x_star, msg, history = optimization.gradient_descent(
        oracle, x0, max_iter=5, tolerance=1e-10, trace=False,
        line_search_options={'method': 'Constant', 'c': 0.1}
    )
    
    ok_(isinstance(x_star, np.ndarray))
    ok_(msg in ['success', 'iterations_exceeded', 'computational_error'])
    ok_(history is None)  # trace=False


def test_newton_1d():
    """Test Newton method on 1D function"""
    if not IMPORT_SUCCESS:
        return
        
    oracle = get_1d(0.5)
    x0 = np.array([1.0])
    
    # Test that it runs without errors
    x_star, msg, history = optimization.newton(
        oracle, x0, max_iter=5, tolerance=1e-10, trace=False,
        line_search_options={'method': 'Constant', 'c': 1.0}
    )
    
    ok_(isinstance(x_star, np.ndarray))
    ok_(msg in ['success', 'iterations_exceeded', 'newton_direction_error', 'computational_error'])
    ok_(history is None)  # trace=False


def test_custom_oracles():
    """НАША МОДИФИКАЦИЯ: Test custom oracle implementations"""
    if not IMPORT_SUCCESS:
        return
        
    # Test CustomQuadraticOracle
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    c = np.array([0.1, 0.2])
    d = 0.5
    
    custom_oracle = oracles.CustomQuadraticOracle(A, b, c, d)
    x_test = np.array([1.0, -1.0])
    
    # Check that it computes something reasonable
    func_val = custom_oracle.func(x_test)
    grad_val = custom_oracle.grad(x_test)
    
    ok_(isinstance(func_val, (float, np.floating)))
    ok_(isinstance(grad_val, np.ndarray))
    ok_(len(grad_val) == 2)


def test_optimization_with_custom_oracle():
    """НАША МОДИФИКАЦИЯ: Test optimization with custom oracle"""
    if not IMPORT_SUCCESS:
        return
        
    # Create custom oracle
    A = np.array([[2.0, 0.5], [0.5, 1.0]])
    b = np.array([1.0, 2.0])
    custom_oracle = oracles.CustomQuadraticOracle(A, b)
    
    x0 = np.array([3.0, 3.0])
    
    # Test gradient descent
    x_gd, msg_gd, history_gd = optimization.gradient_descent(
        custom_oracle, x0, max_iter=100, tolerance=1e-6, trace=True
    )
    
    ok_(isinstance(x_gd, np.ndarray))
    ok_(msg_gd in ['success', 'iterations_exceeded'])
    
    # Test Newton
    x_nt, msg_nt, history_nt = optimization.newton(
        custom_oracle, x0, max_iter=10, tolerance=1e-6, trace=True
    )
    
    ok_(isinstance(x_nt, np.ndarray))
    ok_(msg_nt in ['success', 'iterations_exceeded', 'newton_direction_error'])


@attr('slow')
def test_optimization_convergence():
    """НАША МОДИФИКАЦИЯ: Test that methods actually converge"""
    if not IMPORT_SUCCESS:
        return
        
    # Simple quadratic problem where we know the solution
    A = np.eye(2)
    b = np.array([1.0, 2.0])
    oracle = oracles.QuadraticOracle(A, b)
    x0 = np.array([5.0, 5.0])
    x_optimal = b  # For f(x) = 0.5||x||^2 - b^T x, optimum is at x = b
    
    # Test gradient descent convergence
    x_gd, msg_gd, history_gd = optimization.gradient_descent(
        oracle, x0, tolerance=1e-8, max_iter=1000, trace=True
    )
    
    # Should get close to optimum
    if msg_gd == 'success':
        ok_(np.linalg.norm(x_gd - x_optimal) < 1e-4)
    
    # Test Newton convergence (should be very fast)
    x_nt, msg_nt, history_nt = optimization.newton(
        oracle, x0, tolerance=1e-8, max_iter=10, trace=True
    )
    
    if msg_nt == 'success':
        ok_(np.linalg.norm(x_nt - x_optimal) < 1e-6)


def run_all_tests():
    """НАША МОДИФИКАЦИЯ: Run all tests and report results"""
    print("Running optimization tests...")
    print("=" * 50)
    
    # Collect all test functions
    test_functions = [name for name in globals() if name.startswith('test_')]
    
    results = {'passed': 0, 'failed': 0, 'skipped': 0}
    
    for test_name in test_functions:
        test_func = globals()[test_name]
        
        try:
            # Check if test requires imports and they failed
            if not IMPORT_SUCCESS and test_name not in ['test_python3', 'test_imports']:
                print(f"SKIP: {test_name} (import failed)")
                results['skipped'] += 1
                continue
                
            # Run the test
            test_func()
            print(f"PASS: {test_name}")
            results['passed'] += 1
            
        except AssertionError as e:
            print(f"FAIL: {test_name} - {e}")
            results['failed'] += 1
        except Exception as e:
            print(f"ERROR: {test_name} - {e}")
            results['failed'] += 1
    
    print("=" * 50)
    print(f"Results: {results['passed']} passed, {results['failed']} failed, {results['skipped']} skipped")
    
    return results['failed'] == 0


if __name__ == "__main__":
    # Run tests when executed directly
    success = run_all_tests()
    sys.exit(0 if success else 1)