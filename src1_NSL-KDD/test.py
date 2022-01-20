from pymoo.core.problem import Problem
class MyProblem(Problem):
    def __init__(self,cost_matrix: list, ):
        self.cost_matrix = cost_matrix
        super().__init__(n_var=2,   # 变量数
                         n_obj=2,   # 目标数
                         n_constr=2,    # 约束数
                         xl=np.array([-2, -2]),     # 变量下界
                         xu=np.array([2, 2]),   # 变量上界
                         )

    def _evaluate(self, x, out, *args, **kwargs):

        # 定义目标函数
        f1 = x[:, 0]**2 + x[:, 1]**2    # x1放在x的第0列，x2放在x的第一列
        f2 = (x[:, 0] - 1)**2 + x[:, 1]**2
        # 定义约束条件
        g1 = 2*(x[:, 0] - 0.1) * (x[:, 0] - 0.9) / 0.18
        g2 = -20*(x[:, 0] - 0.4) * (x[:, 0] - 0.6) / 4.8
        # todo
        out["F"] = np.column_stack([f1, f2])
        out["G"] = np.column_stack([g1, g2])


from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.factory import get_sampling, get_crossover, get_mutation
from pymoo.optimize import minimize
from pandas_ml import ConfusionMatrix
# from example import MyProblem
#
# 定义遗传算法
algorithm = NSGA2(
    pop_size=40,
    n_offsprings=10,
    sampling=get_sampling("real_random"),
    crossover=get_crossover("real_sbx", prob=0.9, eta=15),
    mutation=get_mutation("real_pm", eta=20),
    eliminate_duplicates=True
)

res = minimize(MyProblem(),
               algorithm,
               ('n_gen', 40),
               seed=1,
               verbose=True
               )


        
        

