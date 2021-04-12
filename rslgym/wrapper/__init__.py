from .script.RaisimGymVecEnv import RaisimGymVecEnv as VecEnvPython
# import pfrl related only if it is installed
import importlib.util
pfrl_spec = importlib.util.find_spec("pfrl")
found = pfrl_spec is not None
if found:
    print('Found pfrl. importing rslgym version of train_agent_batch_with_evaluation.')
    from .script.pfrl.train_agent_batch import train_agent_batch as train_agent_batch_pfrl
    from .script.pfrl.train_agent_batch import train_agent_batch_with_evaluation as train_agent_batch_with_evaluation_pfrl
    from .script.pfrl.evaluator import eval_performance as eval_performance_pfrl
else:
    print('Cound not find pfrl. Skip importing rslgym version of train_agent_batch_with_evaluation.')
