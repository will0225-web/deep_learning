import inspect
import pandas as pd

def capture_args(func):
    def wrapper(*args, **kwargs):
        # 检查是否需要返回函数信息
        is_need_return_function_info = kwargs.pop('is_need_return_function_info', False)
        # 获取函数名
        func_name = func.__name__
        # 获取函数的所有参数名称
        params = inspect.signature(func).parameters
        
        # 将位置参数和关键字参数合并为一个字典
        all_args = {}
        
        # 添加位置参数
        for i, (key, param) in enumerate(params.items()):
            if i < len(args):
                all_args[key] = args[i]
            elif param.default != param.empty:
                all_args[key] = param.default
                
        # 更新关键字参数
        all_args.update(kwargs)
        # 调用原始函数
        original_result = func(*args, **kwargs)

        # 在这里筛选或处理all_args
        for key, value in all_args.items():
            if isinstance(value, pd.DataFrame) or isinstance(value, pd.Series) or (isinstance(value, (list, tuple)) and len(value) > 10):
                all_args[key] = None  # 将满足条件的参数值替换为None

        if is_need_return_function_info:
        # 返回函数名，参数值和原始函数的结果
            function_info = {
                'func_name': func_name,
                'args': all_args
            }
            return original_result, function_info
        else:
            return original_result
    return wrapper