import numpy as np

def convert_numpy_types(obj):
    """Converte tipos numpy para tipos Python compatíveis com JSON"""
    if isinstance(obj, np.generic):
        return obj.item()  # Converte para tipo nativo do Python

    print(f"OBJECTO CONVERTIDO : {obj}")
    return obj
