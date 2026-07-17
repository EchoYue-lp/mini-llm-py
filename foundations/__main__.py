"""Run every prerequisite lesson in the recommended order."""

from foundations.f00_math_basics import run_demo as run_math
from foundations.f01_numpy_basics import run_demo as run_numpy
from foundations.f02_embedding_geometry import run_demo as run_embedding_geometry
from foundations.f03_pandas_basics import run_demo as run_pandas
from foundations.f04_pytorch_tensors import run_demo as run_tensors
from foundations.f05_pytorch_autograd import run_demo as run_autograd
from foundations.f06_pytorch_training import run_demo as run_training


def main() -> None:
    lessons = [
        ("F00 Math", run_math),
        ("F01 NumPy", run_numpy),
        ("F02 Embedding geometry", run_embedding_geometry),
        ("F03 Pandas", run_pandas),
        ("F04 PyTorch tensors", run_tensors),
        ("F05 PyTorch autograd", run_autograd),
        ("F06 PyTorch training", run_training),
    ]
    for title, lesson in lessons:
        print(f"\n=== {title} ===")
        lesson()


if __name__ == "__main__":
    main()
