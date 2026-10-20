"""Public whole-decode Torch/root envelope for forward-owner release."""
import audit_exact_pool_resources as resources
import compare_forward_owner as comparison


def main():
    previous=resources.comparison.main
    resources.comparison.main=comparison.main
    try:resources.main()
    finally:resources.comparison.main=previous


if __name__=='__main__':main()
